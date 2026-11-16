#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

EXPECTED_METADATA = "1899e3121864ced9e7b50bcf9a4ee5f7bd7614202af035a7f9a45cd442ad6181"
EXPECTED_SPLIT = "d31ef5c7d5fcea49e4f6414ea8bb70254ee5c63008b8c02707ab462c31e24bee"

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def valid_flag(value):
    text = value.strip().lower()
    if text not in ("true", "false", "1", "0"):
        raise ValueError(f"target_valid inesperado: {value!r}")
    return text in ("true", "1")

def recalculate(times, values, t0_ns, original_mean, inefficiency):
    expected = np.arange(10001, dtype=np.float64) * 0.01
    if (times.shape != (10001,) or values.shape != (10001,)
            or not np.allclose(times, expected, rtol=0, atol=1e-7)
            or not np.isfinite(values).all()):
        raise ValueError("Se esperaba RMSD finito en malla 0-100 ns cada 10 ps")
    if not math.isfinite(t0_ns) or not 0 <= t0_ns <= 100:
        raise ValueError(f"t0 original inválido: {t0_ns}")

    original_index = int(np.searchsorted(times, t0_ns - 1e-7, side="left"))
    if original_index >= len(times) or abs(times[original_index] - t0_ns) > 1e-7:
        raise ValueError(f"t0 no coincide con la malla: {t0_ns}")
    recovered = float(np.mean(values[original_index:]))
    if not math.isfinite(original_mean) or abs(recovered - original_mean) > 1e-6:
        raise ValueError(f"La serie no reproduce el target original: {recovered} vs {original_mean}")

    first_after_prefix = int(np.searchsorted(times, 20.0 + 1e-8, side="right"))
    start = max(original_index, first_after_prefix)
    if start >= len(times):
        raise ValueError("No quedan frames posteriores al prefijo")
    segment = values[start:]
    duration_ns = float(times[-1] - times[start])
    g = max(float(inefficiency(segment, fast=False)), 1.0) if len(segment) > 1 else math.nan
    neff = float(len(segment) / g) if math.isfinite(g) else 0.0
    new_valid = bool(duration_ns >= 20.0 - 1e-8 and neff >= 10.0 - 1e-8)

    return {
        "original_mean_delta_A": recovered - original_mean,
        "new_t0_ns": float(times[start]),
        "future_duration_ns": duration_ns,
        "future_n_frames": len(segment),
        "future_g": g,
        "future_neff": neff,
        "future_valid": new_valid,
        "future_target_A": float(np.mean(segment)) if new_valid else math.nan,
    }

def main():
    parser = argparse.ArgumentParser()
    for name in ("inventory", "split", "metadata", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.metadata) != EXPECTED_METADATA or sha(args.split) != EXPECTED_SPLIT:
        raise ValueError("Han cambiado los hashes del metadata o split ATLAS80")

    from pymbar import timeseries

    with args.split.open(newline="", encoding="utf-8") as handle:
        train = {r["system_id"] for r in csv.DictReader(handle) if r["split"] == "train"}
    sources = read_csv(args.inventory)
    if len(train) != 62 or len(sources) != 62 or {r["system"] for r in sources} != train:
        raise ValueError("El inventario resuelto no coincide con las 62 proteínas de train")

    records = []
    source_hashes = {}
    for source in sources:
        system = source["system"]
        master = Path(source["master_targets_csv"])
        if not master.is_file():
            raise FileNotFoundError(master)
        master_rows = [r for r in read_csv(master)
                       if r["system"] == system and r["target_name"] == "rmsd"]
        targets = {int(r["replica"]): r for r in master_rows}
        if len(master_rows) != 3 or set(targets) != {1, 2, 3}:
            raise ValueError(f"Filas RMSD incompletas: {system}")

        source_hashes[system] = {
            "master_csv": str(master),
            "master_sha256": sha(master),
            "replicas": {},
        }
        for replica in (1, 2, 3):
            path = Path(source[f"basic_R{replica}_csv"])
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, usecols=["time_ns", "rmsd_backbone_A"])
            old = targets[replica]
            result = recalculate(
                frame["time_ns"].to_numpy(dtype=np.float64),
                frame["rmsd_backbone_A"].to_numpy(dtype=np.float64),
                float(old["t0_ns"]),
                float(old["automatic_window_value"]),
                timeseries.statistical_inefficiency,
            )
            old_valid = valid_flag(old["target_valid"])
            if old_valid and abs(float(old["target_value"])
                                 - float(old["automatic_window_value"])) > 1e-6:
                raise ValueError(f"Target original inconsistente: {system} R{replica}")
            source_hashes[system]["replicas"][str(replica)] = {
                "path": str(path), "sha256": sha(path),
            }
            records.append({
                "system_id": system,
                "replica": replica,
                "sample_id": f"{system}_R{replica}",
                "original_t0_ns": float(old["t0_ns"]),
                "original_valid": old_valid,
                "original_target_A": float(old["target_value"]) if old_valid else math.nan,
                **result,
                "common_valid": old_valid and result["future_valid"],
            })

    if len(records) != 186 or len({r["sample_id"] for r in records}) != 186:
        raise ValueError("Se esperaban 186 réplicas de train sin duplicados")
    original = [r for r in records if r["original_valid"]]
    future = [r for r in records if r["future_valid"]]
    common = [r for r in records if r["common_valid"]]
    if len(original) != 93 or len({r["system_id"] for r in original}) != 53:
        raise ValueError("La máscara RMSD original no coincide con 93 réplicas / 53 proteínas")

    summary = {
        "scope": "RMSD train only; original t0 retained; t > 20 ns and t >= t0",
        "n_systems": 62,
        "n_replicas": 186,
        "original_valid_replicas": len(original),
        "future_valid_replicas": len(future),
        "common_valid_replicas": len(common),
        "original_valid_proteins": len({r["system_id"] for r in original}),
        "future_valid_proteins": len({r["system_id"] for r in future}),
        "common_valid_proteins": len({r["system_id"] for r in common}),
        "lost_from_original": len(original) - len(common),
        "gained_vs_original": len(future) - len(common),
        "input_sha256": {
            "inventory": sha(args.inventory),
            "split": sha(args.split),
            "metadata": sha(args.metadata),
            "code": sha(Path(__file__)),
        },
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "source_hashes": source_hashes,
    }

    args.output.mkdir(parents=True)
    with (args.output / "train_future_rmsd_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("FUTURE_RMSD_TRAIN_TARGETS=PASS")
    for key in ("original_valid_replicas", "future_valid_replicas", "common_valid_replicas",
                "original_valid_proteins", "future_valid_proteins", "common_valid_proteins",
                "lost_from_original", "gained_vs_original"):
        print(f"{key}={summary[key]}")
    print(f"ROWS={args.output / 'train_future_rmsd_rows.csv'}")

if __name__ == "__main__":
    main()
