#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pymbar import timeseries
from scipy.stats import t as student_t


METRICS = {
    "rmsd_backbone_A": "RMSD",
    "radius_gyration_A": "Rg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta equilibración y construye targets estadísticos."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directorio con los CSV de observables básicos.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directorio de salida.",
    )
    parser.add_argument(
        "--nskip",
        type=int,
        default=10,
        help="Separación entre candidatos a t0 durante la detección.",
    )
    return parser.parse_args()


def validate_series(values: np.ndarray, label: str) -> None:
    if values.ndim != 1:
        raise ValueError(f"{label}: la serie no es unidimensional.")

    if len(values) < 10:
        raise ValueError(f"{label}: hay menos de 10 observaciones.")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: hay valores no finitos.")


def detect_metric_equilibration(
    values: np.ndarray,
    nskip: int,
) -> dict[str, float | int]:
    t0, g, neff = timeseries.detect_equilibration(
        values,
        fast=False,
        nskip=nskip,
    )

    return {
        "t0_index": int(t0),
        "g": float(g),
        "neff": float(neff),
    }


def equilibrium_statistics(
    values: np.ndarray,
) -> dict[str, float | int]:
    validate_series(values, "tramo equilibrado")

    g = float(
        timeseries.statistical_inefficiency(
            values,
            fast=False,
        )
    )

    g = max(g, 1.0)
    n = int(len(values))
    neff = float(n / g)

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    sem = float(std / math.sqrt(neff))

    degrees_freedom = max(neff - 1.0, 1.0)
    critical_value = float(student_t.ppf(0.975, degrees_freedom))
    half_width = critical_value * sem

    return {
        "n_frames": n,
        "g": g,
        "neff": neff,
        "mean_A": mean,
        "std_A": std,
        "sem_A": sem,
        "ci95_low_A": mean - half_width,
        "ci95_high_A": mean + half_width,
    }


def process_replica(
    input_dir: Path,
    replica: int,
    nskip: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    input_csv = (
        input_dir
        / f"1k5n_A_R{replica}_basic_observables.csv"
    )

    if not input_csv.is_file():
        raise FileNotFoundError(f"No existe: {input_csv}")

    dataframe = pd.read_csv(input_csv)

    required_columns = {"time_ns", *METRICS.keys()}
    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en {input_csv}: {sorted(missing)}"
        )

    detections: dict[str, dict[str, float | int]] = {}

    for column, short_name in METRICS.items():
        values = dataframe[column].to_numpy(dtype=float)
        validate_series(values, f"R{replica} {short_name}")

        detection = detect_metric_equilibration(
            values=values,
            nskip=nskip,
        )

        detection["metric"] = short_name
        detection["column"] = column
        detections[column] = detection

    common_t0 = max(
        int(detection["t0_index"])
        for detection in detections.values()
    )

    if common_t0 >= len(dataframe) - 10:
        raise RuntimeError(
            f"R{replica}: el tramo equilibrado común es demasiado corto."
        )

    common_t0_ns = float(dataframe["time_ns"].iloc[common_t0])
    rows: list[dict[str, object]] = []

    for column, short_name in METRICS.items():
        values = dataframe[column].to_numpy(dtype=float)
        equilibrated = values[common_t0:]

        statistics = equilibrium_statistics(equilibrated)
        individual_t0 = int(detections[column]["t0_index"])

        row: dict[str, object] = {
            "system": "1k5n_A",
            "replica": replica,
            "metric": short_name,
            "column": column,
            "n_total_frames": int(len(values)),
            "dt_ps": float(
                1000.0
                * (
                    dataframe["time_ns"].iloc[1]
                    - dataframe["time_ns"].iloc[0]
                )
            ),
            "individual_t0_index": individual_t0,
            "individual_t0_ns": float(
                dataframe["time_ns"].iloc[individual_t0]
            ),
            "individual_detection_g": float(
                detections[column]["g"]
            ),
            "individual_detection_neff": float(
                detections[column]["neff"]
            ),
            "common_t0_index": common_t0,
            "common_t0_ns": common_t0_ns,
            **statistics,
        }

        rows.append(row)

    replica_summary = {
        "system": "1k5n_A",
        "replica": replica,
        "common_t0_index": common_t0,
        "common_t0_ns": common_t0_ns,
        "last_time_ns": float(dataframe["time_ns"].iloc[-1]),
        "equilibrated_duration_ns": float(
            dataframe["time_ns"].iloc[-1] - common_t0_ns
        ),
    }

    return rows, replica_summary


def build_replicate_noise_floor(
    targets: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for metric, group in targets.groupby("metric", sort=True):
        replica_means = group["mean_A"].to_numpy(dtype=float)

        records.append(
            {
                "system": "1k5n_A",
                "metric": metric,
                "n_replicas": int(len(replica_means)),
                "replicate_mean_A": float(np.mean(replica_means)),
                "replicate_std_A": float(
                    np.std(replica_means, ddof=1)
                ),
                "replicate_sem_A": float(
                    np.std(replica_means, ddof=1)
                    / math.sqrt(len(replica_means))
                ),
                "replicate_min_A": float(np.min(replica_means)),
                "replicate_max_A": float(np.max(replica_means)),
                "replicate_range_A": float(
                    np.max(replica_means)
                    - np.min(replica_means)
                ),
            }
        )

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows: list[dict[str, object]] = []
    replica_summaries: list[dict[str, object]] = []

    for replica in (1, 2, 3):
        rows, summary = process_replica(
            input_dir=input_dir,
            replica=replica,
            nskip=args.nskip,
        )

        target_rows.extend(rows)
        replica_summaries.append(summary)

    targets = pd.DataFrame(target_rows)
    summaries = pd.DataFrame(replica_summaries)
    noise_floor = build_replicate_noise_floor(targets)

    targets_csv = output_dir / "equilibrium_targets.csv"
    summaries_csv = output_dir / "equilibration_summary.csv"
    noise_floor_csv = output_dir / "replicate_noise_floor.csv"
    json_file = output_dir / "equilibrium_targets.json"

    targets.to_csv(targets_csv, index=False)
    summaries.to_csv(summaries_csv, index=False)
    noise_floor.to_csv(noise_floor_csv, index=False)

    json_file.write_text(
        json.dumps(
            {
                "targets": target_rows,
                "replicas": replica_summaries,
                "noise_floor": noise_floor.to_dict(
                    orient="records"
                ),
                "method": {
                    "detect_equilibration_fast": False,
                    "detect_equilibration_nskip": args.nskip,
                    "common_t0_rule": (
                        "maximum individual t0 across RMSD and Rg"
                    ),
                    "neff_definition": "N / statistical_inefficiency",
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("EQUILIBRATION TARGETS")
    print("=" * 80)
    print(
        targets[
            [
                "replica",
                "metric",
                "individual_t0_ns",
                "common_t0_ns",
                "n_frames",
                "g",
                "neff",
                "mean_A",
                "sem_A",
                "ci95_low_A",
                "ci95_high_A",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("REPLICATE NOISE FLOOR")
    print("=" * 80)
    print(noise_floor.to_string(index=False))

    print()
    print(f"Targets:     {targets_csv}")
    print(f"Réplicas:    {summaries_csv}")
    print(f"Noise floor: {noise_floor_csv}")
    print(f"JSON:        {json_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
