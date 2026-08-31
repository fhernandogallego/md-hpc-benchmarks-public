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


SS_COLUMNS = {
    "helix_fraction": "helix",
    "strand_fraction": "strand",
    "coil_fraction": "coil",
    "structured_fraction": "structured",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye targets de SASA y estructura secundaria "
            "usando una ventana composicional común."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--replicas", default="1,2,3")
    parser.add_argument("--nskip", type=int, default=1)
    parser.add_argument("--min-duration-ns", type=float, default=20.0)
    parser.add_argument("--min-neff", type=float, default=10.0)
    parser.add_argument("--tail-start-ns", type=float, default=80.0)
    return parser.parse_args()


def parse_replicas(value: str) -> list[int]:
    replicas = sorted(
        {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )

    if not replicas:
        raise ValueError(
            "No se proporcionaron réplicas."
        )

    if any(replica < 1 for replica in replicas):
        raise ValueError(
            "Los índices de réplica deben ser positivos."
        )

    return replicas


def validate_series(values: np.ndarray, label: str) -> None:
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{label}: la serie no es unidimensional.")

    if len(values) < 10:
        raise ValueError(f"{label}: hay menos de 10 valores.")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: hay valores no finitos.")


def detect(values: np.ndarray, nskip: int) -> tuple[int, float, float]:
    validate_series(values, "serie")

    if float(np.ptp(values)) < 1e-12:
        return 0, 1.0, float(len(values))

    t0, g, neff = timeseries.detect_equilibration(
        values,
        fast=False,
        nskip=nskip,
    )

    return int(t0), max(float(g), 1.0), float(neff)


def statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    validate_series(values, "ventana estadística")

    n = int(len(values))

    if float(np.ptp(values)) < 1e-12:
        g = 1.0
    else:
        g = max(
            float(
                timeseries.statistical_inefficiency(
                    values,
                    fast=False,
                )
            ),
            1.0,
        )

    neff = float(n / g)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    sem = float(std / math.sqrt(neff))

    degrees_freedom = max(neff - 1.0, 1.0)
    critical = float(student_t.ppf(0.975, degrees_freedom))
    half_width = critical * sem

    return {
        "n_frames": n,
        "g": g,
        "neff": neff,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def status(duration_valid: bool, neff_valid: bool) -> str:
    if duration_valid and neff_valid:
        return "VALID"

    if not duration_valid and not neff_valid:
        return "INVALID_SHORT_WINDOW_AND_LOW_NEFF"

    if not duration_valid:
        return "INVALID_SHORT_WINDOW"

    return "INVALID_LOW_NEFF"


def process_sasa(
    dataframe: pd.DataFrame,
    system: str,
    replica: int,
    nskip: int,
    min_duration_ns: float,
    min_neff: float,
    tail_start_ns: float,
) -> dict[str, object]:
    times_ns = dataframe["time_ns"].to_numpy(dtype=float)
    values = dataframe["total_sasa_A2"].to_numpy(dtype=float)

    t0, detection_g, detection_neff = detect(values, nskip)
    t0_ns = float(times_ns[t0])
    duration_ns = float(times_ns[-1] - t0_ns)

    auto_stats = statistics(values[t0:])
    tail_stats = statistics(
        values[times_ns >= tail_start_ns - 1e-9]
    )

    duration_valid = duration_ns >= min_duration_ns
    neff_valid = float(auto_stats["neff"]) >= min_neff
    target_valid = bool(duration_valid and neff_valid)

    return {
        "system": system,
        "replica": replica,
        "metric": "SASA",
        "source_column": "total_sasa_A2",
        "unit": "A2",
        "window_type": "independent",
        "t0_index": t0,
        "t0_ns": t0_ns,
        "equilibrated_duration_ns": duration_ns,
        "detection_g": detection_g,
        "detection_neff": detection_neff,
        "auto_n_frames": auto_stats["n_frames"],
        "auto_g": auto_stats["g"],
        "auto_neff": auto_stats["neff"],
        "auto_mean": auto_stats["mean"],
        "auto_std": auto_stats["std"],
        "auto_sem": auto_stats["sem"],
        "auto_ci95_low": auto_stats["ci95_low"],
        "auto_ci95_high": auto_stats["ci95_high"],
        "tail_start_ns": tail_start_ns,
        "tail_mean": tail_stats["mean"],
        "tail_neff": tail_stats["neff"],
        "auto_minus_tail": (
            float(auto_stats["mean"])
            - float(tail_stats["mean"])
        ),
        "duration_valid": duration_valid,
        "neff_valid": neff_valid,
        "target_valid": target_valid,
        "target_status": status(duration_valid, neff_valid),
        "target_mean": (
            auto_stats["mean"] if target_valid else float("nan")
        ),
        "target_sem": (
            auto_stats["sem"] if target_valid else float("nan")
        ),
    }


def process_secondary_structure(
    dataframe: pd.DataFrame,
    system: str,
    replica: int,
    nskip: int,
    min_duration_ns: float,
    min_neff: float,
    tail_start_ns: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    times_ns = dataframe["time_ns"].to_numpy(dtype=float)

    detections: dict[str, dict[str, float | int]] = {}

    # Solo H y E son targets independientes.
    for column in ("helix_fraction", "strand_fraction"):
        values = dataframe[column].to_numpy(dtype=float)
        t0, g, neff = detect(values, nskip)

        detections[column] = {
            "t0_index": t0,
            "t0_ns": float(times_ns[t0]),
            "g": g,
            "neff": neff,
        }

    common_t0_index = max(
        int(detections["helix_fraction"]["t0_index"]),
        int(detections["strand_fraction"]["t0_index"]),
    )
    common_t0_ns = float(times_ns[common_t0_index])
    duration_ns = float(times_ns[-1] - common_t0_ns)

    duration_valid = duration_ns >= min_duration_ns
    metric_records: list[dict[str, object]] = []
    metric_statistics: dict[str, dict[str, float | int]] = {}

    tail_mask = times_ns >= tail_start_ns - 1e-9

    for column, short_name in SS_COLUMNS.items():
        values = dataframe[column].to_numpy(dtype=float)

        auto_stats = statistics(values[common_t0_index:])
        tail_stats = statistics(values[tail_mask])

        metric_statistics[column] = auto_stats

        metric_records.append(
            {
                "system": system,
                "replica": replica,
                "metric": short_name,
                "source_column": column,
                "unit": "fraction",
                "window_type": "shared_secondary_structure",
                "individual_t0_ns": (
                    detections[column]["t0_ns"]
                    if column in detections
                    else float("nan")
                ),
                "common_t0_index": common_t0_index,
                "t0_ns": common_t0_ns,
                "equilibrated_duration_ns": duration_ns,
                "auto_n_frames": auto_stats["n_frames"],
                "auto_g": auto_stats["g"],
                "auto_neff": auto_stats["neff"],
                "auto_mean": auto_stats["mean"],
                "auto_std": auto_stats["std"],
                "auto_sem": auto_stats["sem"],
                "auto_ci95_low": auto_stats["ci95_low"],
                "auto_ci95_high": auto_stats["ci95_high"],
                "tail_start_ns": tail_start_ns,
                "tail_mean": tail_stats["mean"],
                "tail_neff": tail_stats["neff"],
                "auto_minus_tail": (
                    float(auto_stats["mean"])
                    - float(tail_stats["mean"])
                ),
            }
        )

    helix_neff_valid = (
        float(metric_statistics["helix_fraction"]["neff"])
        >= min_neff
    )
    strand_neff_valid = (
        float(metric_statistics["strand_fraction"]["neff"])
        >= min_neff
    )

    group_neff_valid = bool(
        helix_neff_valid and strand_neff_valid
    )
    group_valid = bool(duration_valid and group_neff_valid)
    group_status = status(duration_valid, group_neff_valid)

    for record in metric_records:
        record["duration_valid"] = duration_valid
        record["neff_valid"] = bool(
            float(record["auto_neff"]) >= min_neff
        )
        record["secondary_structure_group_valid"] = group_valid
        record["target_valid"] = group_valid
        record["target_status"] = group_status
        record["target_mean"] = (
            record["auto_mean"]
            if group_valid
            else float("nan")
        )
        record["target_sem"] = (
            record["auto_sem"]
            if group_valid
            else float("nan")
        )

    composition = {
        "system": system,
        "replica": replica,
        "helix_individual_t0_ns": detections[
            "helix_fraction"
        ]["t0_ns"],
        "strand_individual_t0_ns": detections[
            "strand_fraction"
        ]["t0_ns"],
        "common_t0_ns": common_t0_ns,
        "duration_ns": duration_ns,
        "helix_neff": metric_statistics[
            "helix_fraction"
        ]["neff"],
        "strand_neff": metric_statistics[
            "strand_fraction"
        ]["neff"],
        "coil_neff": metric_statistics[
            "coil_fraction"
        ]["neff"],
        "structured_neff": metric_statistics[
            "structured_fraction"
        ]["neff"],
        "duration_valid": duration_valid,
        "primary_neff_valid": group_neff_valid,
        "target_valid": group_valid,
        "target_status": group_status,
        "composition_sum": float(
            metric_statistics["helix_fraction"]["mean"]
            + metric_statistics["strand_fraction"]["mean"]
            + metric_statistics["coil_fraction"]["mean"]
        ),
        "structured_sum": float(
            metric_statistics["helix_fraction"]["mean"]
            + metric_statistics["strand_fraction"]["mean"]
        ),
        "structured_direct": float(
            metric_statistics["structured_fraction"]["mean"]
        ),
    }

    return metric_records, composition


def build_wide(
    targets: pd.DataFrame,
    compositions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for replica in sorted(
        targets["replica"].astype(int).unique()
    ):
        subset = targets.loc[
            targets["replica"] == replica
        ].set_index("source_column")

        composition = compositions.loc[
            compositions["replica"] == replica
        ].iloc[0]

        sasa = subset.loc["total_sasa_A2"]
        helix = subset.loc["helix_fraction"]
        strand = subset.loc["strand_fraction"]
        coil = subset.loc["coil_fraction"]
        structured = subset.loc["structured_fraction"]

        records.append(
            {
                "system": str(subset["system"].iloc[0]),
                "replica": replica,
                "sasa_target_A2": sasa["target_mean"],
                "sasa_target_sem_A2": sasa["target_sem"],
                "sasa_target_valid": bool(sasa["target_valid"]),
                "sasa_target_status": sasa["target_status"],
                "sasa_t0_ns": sasa["t0_ns"],
                "sasa_neff": sasa["auto_neff"],
                "helix_target": helix["target_mean"],
                "helix_target_sem": helix["target_sem"],
                "strand_target": strand["target_mean"],
                "strand_target_sem": strand["target_sem"],
                "coil_target": coil["target_mean"],
                "structured_target": structured["target_mean"],
                "secondary_structure_target_valid": bool(
                    composition["target_valid"]
                ),
                "secondary_structure_target_status": (
                    composition["target_status"]
                ),
                "secondary_structure_t0_ns": (
                    composition["common_t0_ns"]
                ),
                "secondary_structure_duration_ns": (
                    composition["duration_ns"]
                ),
                "helix_neff": composition["helix_neff"],
                "strand_neff": composition["strand_neff"],
            }
        )

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_records: list[dict[str, object]] = []
    composition_records: list[dict[str, object]] = []

    for replica in parse_replicas(args.replicas):
        input_csv = (
            input_dir
            / (
                f"{args.system}_R{replica}"
                "_sasa_dssp_100ps.csv"
            )
        )
        dataframe = pd.read_csv(input_csv)

        if "system" in dataframe.columns:
            observed_systems = set(
                dataframe["system"]
                .astype(str)
                .unique()
            )

            if observed_systems != {args.system}:
                raise RuntimeError(
                    f"{input_csv}: sistemas observados "
                    f"{sorted(observed_systems)}; "
                    f"esperado {args.system}."
                )

        required = {
            "time_ns",
            "total_sasa_A2",
            *SS_COLUMNS.keys(),
        }
        missing = required.difference(dataframe.columns)

        if missing:
            raise RuntimeError(
                f"Faltan columnas en {input_csv}: {sorted(missing)}"
            )

        target_records.append(
            process_sasa(
                dataframe=dataframe,
                system=args.system,
                replica=replica,
                nskip=args.nskip,
                min_duration_ns=args.min_duration_ns,
                min_neff=args.min_neff,
                tail_start_ns=args.tail_start_ns,
            )
        )

        ss_records, composition = process_secondary_structure(
            dataframe=dataframe,
            system=args.system,
            replica=replica,
            nskip=args.nskip,
            min_duration_ns=args.min_duration_ns,
            min_neff=args.min_neff,
            tail_start_ns=args.tail_start_ns,
        )

        target_records.extend(ss_records)
        composition_records.append(composition)

    targets = pd.DataFrame(target_records)
    compositions = pd.DataFrame(composition_records)
    wide = build_wide(targets, compositions)

    long_csv = output_dir / "structural_targets_long.csv"
    wide_csv = output_dir / "structural_model_targets_wide.csv"
    composition_csv = (
        output_dir / "secondary_structure_composition.csv"
    )
    metadata_json = output_dir / "structural_targets.json"

    targets.to_csv(long_csv, index=False)
    wide.to_csv(wide_csv, index=False)
    compositions.to_csv(composition_csv, index=False)

    metadata_json.write_text(
        json.dumps(
            {
                "sasa_window": "independent equilibrium t0",
                "secondary_structure_window": (
                    "shared max(t0_helix, t0_strand)"
                ),
                "primary_secondary_structure_targets": [
                    "helix_fraction",
                    "strand_fraction",
                ],
                "derived_same_window_observables": [
                    "coil_fraction",
                    "structured_fraction",
                ],
                "min_duration_ns": args.min_duration_ns,
                "min_neff": args.min_neff,
                "nskip": args.nskip,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 110)
    print("TARGETS ESTRUCTURALES CORREGIDOS")
    print("=" * 110)
    print(
        wide.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 110)
    print("VENTANAS COMPOSICIONALES")
    print("=" * 110)
    print(
        compositions.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"Targets largos: {long_csv}")
    print(f"Targets modelo: {wide_csv}")
    print(f"Composición:    {composition_csv}")
    print(f"Metadatos:      {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
