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
    "total_sasa_A2": {
        "name": "SASA",
        "unit": "A2",
        "prefix": "sasa",
        "primary_target": True,
    },
    "helix_fraction": {
        "name": "helix_fraction",
        "unit": "fraction",
        "prefix": "helix",
        "primary_target": True,
    },
    "strand_fraction": {
        "name": "strand_fraction",
        "unit": "fraction",
        "prefix": "strand",
        "primary_target": True,
    },
    "coil_fraction": {
        "name": "coil_fraction",
        "unit": "fraction",
        "prefix": "coil",
        "primary_target": False,
    },
    "structured_fraction": {
        "name": "structured_fraction",
        "unit": "fraction",
        "prefix": "structured",
        "primary_target": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye targets equilibrados de SASA y "
            "estructura secundaria."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nskip", type=int, default=1)
    parser.add_argument("--min-duration-ns", type=float, default=20.0)
    parser.add_argument("--min-neff", type=float, default=10.0)
    parser.add_argument("--tail-start-ns", type=float, default=80.0)
    return parser.parse_args()


def validate_series(
    values: np.ndarray,
    label: str,
) -> None:
    if values.ndim != 1:
        raise ValueError(f"{label}: la serie no es unidimensional.")

    if len(values) < 10:
        raise ValueError(f"{label}: hay menos de 10 observaciones.")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: contiene valores no finitos.")


def detect_equilibration(
    values: np.ndarray,
    nskip: int,
) -> tuple[int, float, float]:
    validate_series(values, "serie de equilibración")

    # Protección para series prácticamente constantes.
    if float(np.ptp(values)) < 1e-12:
        return 0, 1.0, float(len(values))

    t0, g, neff = timeseries.detect_equilibration(
        values,
        fast=False,
        nskip=nskip,
    )

    t0 = int(t0)
    g = max(float(g), 1.0)
    neff = float(neff)

    if not np.isfinite(g) or not np.isfinite(neff):
        raise RuntimeError(
            "PyMBAR produjo valores no finitos."
        )

    return t0, g, neff


def calculate_statistics(
    values: np.ndarray,
) -> dict[str, float | int]:
    validate_series(values, "ventana estadística")

    n = int(len(values))

    if float(np.ptp(values)) < 1e-12:
        g = 1.0
    else:
        g = float(
            timeseries.statistical_inefficiency(
                values,
                fast=False,
            )
        )
        g = max(g, 1.0)

    neff = float(n / g)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    sem = float(std / math.sqrt(neff))

    degrees_freedom = max(neff - 1.0, 1.0)
    critical = float(
        student_t.ppf(0.975, degrees_freedom)
    )
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


def validation_status(
    duration_valid: bool,
    neff_valid: bool,
) -> str:
    if duration_valid and neff_valid:
        return "VALID"

    if not duration_valid and not neff_valid:
        return "INVALID_SHORT_WINDOW_AND_LOW_NEFF"

    if not duration_valid:
        return "INVALID_SHORT_WINDOW"

    return "INVALID_LOW_NEFF"


def check_consistency(
    dataframe: pd.DataFrame,
    replica: int,
) -> dict[str, object]:
    composition_sum = (
        dataframe["helix_fraction"]
        + dataframe["strand_fraction"]
        + dataframe["coil_fraction"]
    )

    structured_sum = (
        dataframe["helix_fraction"]
        + dataframe["strand_fraction"]
    )

    max_composition_error = float(
        np.max(np.abs(composition_sum - 1.0))
    )
    max_structured_error = float(
        np.max(
            np.abs(
                structured_sum
                - dataframe["structured_fraction"]
            )
        )
    )

    return {
        "system": "1k5n_A",
        "replica": replica,
        "n_frames": int(len(dataframe)),
        "start_ns": float(dataframe["time_ns"].iloc[0]),
        "end_ns": float(dataframe["time_ns"].iloc[-1]),
        "max_HEC_sum_error": max_composition_error,
        "max_structured_sum_error": max_structured_error,
        "composition_valid": bool(
            max_composition_error < 1e-10
        ),
        "structured_valid": bool(
            max_structured_error < 1e-10
        ),
    }


def process_metric(
    dataframe: pd.DataFrame,
    replica: int,
    column: str,
    nskip: int,
    min_duration_ns: float,
    min_neff: float,
    tail_start_ns: float,
) -> dict[str, object]:
    metadata = METRICS[column]

    times_ns = dataframe["time_ns"].to_numpy(
        dtype=float
    )
    values = dataframe[column].to_numpy(
        dtype=float
    )

    validate_series(
        values,
        f"R{replica} {metadata['name']}",
    )

    t0, detection_g, detection_neff = (
        detect_equilibration(
            values,
            nskip=nskip,
        )
    )

    t0_ns = float(times_ns[t0])
    final_time_ns = float(times_ns[-1])
    duration_ns = float(final_time_ns - t0_ns)

    auto_values = values[t0:]
    auto_stats = calculate_statistics(auto_values)

    tail_mask = times_ns >= tail_start_ns - 1e-9
    tail_values = values[tail_mask]

    if len(tail_values) < 10:
        raise RuntimeError(
            f"R{replica} {column}: ventana final vacía."
        )

    tail_stats = calculate_statistics(tail_values)

    duration_valid = duration_ns >= min_duration_ns
    neff_valid = auto_stats["neff"] >= min_neff
    target_valid = bool(
        duration_valid and neff_valid
    )

    status = validation_status(
        duration_valid=duration_valid,
        neff_valid=neff_valid,
    )

    auto_mean = float(auto_stats["mean"])
    tail_mean = float(tail_stats["mean"])

    absolute_difference = auto_mean - tail_mean

    if abs(auto_mean) > 1e-12:
        relative_difference = (
            absolute_difference / abs(auto_mean)
        )
    else:
        relative_difference = float("nan")

    return {
        "system": "1k5n_A",
        "replica": replica,
        "metric": metadata["name"],
        "source_column": column,
        "unit": metadata["unit"],
        "primary_target": metadata["primary_target"],
        "canonical_resolution_ps": 100.0,
        "canonical_nskip": nskip,
        "t0_index": t0,
        "t0_ns": t0_ns,
        "final_time_ns": final_time_ns,
        "equilibrated_duration_ns": duration_ns,
        "detection_g": detection_g,
        "detection_neff": detection_neff,
        "auto_n_frames": auto_stats["n_frames"],
        "auto_g": auto_stats["g"],
        "auto_neff": auto_stats["neff"],
        "auto_mean": auto_mean,
        "auto_std": auto_stats["std"],
        "auto_sem": auto_stats["sem"],
        "auto_ci95_low": auto_stats["ci95_low"],
        "auto_ci95_high": auto_stats["ci95_high"],
        "tail_start_ns": tail_start_ns,
        "tail_n_frames": tail_stats["n_frames"],
        "tail_g": tail_stats["g"],
        "tail_neff": tail_stats["neff"],
        "tail_mean": tail_mean,
        "tail_std": tail_stats["std"],
        "tail_sem": tail_stats["sem"],
        "auto_minus_tail": absolute_difference,
        "auto_minus_tail_relative": relative_difference,
        "min_duration_ns": min_duration_ns,
        "min_neff": min_neff,
        "duration_valid": duration_valid,
        "neff_valid": neff_valid,
        "target_valid": target_valid,
        "target_status": status,
        "target_mean": (
            auto_mean if target_valid else float("nan")
        ),
        "target_sem": (
            auto_stats["sem"]
            if target_valid
            else float("nan")
        ),
        "target_ci95_low": (
            auto_stats["ci95_low"]
            if target_valid
            else float("nan")
        ),
        "target_ci95_high": (
            auto_stats["ci95_high"]
            if target_valid
            else float("nan")
        ),
    }


def build_wide_table(
    targets: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    primary_columns = {
        column: metadata
        for column, metadata in METRICS.items()
        if metadata["primary_target"]
    }

    for replica, group in targets.groupby(
        "replica",
        sort=True,
    ):
        record: dict[str, object] = {
            "system": "1k5n_A",
            "replica": int(replica),
        }

        indexed = group.set_index("source_column")

        for column, metadata in primary_columns.items():
            row = indexed.loc[column]
            prefix = metadata["prefix"]

            record[f"{prefix}_target"] = row[
                "target_mean"
            ]
            record[f"{prefix}_target_sem"] = row[
                "target_sem"
            ]
            record[f"{prefix}_target_valid"] = bool(
                row["target_valid"]
            )
            record[f"{prefix}_target_status"] = row[
                "target_status"
            ]
            record[f"{prefix}_t0_ns"] = row[
                "t0_ns"
            ]
            record[f"{prefix}_duration_ns"] = row[
                "equilibrated_duration_ns"
            ]
            record[f"{prefix}_neff"] = row[
                "auto_neff"
            ]
            record[f"{prefix}_tail_80_100"] = row[
                "tail_mean"
            ]

        # Valores derivados solo cuando hélice y hebra son válidos.
        helix_valid = bool(
            record["helix_target_valid"]
        )
        strand_valid = bool(
            record["strand_target_valid"]
        )

        if helix_valid and strand_valid:
            helix = float(record["helix_target"])
            strand = float(record["strand_target"])

            record["coil_target_derived"] = (
                1.0 - helix - strand
            )
            record["structured_target_derived"] = (
                helix + strand
            )
            record["secondary_structure_target_valid"] = True
        else:
            record["coil_target_derived"] = float("nan")
            record["structured_target_derived"] = float("nan")
            record["secondary_structure_target_valid"] = False

        records.append(record)

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_records: list[dict[str, object]] = []
    consistency_records: list[dict[str, object]] = []

    for replica in (1, 2, 3):
        input_csv = (
            input_dir
            / f"1k5n_A_R{replica}_sasa_dssp_100ps.csv"
        )

        dataframe = pd.read_csv(input_csv)

        required = {
            "time_ns",
            *METRICS.keys(),
        }
        missing = required.difference(
            dataframe.columns
        )

        if missing:
            raise RuntimeError(
                f"Faltan columnas en {input_csv}: "
                f"{sorted(missing)}"
            )

        if not dataframe["time_ns"].is_monotonic_increasing:
            raise RuntimeError(
                f"R{replica}: el tiempo no es creciente."
            )

        consistency_records.append(
            check_consistency(
                dataframe=dataframe,
                replica=replica,
            )
        )

        for column in METRICS:
            target_records.append(
                process_metric(
                    dataframe=dataframe,
                    replica=replica,
                    column=column,
                    nskip=args.nskip,
                    min_duration_ns=args.min_duration_ns,
                    min_neff=args.min_neff,
                    tail_start_ns=args.tail_start_ns,
                )
            )

    targets = pd.DataFrame(target_records)
    consistency = pd.DataFrame(
        consistency_records
    )
    model_targets = build_wide_table(targets)

    long_csv = (
        output_dir
        / "structural_targets_long.csv"
    )
    wide_csv = (
        output_dir
        / "structural_model_targets_wide.csv"
    )
    consistency_csv = (
        output_dir
        / "structural_target_consistency.csv"
    )
    metadata_json = (
        output_dir
        / "structural_targets.json"
    )

    targets.to_csv(long_csv, index=False)
    model_targets.to_csv(wide_csv, index=False)
    consistency.to_csv(
        consistency_csv,
        index=False,
    )

    metadata_json.write_text(
        json.dumps(
            {
                "primary_targets": [
                    "total_sasa_A2",
                    "helix_fraction",
                    "strand_fraction",
                ],
                "derived_targets": {
                    "coil_fraction": (
                        "1 - helix_fraction - strand_fraction"
                    ),
                    "structured_fraction": (
                        "helix_fraction + strand_fraction"
                    ),
                },
                "resolution_ps": 100.0,
                "nskip": args.nskip,
                "min_duration_ns": args.min_duration_ns,
                "min_neff": args.min_neff,
                "tail_start_ns": args.tail_start_ns,
                "records": target_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 120)
    print("TARGETS SASA Y ESTRUCTURA SECUNDARIA")
    print("=" * 120)

    print(
        targets[
            [
                "replica",
                "metric",
                "unit",
                "t0_ns",
                "equilibrated_duration_ns",
                "auto_neff",
                "auto_mean",
                "tail_mean",
                "auto_minus_tail",
                "target_valid",
                "target_status",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("TARGETS PRIMARIOS PARA EL MODELO")
    print("=" * 120)
    print(
        model_targets.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("Resumen de validez:")
    print(
        targets.groupby(
            [
                "metric",
                "target_status",
            ]
        ).size().to_string()
    )

    print()
    print("Consistencia H/E/C:")
    print(consistency.to_string(index=False))

    print()
    print(f"Targets largos: {long_csv}")
    print(f"Targets modelo: {wide_csv}")
    print(f"Consistencia:   {consistency_csv}")
    print(f"JSON:           {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
