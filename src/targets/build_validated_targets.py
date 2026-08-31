#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pymbar import timeseries
from scipy.stats import t as student_t


METRICS = {
    "rmsd_backbone_A": {
        "name": "RMSD",
        "prefix": "rmsd",
    },
    "radius_gyration_A": {
        "name": "Rg",
        "prefix": "rg",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye targets por observable usando t0 independiente, "
            "control de duración, Neff y ventana final de sensibilidad."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--sensitivity-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--replicas", default="1,2,3")
    parser.add_argument("--nskip", type=int, default=10)
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
    if values.ndim != 1:
        raise ValueError(f"{label}: la serie no es unidimensional.")

    if len(values) < 10:
        raise ValueError(f"{label}: hay menos de 10 valores.")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: hay valores no finitos.")


def calculate_statistics(values: np.ndarray) -> dict[str, float | int]:
    validate_series(values, "ventana estadística")

    n_frames = int(len(values))

    g = float(
        timeseries.statistical_inefficiency(
            values,
            fast=False,
        )
    )
    g = max(g, 1.0)

    neff = float(n_frames / g)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    sem = float(std / math.sqrt(neff))

    degrees_freedom = max(neff - 1.0, 1.0)
    critical = float(student_t.ppf(0.975, degrees_freedom))
    half_width = critical * sem

    return {
        "n_frames": n_frames,
        "g": g,
        "neff": neff,
        "mean_A": mean,
        "std_A": std,
        "sem_A": sem,
        "ci95_low_A": mean - half_width,
        "ci95_high_A": mean + half_width,
    }


def quality_status(
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


def sensitivity_information(
    sensitivity: pd.DataFrame,
    replica: int,
    metric: str,
) -> dict[str, float]:
    subset = sensitivity.loc[
        (sensitivity["replica"] == replica)
        & (sensitivity["metric"] == metric)
    ].copy()

    if subset.empty:
        raise RuntimeError(
            f"No hay sensibilidad para R{replica}, {metric}."
        )

    ten_ps = subset.loc[subset["resolution"] == "10ps"].copy()

    if ten_ps.empty:
        raise RuntimeError(
            f"No hay sensibilidad a 10 ps para R{replica}, {metric}."
        )

    canonical_100ps = subset.loc[
        (subset["resolution"] == "100ps")
        & (subset["nskip"] == 10)
    ]

    result = {
        "sensitivity_10ps_t0_min_ns": float(ten_ps["t0_ns"].min()),
        "sensitivity_10ps_t0_max_ns": float(ten_ps["t0_ns"].max()),
        "sensitivity_10ps_t0_range_ns": float(
            ten_ps["t0_ns"].max() - ten_ps["t0_ns"].min()
        ),
        "sensitivity_10ps_mean_min_A": float(
            ten_ps["equilibrated_mean_A"].min()
        ),
        "sensitivity_10ps_mean_max_A": float(
            ten_ps["equilibrated_mean_A"].max()
        ),
        "sensitivity_10ps_mean_range_A": float(
            ten_ps["equilibrated_mean_A"].max()
            - ten_ps["equilibrated_mean_A"].min()
        ),
    }

    if canonical_100ps.empty:
        result.update(
            {
                "canonical_100ps_t0_ns": float("nan"),
                "canonical_100ps_mean_A": float("nan"),
            }
        )
    else:
        row = canonical_100ps.iloc[0]
        result.update(
            {
                "canonical_100ps_t0_ns": float(row["t0_ns"]),
                "canonical_100ps_mean_A": float(
                    row["equilibrated_mean_A"]
                ),
            }
        )

    return result


def process_metric(
    dataframe: pd.DataFrame,
    sensitivity: pd.DataFrame,
    system: str,
    replica: int,
    column: str,
    nskip: int,
    min_duration_ns: float,
    min_neff: float,
    tail_start_ns: float,
) -> dict[str, object]:
    metadata = METRICS[column]
    metric = metadata["name"]

    times_ns = dataframe["time_ns"].to_numpy(dtype=float)
    values = dataframe[column].to_numpy(dtype=float)

    validate_series(values, f"R{replica} {metric}")

    t0, detection_g, detection_neff = (
        timeseries.detect_equilibration(
            values,
            fast=False,
            nskip=nskip,
        )
    )

    t0 = int(t0)
    t0_ns = float(times_ns[t0])
    final_time_ns = float(times_ns[-1])
    duration_ns = float(final_time_ns - t0_ns)

    auto_values = values[t0:]
    auto_statistics = calculate_statistics(auto_values)

    tail_mask = times_ns >= tail_start_ns - 1e-9

    if int(np.sum(tail_mask)) < 10:
        raise RuntimeError(
            f"R{replica} {metric}: ventana final demasiado corta."
        )

    tail_values = values[tail_mask]
    tail_statistics = calculate_statistics(tail_values)

    duration_valid = duration_ns >= min_duration_ns
    neff_valid = auto_statistics["neff"] >= min_neff
    is_valid = bool(duration_valid and neff_valid)

    status = quality_status(
        duration_valid=duration_valid,
        neff_valid=neff_valid,
    )

    sensitivity_data = sensitivity_information(
        sensitivity=sensitivity,
        replica=replica,
        metric=metric,
    )

    target_mean = (
        auto_statistics["mean_A"]
        if is_valid
        else float("nan")
    )
    target_sem = (
        auto_statistics["sem_A"]
        if is_valid
        else float("nan")
    )
    target_ci95_low = (
        auto_statistics["ci95_low_A"]
        if is_valid
        else float("nan")
    )
    target_ci95_high = (
        auto_statistics["ci95_high_A"]
        if is_valid
        else float("nan")
    )

    return {
        "system": system,
        "replica": replica,
        "metric": metric,
        "source_column": column,
        "canonical_resolution_ps": 10.0,
        "canonical_nskip": nskip,
        "t0_index": t0,
        "t0_ns": t0_ns,
        "final_time_ns": final_time_ns,
        "equilibrated_duration_ns": duration_ns,
        "detection_g": float(detection_g),
        "detection_neff": float(detection_neff),
        "auto_n_frames": auto_statistics["n_frames"],
        "auto_g": auto_statistics["g"],
        "auto_neff": auto_statistics["neff"],
        "auto_mean_A": auto_statistics["mean_A"],
        "auto_std_A": auto_statistics["std_A"],
        "auto_sem_A": auto_statistics["sem_A"],
        "auto_ci95_low_A": auto_statistics["ci95_low_A"],
        "auto_ci95_high_A": auto_statistics["ci95_high_A"],
        "min_duration_ns": min_duration_ns,
        "min_neff": min_neff,
        "duration_valid": duration_valid,
        "neff_valid": neff_valid,
        "target_valid": is_valid,
        "target_status": status,
        "target_mean_A": target_mean,
        "target_sem_A": target_sem,
        "target_ci95_low_A": target_ci95_low,
        "target_ci95_high_A": target_ci95_high,
        "tail_start_ns": tail_start_ns,
        "tail_n_frames": tail_statistics["n_frames"],
        "tail_g": tail_statistics["g"],
        "tail_neff": tail_statistics["neff"],
        "tail_mean_A": tail_statistics["mean_A"],
        "tail_std_A": tail_statistics["std_A"],
        "tail_sem_A": tail_statistics["sem_A"],
        "auto_minus_tail_A": (
            auto_statistics["mean_A"]
            - tail_statistics["mean_A"]
        ),
        **sensitivity_data,
    }


def build_wide_table(targets: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for (system, replica), group in targets.groupby(
        ["system", "replica"],
        sort=True,
    ):
        record: dict[str, object] = {
            "system": system,
            "replica": int(replica),
        }

        for _, row in group.iterrows():
            prefix = (
                "rmsd"
                if row["metric"] == "RMSD"
                else "rg"
            )

            record[f"{prefix}_target_A"] = row["target_mean_A"]
            record[f"{prefix}_target_sem_A"] = row["target_sem_A"]
            record[f"{prefix}_target_valid"] = bool(
                row["target_valid"]
            )
            record[f"{prefix}_target_status"] = row["target_status"]
            record[f"{prefix}_t0_ns"] = row["t0_ns"]
            record[f"{prefix}_duration_ns"] = row[
                "equilibrated_duration_ns"
            ]
            record[f"{prefix}_neff"] = row["auto_neff"]
            record[f"{prefix}_tail_80_100_mean_A"] = row[
                "tail_mean_A"
            ]

        records.append(record)

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    sensitivity_csv = args.sensitivity_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity = pd.read_csv(sensitivity_csv)

    required_sensitivity = {
        "system",
        "replica",
        "metric",
        "resolution",
        "nskip",
        "t0_ns",
        "equilibrated_mean_A",
    }
    missing_sensitivity = (
        required_sensitivity.difference(
            sensitivity.columns
        )
    )

    if missing_sensitivity:
        raise RuntimeError(
            "Faltan columnas en el análisis de sensibilidad: "
            f"{sorted(missing_sensitivity)}"
        )

    sensitivity_systems = set(
        sensitivity["system"].astype(str).unique()
    )

    if sensitivity_systems != {args.system}:
        raise RuntimeError(
            "El CSV de sensibilidad no corresponde al sistema "
            f"{args.system}: {sorted(sensitivity_systems)}"
        )

    records: list[dict[str, object]] = []

    for replica in parse_replicas(args.replicas):
        input_csv = (
            input_dir
            / (
                f"{args.system}_R{replica}"
                "_basic_observables.csv"
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

        required = {"time_ns", *METRICS.keys()}
        missing = required.difference(dataframe.columns)

        if missing:
            raise RuntimeError(
                f"Faltan columnas en {input_csv}: {sorted(missing)}"
            )

        for column in METRICS:
            records.append(
                process_metric(
                    dataframe=dataframe,
                    sensitivity=sensitivity,
                    system=args.system,
                    replica=replica,
                    column=column,
                    nskip=args.nskip,
                    min_duration_ns=args.min_duration_ns,
                    min_neff=args.min_neff,
                    tail_start_ns=args.tail_start_ns,
                )
            )

    targets = pd.DataFrame(records)
    model_targets = build_wide_table(targets)

    targets_csv = output_dir / "validated_targets_long.csv"
    model_csv = output_dir / "model_targets_wide.csv"
    targets_json = output_dir / "validated_targets.json"

    targets.to_csv(targets_csv, index=False)
    model_targets.to_csv(model_csv, index=False)
    targets.to_json(
        targets_json,
        orient="records",
        indent=2,
    )

    print("=" * 110)
    print("TARGETS VALIDADOS")
    print("=" * 110)
    print(
        targets[
            [
                "replica",
                "metric",
                "t0_ns",
                "equilibrated_duration_ns",
                "auto_neff",
                "auto_mean_A",
                "tail_mean_A",
                "target_valid",
                "target_status",
                "target_mean_A",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 110)
    print("TABLA PARA EL MODELO")
    print("=" * 110)
    print(model_targets.to_string(index=False))

    print()
    print("Resumen de validez:")
    print(
        targets.groupby(
            ["metric", "target_status"]
        ).size().to_string()
    )

    print()
    print(f"Targets largos: {targets_csv}")
    print(f"Targets modelo: {model_csv}")
    print(f"JSON:           {targets_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
