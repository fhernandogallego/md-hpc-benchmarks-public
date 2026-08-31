#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymbar import timeseries


METRICS = {
    "rmsd_backbone_A": "RMSD",
    "radius_gyration_A": "Rg",
}

NSKIP_VALUES = (1, 5, 10, 20, 50, 100)
STRIDES = {
    1: "10ps",
    10: "100ps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--replicas", default="1,2,3")
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


def detect(
    dataframe: pd.DataFrame,
    system: str,
    column: str,
    replica: int,
    stride: int,
    resolution: str,
    nskip: int,
) -> dict[str, object]:
    reduced = dataframe.iloc[::stride].reset_index(drop=True)
    values = reduced[column].to_numpy(dtype=float)

    t0, g, neff = timeseries.detect_equilibration(
        values,
        fast=False,
        nskip=nskip,
    )

    t0 = int(t0)
    t0_ns = float(reduced["time_ns"].iloc[t0])
    last_ns = float(reduced["time_ns"].iloc[-1])
    duration_ns = last_ns - t0_ns

    equilibrated = values[t0:]

    return {
        "system": system,
        "replica": replica,
        "metric": METRICS[column],
        "column": column,
        "resolution": resolution,
        "stride": stride,
        "nskip": nskip,
        "n_input_points": int(len(values)),
        "t0_index": t0,
        "t0_ns": t0_ns,
        "duration_ns": duration_ns,
        "n_equilibrated_points": int(len(equilibrated)),
        "detection_g": float(g),
        "detection_neff": float(neff),
        "equilibrated_mean_A": float(np.mean(equilibrated)),
        "equilibrated_std_A": float(np.std(equilibrated, ddof=1)),
        "duration_at_least_20ns": bool(duration_ns >= 20.0),
        "neff_at_least_10": bool(neff >= 10.0),
    }


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []

    for replica in parse_replicas(args.replicas):
        filename = (
            input_dir
            / (
                f"{args.system}_R{replica}"
                "_basic_observables.csv"
            )
        )

        dataframe = pd.read_csv(filename)

        required = {
            "time_ns",
            *METRICS.keys(),
        }
        missing = required.difference(
            dataframe.columns
        )

        if missing:
            raise RuntimeError(
                f"{filename}: faltan columnas "
                f"{sorted(missing)}"
            )

        if "system" in dataframe.columns:
            observed_systems = set(
                dataframe["system"]
                .astype(str)
                .unique()
            )

            if observed_systems != {args.system}:
                raise RuntimeError(
                    f"{filename}: sistemas observados "
                    f"{sorted(observed_systems)}; "
                    f"esperado {args.system}."
                )

        for column in METRICS:
            for stride, resolution in STRIDES.items():
                for nskip in NSKIP_VALUES:
                    records.append(
                        detect(
                            dataframe=dataframe,
                            system=args.system,
                            column=column,
                            replica=replica,
                            stride=stride,
                            resolution=resolution,
                            nskip=nskip,
                        )
                    )

    results = pd.DataFrame(records)

    output_csv = output_dir / "equilibration_sensitivity.csv"
    results.to_csv(output_csv, index=False)

    summary = (
        results.groupby(
            ["replica", "metric", "resolution"],
            as_index=False,
        )
        .agg(
            t0_min_ns=("t0_ns", "min"),
            t0_max_ns=("t0_ns", "max"),
            duration_min_ns=("duration_ns", "min"),
            duration_max_ns=("duration_ns", "max"),
            neff_min=("detection_neff", "min"),
            neff_max=("detection_neff", "max"),
            target_mean_min_A=("equilibrated_mean_A", "min"),
            target_mean_max_A=("equilibrated_mean_A", "max"),
        )
    )

    summary_csv = output_dir / "equilibration_sensitivity_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("=" * 100)
    print("SENSIBILIDAD DE LA DETECCIÓN DE EQUILIBRACIÓN")
    print("=" * 100)
    print(
        results[
            [
                "replica",
                "metric",
                "resolution",
                "nskip",
                "t0_ns",
                "duration_ns",
                "detection_g",
                "detection_neff",
                "equilibrated_mean_A",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("RANGO DE RESULTADOS")
    print("=" * 100)
    print(summary.to_string(index=False))

    print()
    print(f"Detalle: {output_csv}")
    print(f"Resumen: {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
