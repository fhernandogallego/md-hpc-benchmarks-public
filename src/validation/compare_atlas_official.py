#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara RMSD y radio de giro calculados con los valores "
            "oficiales descargados de ATLAS."
        )
    )
    parser.add_argument("--official-dir", required=True, type=Path)
    parser.add_argument("--calculated-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def calculate_metrics(
    official: np.ndarray,
    calculated: np.ndarray,
) -> dict[str, float | int]:
    official = np.asarray(official, dtype=float)
    calculated = np.asarray(calculated, dtype=float)

    valid = np.isfinite(official) & np.isfinite(calculated)
    official = official[valid]
    calculated = calculated[valid]

    if len(official) == 0:
        raise RuntimeError("No hay valores válidos para comparar.")

    errors = calculated - official

    if len(official) > 1:
        correlation = float(np.corrcoef(official, calculated)[0, 1])
    else:
        correlation = float("nan")

    return {
        "n_points": int(len(official)),
        "official_mean_A": float(np.mean(official)),
        "calculated_mean_A": float(np.mean(calculated)),
        "bias_A": float(np.mean(errors)),
        "mae_A": float(np.mean(np.abs(errors))),
        "rmse_A": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error_A": float(np.max(np.abs(errors))),
        "pearson_r": correlation,
    }


def prepare_official_table(
    dataframe: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    result = dataframe[["time", value_column]].copy()
    result.columns = ["time_ns", "official_A"]
    result["time_ns"] = result["time_ns"].astype(float).round(6)
    result["official_A"] = result["official_A"].astype(float)
    return result


def prepare_calculated_table(
    filename: Path,
    value_column: str,
) -> pd.DataFrame:
    dataframe = pd.read_csv(filename)

    required = {"time_ns", value_column}
    missing = required.difference(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"Faltan columnas en {filename}: {sorted(missing)}"
        )

    result = dataframe[["time_ns", value_column]].copy()
    result.columns = ["time_ns", "calculated_A"]
    result["time_ns"] = result["time_ns"].astype(float).round(6)
    result["calculated_A"] = result["calculated_A"].astype(float)
    return result


def compare_metric(
    metric: str,
    replica: int,
    official: pd.DataFrame,
    calculated: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    merged = official.merge(
        calculated,
        on="time_ns",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise RuntimeError(
            f"No hay tiempos coincidentes para {metric}, R{replica}."
        )

    merged["difference_A"] = (
        merged["calculated_A"] - merged["official_A"]
    )
    merged["absolute_error_A"] = merged["difference_A"].abs()

    output_csv = output_dir / f"{metric.lower()}_R{replica}_comparison.csv"
    merged.to_csv(output_csv, index=False)

    metrics = calculate_metrics(
        official=merged["official_A"].to_numpy(),
        calculated=merged["calculated_A"].to_numpy(),
    )

    result: dict[str, object] = {
        "metric": metric,
        "replica": replica,
        **metrics,
        "comparison_csv": str(output_csv),
    }

    print("=" * 72)
    print(f"{metric} — R{replica}")
    print(f"Puntos comparados: {metrics['n_points']}")
    print(f"Media oficial:     {metrics['official_mean_A']:.4f} Å")
    print(f"Media calculada:   {metrics['calculated_mean_A']:.4f} Å")
    print(f"Sesgo:             {metrics['bias_A']:.4f} Å")
    print(f"MAE:               {metrics['mae_A']:.4f} Å")
    print(f"RMSE:              {metrics['rmse_A']:.4f} Å")
    print(f"Error máximo:      {metrics['max_abs_error_A']:.4f} Å")
    print(f"Correlación:       {metrics['pearson_r']:.6f}")
    print()
    print("Primeras diferencias:")
    print(
        merged.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return result


def main() -> int:
    args = parse_args()

    official_dir = args.official_dir.resolve()
    calculated_dir = args.calculated_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rmsd_file = official_dir / "1k5n_A_RMSD.tsv"
    gyrate_file = official_dir / "1k5n_A_gyrate.tsv"

    official_rmsd = pd.read_csv(rmsd_file, sep=r"\s+")
    official_gyrate = pd.read_csv(gyrate_file, sep=r"\s+")

    results: list[dict[str, object]] = []

    for replica in (1, 2, 3):
        calculated_file = (
            calculated_dir
            / f"1k5n_A_R{replica}_basic_observables_100ps.csv"
        )

        calculated_rmsd = prepare_calculated_table(
            calculated_file,
            "rmsd_backbone_A",
        )
        calculated_rg = prepare_calculated_table(
            calculated_file,
            "radius_gyration_A",
        )

        official_rmsd_replica = prepare_official_table(
            official_rmsd,
            f"RMSD_R{replica}",
        )
        official_rg_replica = prepare_official_table(
            official_gyrate,
            f"gyr_R{replica}",
        )

        results.append(
            compare_metric(
                metric="RMSD",
                replica=replica,
                official=official_rmsd_replica,
                calculated=calculated_rmsd,
                output_dir=output_dir,
            )
        )

        results.append(
            compare_metric(
                metric="Rg",
                replica=replica,
                official=official_rg_replica,
                calculated=calculated_rg,
                output_dir=output_dir,
            )
        )

    summary = pd.DataFrame(results)
    summary_csv = output_dir / "comparison_summary.csv"
    summary_json = output_dir / "comparison_summary.json"

    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("RESUMEN DE COMPARACIÓN")
    print("=" * 72)
    print(
        summary[
            [
                "metric",
                "replica",
                "n_points",
                "bias_A",
                "mae_A",
                "rmse_A",
                "max_abs_error_A",
                "pearson_r",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Resumen CSV:  {summary_csv}")
    print(f"Resumen JSON: {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
