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
            "Compara el RMSF CA completo calculado "
            "contra el RMSF oficial de ATLAS."
        )
    )

    parser.add_argument(
        "--system",
        required=True,
    )
    parser.add_argument(
        "--official-rmsf",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--calculated-long",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--official-scale-to-A",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def choose_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    for column in candidates:
        if column in dataframe.columns:
            return column

    raise RuntimeError(
        f"No se encontró {description}. "
        f"Columnas disponibles: "
        f"{list(dataframe.columns)}"
    )


def main() -> int:
    args = parse_args()

    official_file = (
        args.official_rmsf.resolve()
    )
    calculated_file = (
        args.calculated_long.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    for filename in (
        official_file,
        calculated_file,
    ):
        if not filename.is_file():
            raise FileNotFoundError(filename)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    official = pd.read_csv(
        official_file,
        sep=r"\s+",
    )

    calculated = pd.read_csv(
        calculated_file,
    )

    rmsf_column = choose_column(
        calculated,
        [
            "rmsf_mean_A",
            "rmsf_A",
            "ca_rmsf_A",
        ],
        "la columna RMSF calculada",
    )

    order_candidates = [
        "ca_position",
        "position",
        "residue_position",
        "topology_resindex",
        "topology_index",
        "residue_index",
        "topology_resid",
    ]

    order_column = next(
        (
            column
            for column in order_candidates
            if column in calculated.columns
        ),
        None,
    )

    required_columns = {
        "system",
        "replica",
        "window",
        rmsf_column,
    }

    missing = required_columns.difference(
        calculated.columns
    )

    if missing:
        raise RuntimeError(
            f"Faltan columnas calculadas: "
            f"{sorted(missing)}"
        )

    calculated = calculated.loc[
        calculated["system"].astype(str)
        == args.system
    ].copy()

    calculated = calculated.loc[
        calculated["window"].astype(str)
        == "full"
    ].copy()

    if calculated.empty:
        raise RuntimeError(
            "No hay filas de la ventana full."
        )

    if "seq" not in official.columns:
        raise RuntimeError(
            "El TSV oficial no contiene la columna seq."
        )

    summaries: list[dict[str, object]] = []

    for replica in (1, 2, 3):
        official_column = (
            f"RMSF_R{replica}"
        )

        if official_column not in official.columns:
            raise RuntimeError(
                f"Falta {official_column}."
            )

        subset = calculated.loc[
            pd.to_numeric(
                calculated["replica"],
                errors="raise",
            ).astype(int)
            == replica
        ].copy()

        if order_column is not None:
            subset = subset.sort_values(
                order_column
            )

        subset = subset.reset_index(
            drop=True
        )

        official_A = (
            pd.to_numeric(
                official[official_column],
                errors="raise",
            ).to_numpy(
                dtype=np.float64
            )
            * args.official_scale_to_A
        )

        calculated_A = pd.to_numeric(
            subset[rmsf_column],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        if len(official_A) != len(calculated_A):
            raise RuntimeError(
                f"R{replica}: oficial tiene "
                f"{len(official_A)} residuos y "
                f"calculado {len(calculated_A)}."
            )

        if not np.all(
            np.isfinite(calculated_A)
        ):
            raise RuntimeError(
                f"R{replica}: RMSF calculado no finito."
            )

        difference_A = (
            calculated_A - official_A
        )

        if np.std(official_A) == 0.0:
            pearson_r = float("nan")
        else:
            pearson_r = float(
                np.corrcoef(
                    official_A,
                    calculated_A,
                )[0, 1]
            )

        comparison = pd.DataFrame(
            {
                "system": args.system,
                "replica": replica,
                "position": np.arange(
                    1,
                    len(official_A) + 1,
                ),
                "official_sequence": (
                    official["seq"]
                    .astype(str)
                    .to_numpy()
                ),
                "official_rmsf_A": (
                    official_A
                ),
                "calculated_rmsf_A": (
                    calculated_A
                ),
                "difference_A": (
                    difference_A
                ),
                "absolute_error_A": (
                    np.abs(difference_A)
                ),
            }
        )

        if order_column is not None:
            comparison[
                f"calculated_{order_column}"
            ] = subset[
                order_column
            ].to_numpy()

        comparison_file = (
            output_dir
            / (
                f"{args.system}_R{replica}_"
                "rmsf_comparison.csv"
            )
        )

        comparison.to_csv(
            comparison_file,
            index=False,
        )

        summary = {
            "system": args.system,
            "replica": replica,
            "n_residues": int(
                len(official_A)
            ),
            "official_mean_A": float(
                np.mean(official_A)
            ),
            "calculated_mean_A": float(
                np.mean(calculated_A)
            ),
            "bias_A": float(
                np.mean(difference_A)
            ),
            "mae_A": float(
                np.mean(
                    np.abs(difference_A)
                )
            ),
            "rmse_A": float(
                np.sqrt(
                    np.mean(
                        difference_A**2
                    )
                )
            ),
            "max_abs_error_A": float(
                np.max(
                    np.abs(difference_A)
                )
            ),
            "pearson_r": pearson_r,
            "comparison_csv": str(
                comparison_file
            ),
        }

        summaries.append(summary)

        print("=" * 78)
        print(f"RMSF — {args.system} R{replica}")
        print("=" * 78)
        print(
            f"Residuos:          "
            f"{summary['n_residues']}"
        )
        print(
            f"Media oficial:     "
            f"{summary['official_mean_A']:.6f} Å"
        )
        print(
            f"Media calculada:   "
            f"{summary['calculated_mean_A']:.6f} Å"
        )
        print(
            f"Sesgo:             "
            f"{summary['bias_A']:.6f} Å"
        )
        print(
            f"MAE:               "
            f"{summary['mae_A']:.6f} Å"
        )
        print(
            f"RMSE:              "
            f"{summary['rmse_A']:.6f} Å"
        )
        print(
            f"Error máximo:      "
            f"{summary['max_abs_error_A']:.6f} Å"
        )
        print(
            f"Correlación:       "
            f"{summary['pearson_r']:.6f}"
        )

    summary_dataframe = pd.DataFrame(
        summaries
    )

    summary_csv = (
        output_dir
        / "rmsf_comparison_summary.csv"
    )
    summary_json = (
        output_dir
        / "rmsf_comparison_summary.json"
    )

    summary_dataframe.to_csv(
        summary_csv,
        index=False,
    )

    summary_json.write_text(
        json.dumps(
            {
                "system": args.system,
                "official_scale_to_A": (
                    args.official_scale_to_A
                ),
                "rmsf_column": (
                    rmsf_column
                ),
                "order_column": (
                    order_column
                ),
                "replicas": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("RESUMEN RMSF")
    print("=" * 78)
    print(
        summary_dataframe[
            [
                "replica",
                "n_residues",
                "bias_A",
                "mae_A",
                "rmse_A",
                "max_abs_error_A",
                "pearson_r",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(f"Resumen CSV:  {summary_csv}")
    print(f"Resumen JSON: {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
