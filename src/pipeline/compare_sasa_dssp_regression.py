#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIMESERIES_COLUMNS = [
    "time_ns",
    "total_sasa_A2",
    "mean_residue_sasa_A2",
    "helix_fraction",
    "strand_fraction",
    "coil_fraction",
    "structured_fraction",
]

SUMMARY_KEYS = [
    "system",
    "replica",
    "window",
]

SUMMARY_COLUMNS = [
    "n_frames",
    "start_ns",
    "end_ns",
    "total_sasa_mean_A2",
    "total_sasa_std_A2",
    "helix_fraction_mean",
    "strand_fraction_mean",
    "coil_fraction_mean",
    "structured_fraction_mean",
]

PROFILE_KEYS = [
    "system",
    "replica",
    "residue_index",
]

PROFILE_COLUMNS = [
    "prefix_sasa_mean_A2",
    "prefix_sasa_std_A2",
    "full_sasa_mean_A2",
    "full_sasa_std_A2",
    "prefix_helix_occupancy",
    "prefix_strand_occupancy",
    "prefix_coil_occupancy",
    "full_helix_occupancy",
    "full_strand_occupancy",
    "full_coil_occupancy",
]


PROFILE_ALIASES = {
    "prefix_helix_occupancy": [
        "prefix_helix_occupancy",
        "prefix_H_occupancy",
        "prefix_H_fraction",
        "prefix_helix_fraction",
    ],
    "prefix_strand_occupancy": [
        "prefix_strand_occupancy",
        "prefix_E_occupancy",
        "prefix_E_fraction",
        "prefix_strand_fraction",
    ],
    "prefix_coil_occupancy": [
        "prefix_coil_occupancy",
        "prefix_C_occupancy",
        "prefix_C_fraction",
        "prefix_coil_fraction",
    ],
    "full_helix_occupancy": [
        "full_helix_occupancy",
        "full_H_occupancy",
        "full_H_fraction",
        "full_helix_fraction",
    ],
    "full_strand_occupancy": [
        "full_strand_occupancy",
        "full_E_occupancy",
        "full_E_fraction",
        "full_strand_fraction",
    ],
    "full_coil_occupancy": [
        "full_coil_occupancy",
        "full_C_occupancy",
        "full_C_fraction",
        "full_coil_fraction",
    ],
}


COLUMN_ABSOLUTE_TOLERANCES = {
    # La suma de cientos de SASA residuales puede acumular
    # pequeñas diferencias de redondeo de OpenMP/MDTraj.
    "total_sasa_A2": 5e-3,

    # SASA total dividido entre el número de residuos.
    "mean_residue_sasa_A2": 5e-5,

    # Estadísticos agregados de la serie temporal.
    "total_sasa_mean_A2": 1e-3,
    "total_sasa_std_A2": 1e-3,

    # Estadísticos SASA calculados por residuo.
    "prefix_sasa_mean_A2": 5e-4,
    "prefix_sasa_std_A2": 5e-4,
    "full_sasa_mean_A2": 5e-4,
    "full_sasa_std_A2": 5e-4,
}


def normalize_profile_schema(
    dataframe: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    result = dataframe.copy()

    for standard_name, aliases in PROFILE_ALIASES.items():
        if standard_name in result.columns:
            continue

        matches = [
            alias
            for alias in aliases
            if alias in result.columns
        ]

        if len(matches) == 0:
            raise RuntimeError(
                f"{label}: no se encontró una columna "
                f"equivalente a {standard_name}. "
                f"Alternativas: {aliases}"
            )

        if len(matches) > 1:
            raise RuntimeError(
                f"{label}: varias columnas equivalentes "
                f"a {standard_name}: {matches}"
            )

        result = result.rename(
            columns={matches[0]: standard_name}
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara SASA/DSSP parametrizado "
            "contra el POC canónico."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument(
        "--canonical-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
    )
    return parser.parse_args()


def compare_numeric_tables(
    canonical: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: list[str],
    label: str,
    tolerance: float,
) -> tuple[list[dict[str, object]], bool]:
    missing_canonical = set(columns).difference(
        canonical.columns
    )
    missing_candidate = set(columns).difference(
        candidate.columns
    )

    if missing_canonical:
        raise RuntimeError(
            f"{label}: faltan columnas canónicas: "
            f"{sorted(missing_canonical)}"
        )

    if missing_candidate:
        raise RuntimeError(
            f"{label}: faltan columnas candidatas: "
            f"{sorted(missing_candidate)}"
        )

    if len(canonical) != len(candidate):
        raise RuntimeError(
            f"{label}: distinto número de filas: "
            f"{len(canonical)} frente a {len(candidate)}."
        )

    records: list[dict[str, object]] = []
    table_pass = True

    for column in columns:
        first = canonical[column].to_numpy(
            dtype=float
        )
        second = candidate[column].to_numpy(
            dtype=float
        )

        same_nan = (
            np.isnan(first)
            == np.isnan(second)
        )

        if not np.all(same_nan):
            maximum = float("inf")
            mean = float("inf")
            column_pass = False
        else:
            finite = (
                np.isfinite(first)
                & np.isfinite(second)
            )

            if finite.any():
                differences = np.abs(
                    first[finite]
                    - second[finite]
                )
                maximum = float(
                    np.max(differences)
                )
                mean = float(
                    np.mean(differences)
                )
            else:
                maximum = 0.0
                mean = 0.0


        column_tolerance = (
            COLUMN_ABSOLUTE_TOLERANCES.get(
                column,
                tolerance,
            )
        )

        column_pass = (
            np.isfinite(maximum)
            and maximum <= column_tolerance
        )

        records.append(
            {
                "table": label,
                "column": column,
                "n_rows": int(len(canonical)),
                "tolerance": column_tolerance,
                "max_abs_difference": maximum,
                "mean_abs_difference": mean,
                "regression_pass": column_pass,
            }
        )

        table_pass = (
            table_pass and column_pass
        )

    return records, table_pass


def main() -> int:
    args = parse_args()

    canonical_dir = (
        args.canonical_dir.resolve()
    )
    candidate_dir = (
        args.candidate_dir.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[dict[str, object]] = []
    overall_pass = True

    for replica in (1, 2, 3):
        filename = (
            f"{args.system}_R{replica}"
            "_sasa_dssp_100ps.csv"
        )

        canonical = pd.read_csv(
            canonical_dir / filename
        )
        candidate = pd.read_csv(
            candidate_dir / filename
        )

        canonical = canonical.sort_values(
            "time_ns"
        ).reset_index(drop=True)
        candidate = candidate.sort_values(
            "time_ns"
        ).reset_index(drop=True)

        current_records, current_pass = (
            compare_numeric_tables(
                canonical=canonical,
                candidate=candidate,
                columns=TIMESERIES_COLUMNS,
                label=f"R{replica}_timeseries",
                tolerance=args.tolerance,
            )
        )

        records.extend(current_records)
        overall_pass = (
            overall_pass and current_pass
        )

    canonical_summary = pd.read_csv(
        canonical_dir / "sasa_dssp_summary.csv"
    )
    candidate_summary = pd.read_csv(
        candidate_dir / "sasa_dssp_summary.csv"
    )

    canonical_summary = (
        canonical_summary.sort_values(
            SUMMARY_KEYS
        ).reset_index(drop=True)
    )
    candidate_summary = (
        candidate_summary.sort_values(
            SUMMARY_KEYS
        ).reset_index(drop=True)
    )

    if not canonical_summary[
        SUMMARY_KEYS
    ].equals(
        candidate_summary[
            SUMMARY_KEYS
        ]
    ):
        raise RuntimeError(
            "Las claves del resumen SASA/DSSP no coinciden."
        )

    current_records, current_pass = (
        compare_numeric_tables(
            canonical=canonical_summary,
            candidate=candidate_summary,
            columns=SUMMARY_COLUMNS,
            label="summary",
            tolerance=args.tolerance,
        )
    )

    records.extend(current_records)
    overall_pass = (
        overall_pass and current_pass
    )

    canonical_profiles = pd.read_csv(
        canonical_dir
        / "sasa_dssp_residue_profiles.csv"
    )
    candidate_profiles = pd.read_csv(
        candidate_dir
        / "sasa_dssp_residue_profiles.csv"
    )

    canonical_profiles = normalize_profile_schema(
        canonical_profiles,
        label="residue_profiles canónico",
    )
    candidate_profiles = normalize_profile_schema(
        candidate_profiles,
        label="residue_profiles candidato",
    )

    canonical_profiles = (
        canonical_profiles.sort_values(
            PROFILE_KEYS
        ).reset_index(drop=True)
    )
    candidate_profiles = (
        candidate_profiles.sort_values(
            PROFILE_KEYS
        ).reset_index(drop=True)
    )

    if not canonical_profiles[
        PROFILE_KEYS
    ].equals(
        candidate_profiles[
            PROFILE_KEYS
        ]
    ):
        raise RuntimeError(
            "Las claves de los perfiles por residuo "
            "no coinciden."
        )

    current_records, current_pass = (
        compare_numeric_tables(
            canonical=canonical_profiles,
            candidate=candidate_profiles,
            columns=PROFILE_COLUMNS,
            label="residue_profiles",
            tolerance=args.tolerance,
        )
    )

    records.extend(current_records)
    overall_pass = (
        overall_pass and current_pass
    )

    results = pd.DataFrame(records)

    summary = (
        results.groupby(
            "table",
            as_index=False,
        )
        .agg(
            compared_columns=(
                "column",
                "count",
            ),
            maximum_numeric_difference=(
                "max_abs_difference",
                "max",
            ),
            failed_columns=(
                "regression_pass",
                lambda values: int(
                    (~values).sum()
                ),
            ),
            regression_pass=(
                "regression_pass",
                "all",
            ),
        )
    )

    output_csv = (
        output_dir
        / "sasa_dssp_regression_summary.csv"
    )
    details_csv = (
        output_dir
        / "sasa_dssp_regression_details.csv"
    )
    output_json = (
        output_dir
        / "sasa_dssp_regression.json"
    )

    summary.to_csv(
        output_csv,
        index=False,
    )
    results.to_csv(
        details_csv,
        index=False,
    )

    output_json.write_text(
        json.dumps(
            {
                "system": args.system,
                "tolerance": args.tolerance,
                "overall_pass": overall_pass,
                "summary": summary.to_dict(
                    orient="records"
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("REGRESIÓN SASA/DSSP")
    print("=" * 100)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.12e}"
            ),
        )
    )
    print()
    print(
        "Estado global: "
        + ("PASS" if overall_pass else "FAIL")
    )
    print(f"Resumen:  {output_csv}")
    print(f"Detalles: {details_csv}")
    print(f"JSON:     {output_json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
