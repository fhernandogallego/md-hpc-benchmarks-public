#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)


FILES = (
    "structural_targets_long.csv",
    "structural_model_targets_wide.csv",
    "secondary_structure_composition.csv",
)

KEY_CANDIDATES = {
    "structural_targets_long.csv": (
        "system",
        "replica",
        "metric",
        "source_column",
    ),
    "structural_model_targets_wide.csv": (
        "system",
        "replica",
    ),
    "secondary_structure_composition.csv": (
        "system",
        "replica",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara targets estructurales aplicando una "
            "tolerancia específica a resultados SASA."
        )
    )
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
        "--default-tolerance",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--sasa-tolerance",
        type=float,
        default=1e-4,
    )
    return parser.parse_args()


def sorted_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
) -> pd.DataFrame:
    keys = [
        column
        for column in KEY_CANDIDATES[filename]
        if column in dataframe.columns
    ]

    if not keys:
        return dataframe.reset_index(drop=True)

    return (
        dataframe.sort_values(keys)
        .reset_index(drop=True)
    )


def sasa_rows(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    mask = np.zeros(
        len(dataframe),
        dtype=bool,
    )

    for column in (
        "metric",
        "source_column",
        "observable",
        "target_name",
    ):
        if column not in dataframe.columns:
            continue

        mask |= (
            dataframe[column]
            .astype(str)
            .str.contains(
                "sasa",
                case=False,
                regex=False,
            )
            .to_numpy()
        )

    return mask


def column_tolerances(
    dataframe: pd.DataFrame,
    filename: str,
    column: str,
    default_tolerance: float,
    sasa_tolerance: float,
) -> np.ndarray:
    tolerances = np.full(
        len(dataframe),
        default_tolerance,
        dtype=float,
    )

    if filename == "structural_targets_long.csv":
        tolerances[
            sasa_rows(dataframe)
        ] = sasa_tolerance

    elif (
        filename
        == "structural_model_targets_wide.csv"
        and column.lower().startswith("sasa_")
    ):
        tolerances[:] = sasa_tolerance

    return tolerances


def compare_file(
    canonical_file: Path,
    candidate_file: Path,
    default_tolerance: float,
    sasa_tolerance: float,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    filename = canonical_file.name

    canonical = pd.read_csv(canonical_file)
    candidate = pd.read_csv(candidate_file)

    if list(canonical.columns) != list(candidate.columns):
        raise RuntimeError(
            f"{filename}: las columnas o su orden no coinciden.\n"
            f"Canónico:  {list(canonical.columns)}\n"
            f"Candidato: {list(candidate.columns)}"
        )

    if canonical.shape != candidate.shape:
        raise RuntimeError(
            f"{filename}: formas diferentes: "
            f"{canonical.shape} y {candidate.shape}"
        )

    canonical = sorted_dataframe(
        canonical,
        filename,
    )
    candidate = sorted_dataframe(
        candidate,
        filename,
    )

    details: list[dict[str, object]] = []
    overall_pass = True
    total_text_mismatches = 0
    total_numeric_failures = 0
    maximum_numeric_difference = 0.0

    for column in canonical.columns:
        canonical_column = canonical[column]
        candidate_column = candidate[column]

        numeric = (
            is_numeric_dtype(canonical_column)
            and is_numeric_dtype(candidate_column)
            and not is_bool_dtype(canonical_column)
            and not is_bool_dtype(candidate_column)
        )

        if not numeric:
            first = (
                canonical_column
                .fillna("<NA>")
                .astype(str)
            )
            second = (
                candidate_column
                .fillna("<NA>")
                .astype(str)
            )

            mismatch_count = int(
                (first != second).sum()
            )

            total_text_mismatches += mismatch_count
            column_pass = mismatch_count == 0
            overall_pass &= column_pass

            details.append(
                {
                    "filename": filename,
                    "column": column,
                    "column_type": "text",
                    "maximum_difference": None,
                    "maximum_tolerance": None,
                    "failed_cells": mismatch_count,
                    "regression_pass": column_pass,
                }
            )
            continue

        first = canonical_column.to_numpy(
            dtype=float
        )
        second = candidate_column.to_numpy(
            dtype=float
        )

        nan_mismatch = (
            np.isnan(first)
            != np.isnan(second)
        )

        finite = (
            np.isfinite(first)
            & np.isfinite(second)
        )

        differences = np.zeros(
            len(first),
            dtype=float,
        )
        differences[finite] = np.abs(
            first[finite] - second[finite]
        )
        differences[nan_mismatch] = np.inf

        tolerances = column_tolerances(
            dataframe=canonical,
            filename=filename,
            column=column,
            default_tolerance=default_tolerance,
            sasa_tolerance=sasa_tolerance,
        )

        failures = (
            nan_mismatch
            | (
                finite
                & (differences > tolerances)
            )
        )

        failed_cells = int(
            failures.sum()
        )

        finite_differences = differences[
            np.isfinite(differences)
        ]

        maximum_difference = (
            float(np.max(finite_differences))
            if len(finite_differences)
            else 0.0
        )

        maximum_numeric_difference = max(
            maximum_numeric_difference,
            maximum_difference,
        )

        total_numeric_failures += failed_cells
        column_pass = failed_cells == 0
        overall_pass &= column_pass

        details.append(
            {
                "filename": filename,
                "column": column,
                "column_type": "numeric",
                "maximum_difference": (
                    maximum_difference
                ),
                "maximum_tolerance": float(
                    np.max(tolerances)
                ),
                "failed_cells": failed_cells,
                "regression_pass": column_pass,
            }
        )

    summary = {
        "filename": filename,
        "n_rows": int(len(canonical)),
        "n_columns": int(len(canonical.columns)),
        "maximum_numeric_difference": (
            maximum_numeric_difference
        ),
        "numeric_failed_cells": (
            total_numeric_failures
        ),
        "text_mismatch_count": (
            total_text_mismatches
        ),
        "regression_pass": bool(overall_pass),
    }

    return summary, details


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

    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for filename in FILES:
        canonical_file = (
            canonical_dir / filename
        )
        candidate_file = (
            candidate_dir / filename
        )

        if not canonical_file.is_file():
            raise FileNotFoundError(
                canonical_file
            )

        if not candidate_file.is_file():
            raise FileNotFoundError(
                candidate_file
            )

        summary, file_details = compare_file(
            canonical_file=canonical_file,
            candidate_file=candidate_file,
            default_tolerance=(
                args.default_tolerance
            ),
            sasa_tolerance=(
                args.sasa_tolerance
            ),
        )

        summaries.append(summary)
        details.extend(file_details)

    summary_dataframe = pd.DataFrame(
        summaries
    )
    details_dataframe = pd.DataFrame(
        details
    )

    overall_pass = bool(
        summary_dataframe[
            "regression_pass"
        ].all()
    )

    summary_csv = (
        output_dir
        / "structural_targets_regression_summary.csv"
    )
    details_csv = (
        output_dir
        / "structural_targets_regression_details.csv"
    )
    metadata_json = (
        output_dir
        / "structural_targets_regression.json"
    )

    summary_dataframe.to_csv(
        summary_csv,
        index=False,
    )
    details_dataframe.to_csv(
        details_csv,
        index=False,
    )

    metadata_json.write_text(
        json.dumps(
            {
                "default_tolerance": (
                    args.default_tolerance
                ),
                "sasa_tolerance": (
                    args.sasa_tolerance
                ),
                "overall_pass": overall_pass,
                "summary": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 110)
    print(
        "REGRESIÓN END-TO-END DE TARGETS ESTRUCTURALES"
    )
    print("=" * 110)
    print(
        summary_dataframe.to_string(
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
    print(f"Resumen:  {summary_csv}")
    print(f"Detalles: {details_csv}")
    print(f"JSON:     {metadata_json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
