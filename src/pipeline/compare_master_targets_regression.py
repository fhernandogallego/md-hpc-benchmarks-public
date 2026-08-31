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


FILES = {
    "master_targets_long.csv": (
        "system",
        "replica",
        "target_name",
    ),
    "master_targets_wide.csv": (
        "system",
        "replica",
    ),
    "master_target_masks.csv": (
        "system",
        "replica",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara los master targets canónicos y "
            "parametrizados con tolerancia específica para SASA."
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


def read_table(filename: Path) -> pd.DataFrame:
    if not filename.is_file():
        raise FileNotFoundError(filename)

    return pd.read_csv(
        filename,
        dtype={
            "system": "string",
            "target_mask_order": "string",
            "target_mask_signature": "string",
        },
    )


def normalize_text(
    series: pd.Series,
    column: str,
) -> pd.Series:
    result = (
        series.fillna("<NA>")
        .astype(str)
    )

    lowered = column.lower()

    if (
        "file" in lowered
        or "path" in lowered
    ):
        result = result.map(
            lambda value: (
                value
                if value == "<NA>"
                else Path(value).name
            )
        )

    return result


def tolerance_vector(
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

    if filename == "master_targets_long.csv":
        if "target_name" in dataframe.columns:
            sasa_mask = (
                dataframe["target_name"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("sasa")
                .to_numpy()
            )

            tolerances[sasa_mask] = (
                sasa_tolerance
            )

    elif filename == "master_targets_wide.csv":
        if column.lower().startswith("sasa_"):
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

    canonical = read_table(
        canonical_file
    )
    candidate = read_table(
        candidate_file
    )

    if list(canonical.columns) != list(
        candidate.columns
    ):
        raise RuntimeError(
            f"{filename}: columnas diferentes.\n"
            f"Canónico:  {list(canonical.columns)}\n"
            f"Candidato: {list(candidate.columns)}"
        )

    if canonical.shape != candidate.shape:
        raise RuntimeError(
            f"{filename}: formas diferentes: "
            f"{canonical.shape} frente a "
            f"{candidate.shape}."
        )

    sort_columns = [
        column
        for column in FILES[filename]
        if column in canonical.columns
    ]

    canonical = (
        canonical.sort_values(sort_columns)
        .reset_index(drop=True)
    )
    candidate = (
        candidate.sort_values(sort_columns)
        .reset_index(drop=True)
    )

    details: list[dict[str, object]] = []

    overall_pass = True
    maximum_numeric_difference = 0.0
    numeric_failed_cells = 0
    text_mismatch_count = 0

    for column in canonical.columns:
        first_series = canonical[column]
        second_series = candidate[column]

        numeric = (
            is_numeric_dtype(first_series)
            and is_numeric_dtype(second_series)
            and not is_bool_dtype(first_series)
            and not is_bool_dtype(second_series)
        )

        if not numeric:
            first = normalize_text(
                first_series,
                column,
            )
            second = normalize_text(
                second_series,
                column,
            )

            mismatches = int(
                (first != second).sum()
            )

            text_mismatch_count += mismatches
            column_pass = mismatches == 0
            overall_pass &= column_pass

            details.append(
                {
                    "filename": filename,
                    "column": column,
                    "column_type": "text",
                    "maximum_difference": None,
                    "maximum_tolerance": None,
                    "failed_cells": mismatches,
                    "regression_pass": column_pass,
                }
            )
            continue

        first = first_series.to_numpy(
            dtype=float
        )
        second = second_series.to_numpy(
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

        tolerances = tolerance_vector(
            dataframe=canonical,
            filename=filename,
            column=column,
            default_tolerance=(
                default_tolerance
            ),
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

        numeric_failed_cells += failed_cells

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
        "n_columns": int(
            len(canonical.columns)
        ),
        "maximum_numeric_difference": (
            maximum_numeric_difference
        ),
        "numeric_failed_cells": (
            numeric_failed_cells
        ),
        "text_mismatch_count": (
            text_mismatch_count
        ),
        "regression_pass": bool(
            overall_pass
        ),
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
        summary, current_details = compare_file(
            canonical_file=(
                canonical_dir / filename
            ),
            candidate_file=(
                candidate_dir / filename
            ),
            default_tolerance=(
                args.default_tolerance
            ),
            sasa_tolerance=(
                args.sasa_tolerance
            ),
        )

        summaries.append(summary)
        details.extend(current_details)

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
        / "master_targets_regression_summary.csv"
    )
    details_csv = (
        output_dir
        / "master_targets_regression_details.csv"
    )
    output_json = (
        output_dir
        / "master_targets_regression.json"
    )

    summary_dataframe.to_csv(
        summary_csv,
        index=False,
    )
    details_dataframe.to_csv(
        details_csv,
        index=False,
    )

    output_json.write_text(
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
        "REGRESIÓN END-TO-END DE MASTER TARGETS"
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
    print(f"JSON:     {output_json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
