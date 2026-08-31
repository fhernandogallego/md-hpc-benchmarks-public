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
            "Compara archivos CSV canónicos y candidatos."
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
        "--filenames",
        required=True,
        help="Nombres separados por comas.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--path-columns",
        default="",
        help=(
            "Columnas de rutas separadas por comas. "
            "Se comparará únicamente el nombre final "
            "del archivo, no la raíz del directorio."
        ),
    )
    return parser.parse_args()


def parse_filenames(value: str) -> list[str]:
    filenames = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    if not filenames:
        raise ValueError(
            "No se proporcionaron archivos."
        )

    return filenames


def compare_dataframes(
    canonical: pd.DataFrame,
    candidate: pd.DataFrame,
    filename: str,
    tolerance: float,
    path_columns: set[str],
) -> dict[str, object]:
    if list(canonical.columns) != list(
        candidate.columns
    ):
        raise RuntimeError(
            f"{filename}: columnas diferentes.\n"
            f"Canónico: {list(canonical.columns)}\n"
            f"Candidato: {list(candidate.columns)}"
        )

    if canonical.shape != candidate.shape:
        raise RuntimeError(
            f"{filename}: formas diferentes, "
            f"{canonical.shape} frente a {candidate.shape}."
        )

    numeric_columns = [
        column
        for column in canonical.columns
        if (
            pd.api.types.is_numeric_dtype(
                canonical[column]
            )
            and pd.api.types.is_numeric_dtype(
                candidate[column]
            )
        )
    ]

    non_numeric_columns = [
        column
        for column in canonical.columns
        if column not in numeric_columns
    ]

    numeric_maximums: dict[str, float] = {}
    maximum_numeric_difference = 0.0
    numeric_pass = True

    for column in numeric_columns:
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
            numeric_pass = False
            numeric_maximums[column] = float(
                "inf"
            )
            maximum_numeric_difference = float(
                "inf"
            )
            continue

        finite = (
            np.isfinite(first)
            & np.isfinite(second)
        )

        if finite.any():
            maximum = float(
                np.max(
                    np.abs(
                        first[finite]
                        - second[finite]
                    )
                )
            )
        else:
            maximum = 0.0

        numeric_maximums[column] = maximum
        maximum_numeric_difference = max(
            maximum_numeric_difference,
            maximum,
        )

        if maximum > tolerance:
            numeric_pass = False

    text_mismatch_count = 0
    text_mismatches_by_column: dict[str, int] = {}

    for column in non_numeric_columns:
        first = canonical[column].fillna(
            "<NA>"
        ).astype(str)
        second = candidate[column].fillna(
            "<NA>"
        ).astype(str)

        if column in path_columns:
            first = first.map(
                lambda value: (
                    value
                    if value == "<NA>"
                    else Path(value).name
                )
            )
            second = second.map(
                lambda value: (
                    value
                    if value == "<NA>"
                    else Path(value).name
                )
            )

        mismatches = int(
            (first != second).sum()
        )

        text_mismatches_by_column[
            column
        ] = mismatches
        text_mismatch_count += mismatches

    regression_pass = bool(
        numeric_pass
        and text_mismatch_count == 0
    )

    return {
        "filename": filename,
        "n_rows": int(len(canonical)),
        "n_columns": int(
            len(canonical.columns)
        ),
        "numeric_column_count": len(
            numeric_columns
        ),
        "non_numeric_column_count": len(
            non_numeric_columns
        ),
        "maximum_numeric_difference": (
            maximum_numeric_difference
        ),
        "text_mismatch_count": (
            text_mismatch_count
        ),
        "regression_pass": regression_pass,
        "numeric_maximums": (
            numeric_maximums
        ),
        "text_mismatches_by_column": (
            text_mismatches_by_column
        ),
    }


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

    filenames = parse_filenames(
        args.filenames
    )

    path_columns = {
        item.strip()
        for item in args.path_columns.split(",")
        if item.strip()
    }

    records: list[dict[str, object]] = []

    for filename in filenames:
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

        canonical = pd.read_csv(
            canonical_file
        )
        candidate = pd.read_csv(
            candidate_file
        )

        records.append(
            compare_dataframes(
                canonical=canonical,
                candidate=candidate,
                filename=filename,
                tolerance=args.tolerance,
                path_columns=path_columns,
            )
        )

    overall_pass = all(
        bool(record["regression_pass"])
        for record in records
    )

    summary_records = [
        {
            key: value
            for key, value in record.items()
            if key
            not in {
                "numeric_maximums",
                "text_mismatches_by_column",
            }
        }
        for record in records
    ]

    summary = pd.DataFrame(
        summary_records
    )

    output_csv = (
        output_dir
        / "csv_regression_summary.csv"
    )
    output_json = (
        output_dir
        / "csv_regression_details.json"
    )

    summary.to_csv(
        output_csv,
        index=False,
    )

    output_json.write_text(
        json.dumps(
            {
                "tolerance": args.tolerance,
                "overall_pass": overall_pass,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 110)
    print("REGRESIÓN DE ARCHIVOS CSV")
    print("=" * 110)

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
    print(f"Resumen:       {output_csv}")
    print(f"Detalles:      {output_json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
