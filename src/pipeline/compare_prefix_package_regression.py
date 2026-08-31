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


FEATURE_NAMES = (
    "rmsd_backbone_A",
    "radius_gyration_A",
    "core_rmsd_q30_A",
    "core_rmsd_q40_A",
    "core_rmsd_q50_A",
    "total_sasa_A2",
    "helix_fraction",
    "strand_fraction",
)

TARGET_NAMES = (
    "rmsd",
    "rg",
    "sasa",
    "helix",
    "strand",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara dos paquetes de secuencias temporales "
            "aplicando tolerancias específicas a SASA."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--replicas", default="1,2,3")
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
        "--sasa-feature-tolerance",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--sasa-target-tolerance",
        type=float,
        default=1e-4,
    )
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

    return replicas


def read_csv(filename: Path) -> pd.DataFrame:
    if not filename.is_file():
        raise FileNotFoundError(filename)

    dataframe = pd.read_csv(filename)

    if "target_mask_signature" in dataframe.columns:
        dataframe["target_mask_signature"] = (
            pd.read_csv(
                filename,
                usecols=["target_mask_signature"],
                dtype={
                    "target_mask_signature": "string",
                },
            )["target_mask_signature"]
        )

    return dataframe


def normalized_text(
    series: pd.Series,
    column: str,
) -> pd.Series:
    values = (
        series.fillna("<NA>")
        .astype(str)
    )

    lowered = column.lower()

    if (
        "path" in lowered
        or "file" in lowered
        or lowered.endswith("_csv")
    ):
        values = values.map(
            lambda value: (
                value
                if value == "<NA>"
                else Path(value).name
            )
        )

    return values


def sort_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> pd.DataFrame:
    if "_prefix_sequence_" in filename:
        keys = ["step_index"]
    elif filename == "prefix_feature_summary.csv":
        keys = [
            "system",
            "replica",
            "feature",
        ]
    elif filename == "prefix_sequence_manifest.csv":
        keys = [
            "system",
            "replica",
        ]
    else:
        keys = []

    keys = [
        key
        for key in keys
        if key in dataframe.columns
    ]

    if not keys:
        return dataframe.reset_index(drop=True)

    return (
        dataframe.sort_values(keys)
        .reset_index(drop=True)
    )


def csv_tolerances(
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

    if "_prefix_sequence_" in filename:
        if column == "total_sasa_A2":
            tolerances[:] = sasa_tolerance

    elif filename == "prefix_feature_summary.csv":
        if "feature" in dataframe.columns:
            sasa_rows = (
                dataframe["feature"]
                .astype(str)
                .eq("total_sasa_A2")
                .to_numpy()
            )
            tolerances[sasa_rows] = sasa_tolerance

    return tolerances


def compare_csv(
    canonical_file: Path,
    candidate_file: Path,
    default_tolerance: float,
    sasa_tolerance: float,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    filename = canonical_file.name

    canonical = sort_table(
        read_csv(canonical_file),
        filename,
    )
    candidate = sort_table(
        read_csv(candidate_file),
        filename,
    )

    if list(canonical.columns) != list(
        candidate.columns
    ):
        raise RuntimeError(
            f"{filename}: columnas diferentes."
        )

    if canonical.shape != candidate.shape:
        raise RuntimeError(
            f"{filename}: formas diferentes: "
            f"{canonical.shape} y {candidate.shape}."
        )

    details: list[dict[str, object]] = []
    maximum_difference = 0.0
    failed_cells = 0
    text_mismatches = 0
    overall_pass = True

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
            first = normalized_text(
                first_series,
                column,
            )
            second = normalized_text(
                second_series,
                column,
            )

            mismatches = int(
                (first != second).sum()
            )

            text_mismatches += mismatches
            column_pass = mismatches == 0
            overall_pass &= column_pass

            details.append(
                {
                    "artifact": filename,
                    "field": column,
                    "type": "text",
                    "maximum_difference": None,
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

        finite = (
            np.isfinite(first)
            & np.isfinite(second)
        )
        equal_special = (
            (
                np.isnan(first)
                & np.isnan(second)
            )
            | (
                np.isposinf(first)
                & np.isposinf(second)
            )
            | (
                np.isneginf(first)
                & np.isneginf(second)
            )
        )

        differences = np.zeros(
            len(first),
            dtype=float,
        )
        differences[finite] = np.abs(
            first[finite] - second[finite]
        )

        tolerances = csv_tolerances(
            dataframe=canonical,
            filename=filename,
            column=column,
            default_tolerance=default_tolerance,
            sasa_tolerance=sasa_tolerance,
        )

        failures = (
            (
                finite
                & (differences > tolerances)
            )
            | ~(finite | equal_special)
        )

        current_failures = int(
            failures.sum()
        )

        finite_differences = differences[
            finite
        ]

        current_maximum = (
            float(np.max(finite_differences))
            if finite_differences.size
            else 0.0
        )

        maximum_difference = max(
            maximum_difference,
            current_maximum,
        )
        failed_cells += current_failures

        column_pass = current_failures == 0
        overall_pass &= column_pass

        details.append(
            {
                "artifact": filename,
                "field": column,
                "type": "numeric",
                "maximum_difference": (
                    current_maximum
                ),
                "failed_cells": (
                    current_failures
                ),
                "regression_pass": column_pass,
            }
        )

    summary = {
        "artifact": filename,
        "maximum_numeric_difference": (
            maximum_difference
        ),
        "numeric_failed_cells": failed_cells,
        "text_mismatch_count": text_mismatches,
        "regression_pass": bool(overall_pass),
    }

    return summary, details


def compare_numeric_array(
    first: np.ndarray,
    second: np.ndarray,
    tolerances: np.ndarray,
) -> tuple[float, int]:
    if first.shape != second.shape:
        raise RuntimeError(
            f"Formas diferentes: "
            f"{first.shape} y {second.shape}."
        )

    first = np.asarray(
        first,
        dtype=float,
    )
    second = np.asarray(
        second,
        dtype=float,
    )

    finite = (
        np.isfinite(first)
        & np.isfinite(second)
    )

    equal_special = (
        (
            np.isnan(first)
            & np.isnan(second)
        )
        | (
            np.isposinf(first)
            & np.isposinf(second)
        )
        | (
            np.isneginf(first)
            & np.isneginf(second)
        )
    )

    differences = np.zeros(
        first.shape,
        dtype=float,
    )
    differences[finite] = np.abs(
        first[finite] - second[finite]
    )

    failures = (
        (
            finite
            & (differences > tolerances)
        )
        | ~(finite | equal_special)
    )

    maximum = (
        float(np.max(differences[finite]))
        if np.any(finite)
        else 0.0
    )

    return maximum, int(failures.sum())


def compare_npz(
    canonical_file: Path,
    candidate_file: Path,
    default_tolerance: float,
    sasa_feature_tolerance: float,
    sasa_target_tolerance: float,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    details: list[dict[str, object]] = []
    overall_pass = True
    maximum_difference = 0.0
    failed_cells = 0
    text_mismatches = 0

    with (
        np.load(
            canonical_file,
            allow_pickle=False,
        ) as canonical,
        np.load(
            candidate_file,
            allow_pickle=False,
        ) as candidate,
    ):
        if canonical.files != candidate.files:
            raise RuntimeError(
                "El orden o los nombres de los arrays "
                "del NPZ no coinciden."
            )

        canonical_features = tuple(
            canonical["feature_names"].astype(str)
        )
        canonical_targets = tuple(
            canonical["target_names"].astype(str)
        )

        if canonical_features != FEATURE_NAMES:
            raise RuntimeError(
                "Orden canónico de features inesperado."
            )

        if canonical_targets != TARGET_NAMES:
            raise RuntimeError(
                "Orden canónico de targets inesperado."
            )

        for key in canonical.files:
            first = canonical[key]
            second = candidate[key]

            if key == "X_raw":
                tolerances = np.full(
                    first.shape,
                    default_tolerance,
                    dtype=float,
                )
                sasa_index = FEATURE_NAMES.index(
                    "total_sasa_A2"
                )
                tolerances[
                    :,
                    :,
                    sasa_index,
                ] = sasa_feature_tolerance

                current_maximum, failures = (
                    compare_numeric_array(
                        first,
                        second,
                        tolerances,
                    )
                )

            elif key == "target_values":
                tolerances = np.full(
                    first.shape,
                    default_tolerance,
                    dtype=float,
                )
                sasa_index = TARGET_NAMES.index(
                    "sasa"
                )
                tolerances[
                    :,
                    sasa_index,
                ] = sasa_target_tolerance

                current_maximum, failures = (
                    compare_numeric_array(
                        first,
                        second,
                        tolerances,
                    )
                )

            elif (
                np.issubdtype(
                    first.dtype,
                    np.number,
                )
                and first.dtype != np.bool_
            ):
                tolerances = np.full(
                    first.shape,
                    default_tolerance,
                    dtype=float,
                )

                current_maximum, failures = (
                    compare_numeric_array(
                        first,
                        second,
                        tolerances,
                    )
                )

            else:
                equal = (
                    first.shape == second.shape
                    and np.array_equal(
                        first,
                        second,
                    )
                )
                current_maximum = 0.0
                failures = 0 if equal else 1

            maximum_difference = max(
                maximum_difference,
                current_maximum,
            )
            failed_cells += failures

            array_pass = failures == 0
            overall_pass &= array_pass

            details.append(
                {
                    "artifact": canonical_file.name,
                    "field": key,
                    "type": "array",
                    "maximum_difference": (
                        current_maximum
                    ),
                    "failed_cells": failures,
                    "regression_pass": array_pass,
                }
            )

    summary = {
        "artifact": canonical_file.name,
        "maximum_numeric_difference": (
            maximum_difference
        ),
        "numeric_failed_cells": failed_cells,
        "text_mismatch_count": text_mismatches,
        "regression_pass": bool(overall_pass),
    }

    return summary, details


def compare_json(
    canonical_file: Path,
    candidate_file: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    canonical = json.loads(
        canonical_file.read_text(
            encoding="utf-8"
        )
    )
    candidate = json.loads(
        candidate_file.read_text(
            encoding="utf-8"
        )
    )

    equal = canonical == candidate

    summary = {
        "artifact": canonical_file.name,
        "maximum_numeric_difference": 0.0,
        "numeric_failed_cells": 0,
        "text_mismatch_count": (
            0 if equal else 1
        ),
        "regression_pass": equal,
    }

    details = [
        {
            "artifact": canonical_file.name,
            "field": "complete_json",
            "type": "json",
            "maximum_difference": None,
            "failed_cells": (
                0 if equal else 1
            ),
            "regression_pass": equal,
        }
    ]

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

    replicas = parse_replicas(
        args.replicas
    )

    csv_files = [
        *[
            (
                f"{args.system}_R{replica}"
                "_prefix_sequence_100ps.csv"
            )
            for replica in replicas
        ],
        "prefix_feature_summary.csv",
        "prefix_sequence_manifest.csv",
    ]

    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for filename in csv_files:
        summary, current_details = compare_csv(
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
                args.sasa_feature_tolerance
            ),
        )
        summaries.append(summary)
        details.extend(current_details)

    summary, current_details = compare_npz(
        canonical_file=(
            canonical_dir
            / "prefix_sequences_raw.npz"
        ),
        candidate_file=(
            candidate_dir
            / "prefix_sequences_raw.npz"
        ),
        default_tolerance=(
            args.default_tolerance
        ),
        sasa_feature_tolerance=(
            args.sasa_feature_tolerance
        ),
        sasa_target_tolerance=(
            args.sasa_target_tolerance
        ),
    )
    summaries.append(summary)
    details.extend(current_details)

    summary, current_details = compare_json(
        canonical_file=(
            canonical_dir
            / "prefix_sequences_metadata.json"
        ),
        candidate_file=(
            candidate_dir
            / "prefix_sequences_metadata.json"
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
        / "prefix_package_regression_summary.csv"
    )
    details_csv = (
        output_dir
        / "prefix_package_regression_details.csv"
    )
    output_json = (
        output_dir
        / "prefix_package_regression.json"
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
                "overall_pass": overall_pass,
                "default_tolerance": (
                    args.default_tolerance
                ),
                "sasa_feature_tolerance": (
                    args.sasa_feature_tolerance
                ),
                "sasa_target_tolerance": (
                    args.sasa_target_tolerance
                ),
                "summary": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 112)
    print(
        "REGRESIÓN END-TO-END DEL PAQUETE DE PREFIJOS"
    )
    print("=" * 112)

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
