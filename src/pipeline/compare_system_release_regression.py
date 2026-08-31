#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCIENTIFIC_DIRS = (
    "basic",
    "rmsf",
    "core_overlap",
    "core_rmsd",
    "sasa_dssp",
    "targets",
    "structural_targets_compositional",
    "master_targets",
    "prefix_sequences",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara los productos científicos de una ejecución "
            "orquestada contra una release congelada."
        )
    )

    parser.add_argument(
        "--system",
        required=True,
    )
    parser.add_argument(
        "--dataset-group",
        required=True,
    )
    parser.add_argument(
        "--canonical-release",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=1e-9,
    )
    parser.add_argument(
        "--npz-tolerance",
        type=float,
        default=1e-12,
    )

    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return "<NA>"

    text = str(value)

    # Las rutas absolutas cambian legítimamente entre la release
    # y el nuevo RUN_ROOT. Comparamos su basename.
    if text.startswith("/"):
        return Path(text).name

    return text


def compare_csv(
    canonical_file: Path,
    candidate_file: Path,
    tolerance: float,
) -> dict[str, object]:

    result: dict[str, object] = {
        "file": str(canonical_file),
        "candidate": str(candidate_file),
        "pass": True,
        "max_abs_difference": 0.0,
        "errors": [],
    }

    errors: list[str] = result["errors"]  # type: ignore[assignment]

    if not candidate_file.is_file():
        errors.append("candidate_missing")
        result["pass"] = False
        return result

    canonical = pd.read_csv(canonical_file)
    candidate = pd.read_csv(candidate_file)

    if canonical.shape != candidate.shape:
        errors.append(
            f"shape: canonical={canonical.shape}, "
            f"candidate={candidate.shape}"
        )
        result["pass"] = False
        return result

    if list(canonical.columns) != list(candidate.columns):
        errors.append("column_order_or_names")
        result["pass"] = False
        return result

    max_difference = 0.0

    for column in canonical.columns:
        left = canonical[column]
        right = candidate[column]

        left_bool = pd.api.types.is_bool_dtype(left.dtype)
        right_bool = pd.api.types.is_bool_dtype(right.dtype)

        if left_bool and right_bool:
            if not np.array_equal(
                left.to_numpy(),
                right.to_numpy(),
            ):
                errors.append(f"{column}: boolean_difference")
            continue

        left_numeric = pd.api.types.is_numeric_dtype(left.dtype)
        right_numeric = pd.api.types.is_numeric_dtype(right.dtype)

        if left_numeric and right_numeric:
            a = left.to_numpy(dtype=np.float64)
            b = right.to_numpy(dtype=np.float64)

            close = np.isclose(
                a,
                b,
                rtol=0.0,
                atol=tolerance,
                equal_nan=True,
            )

            if not np.all(close):
                bad = int(np.size(close) - np.count_nonzero(close))
                errors.append(
                    f"{column}: {bad} numeric differences "
                    f"> {tolerance}"
                )

            finite = np.isfinite(a) & np.isfinite(b)

            if np.any(finite):
                diff = float(
                    np.max(
                        np.abs(
                            a[finite] - b[finite]
                        )
                    )
                )
                max_difference = max(
                    max_difference,
                    diff,
                )

            continue

        a_text = np.asarray(
            [normalize_text(value) for value in left],
            dtype=str,
        )
        b_text = np.asarray(
            [normalize_text(value) for value in right],
            dtype=str,
        )

        if not np.array_equal(a_text, b_text):
            bad = int(
                np.count_nonzero(
                    a_text != b_text
                )
            )
            errors.append(
                f"{column}: {bad} text differences"
            )

    result["max_abs_difference"] = max_difference
    result["pass"] = len(errors) == 0

    return result


def compare_npz(
    canonical_file: Path,
    candidate_file: Path,
    tolerance: float,
) -> dict[str, object]:

    result: dict[str, object] = {
        "pass": True,
        "keys": {},
        "errors": [],
    }

    errors: list[str] = result["errors"]  # type: ignore[assignment]
    key_results: dict[str, object] = result["keys"]  # type: ignore[assignment]

    if not canonical_file.is_file():
        errors.append(
            f"canonical_npz_missing:{canonical_file}"
        )
        result["pass"] = False
        return result

    if not candidate_file.is_file():
        errors.append(
            f"candidate_npz_missing:{candidate_file}"
        )
        result["pass"] = False
        return result

    with np.load(
        canonical_file,
        allow_pickle=False,
    ) as canonical, np.load(
        candidate_file,
        allow_pickle=False,
    ) as candidate:

        canonical_keys = sorted(canonical.files)
        candidate_keys = sorted(candidate.files)

        if canonical_keys != candidate_keys:
            errors.append(
                "npz_keys_differ: "
                f"{canonical_keys} vs {candidate_keys}"
            )
            result["pass"] = False
            return result

        for key in canonical_keys:
            a = canonical[key]
            b = candidate[key]

            record: dict[str, object] = {
                "shape": list(a.shape),
                "dtype_canonical": str(a.dtype),
                "dtype_candidate": str(b.dtype),
                "pass": True,
            }

            if a.shape != b.shape:
                record["pass"] = False
                record["error"] = (
                    f"shape {a.shape} vs {b.shape}"
                )
                key_results[key] = record
                errors.append(f"{key}: shape")
                continue

            if a.dtype.kind == "b" or b.dtype.kind == "b":
                equal = np.array_equal(a, b)
                record["exact_equal"] = bool(equal)
                record["pass"] = bool(equal)

                if not equal:
                    errors.append(f"{key}: boolean_difference")

            elif (
                a.dtype.kind in "iufc"
                and b.dtype.kind in "iufc"
            ):
                a_float = a.astype(np.float64)
                b_float = b.astype(np.float64)

                exact = np.array_equal(
                    a_float,
                    b_float,
                    equal_nan=True,
                )

                close = np.allclose(
                    a_float,
                    b_float,
                    rtol=0.0,
                    atol=tolerance,
                    equal_nan=True,
                )

                finite = (
                    np.isfinite(a_float)
                    & np.isfinite(b_float)
                )

                if np.any(finite):
                    max_diff = float(
                        np.max(
                            np.abs(
                                a_float[finite]
                                - b_float[finite]
                            )
                        )
                    )
                else:
                    max_diff = 0.0

                record["exact_equal"] = bool(exact)
                record["max_abs_difference"] = max_diff
                record["pass"] = bool(close)

                if not close:
                    errors.append(
                        f"{key}: numeric_difference "
                        f"{max_diff} > {tolerance}"
                    )

            else:
                equal = np.array_equal(
                    a.astype(str),
                    b.astype(str),
                )

                record["exact_equal"] = bool(equal)
                record["pass"] = bool(equal)

                if not equal:
                    errors.append(
                        f"{key}: string_difference"
                    )

            key_results[key] = record

    result["pass"] = len(errors) == 0
    return result


def main() -> int:
    args = parse_args()

    release = args.canonical_release.resolve()
    run_root = args.candidate_run.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_processed = (
        release
        / "processed"
    )

    candidate_processed = (
        run_root
        / "processed"
        / args.dataset_group
        / args.system
    )

    if not canonical_processed.is_dir():
        raise RuntimeError(
            f"No existe canonical processed: "
            f"{canonical_processed}"
        )

    if not candidate_processed.is_dir():
        raise RuntimeError(
            f"No existe candidate processed: "
            f"{candidate_processed}"
        )

    status_file = run_root / "PIPELINE_STATUS.txt"

    if not status_file.is_file():
        raise RuntimeError(
            f"No existe PIPELINE_STATUS.txt: "
            f"{status_file}"
        )

    status_text = status_file.read_text()

    if "pipeline_status=PASS" not in status_text:
        raise RuntimeError(
            "El candidate run no tiene "
            "pipeline_status=PASS."
        )

    csv_results: list[dict[str, object]] = []
    archived_directory_status: list[dict[str, str]] = []

    for directory_name in SCIENTIFIC_DIRS:
        canonical_dir = (
            canonical_processed
            / directory_name
        )

        candidate_dir = (
            candidate_processed
            / directory_name
        )

        if not canonical_dir.is_dir():
            archived_directory_status.append(
                {
                    "directory": directory_name,
                    "status": "NOT_ARCHIVED_IN_REFERENCE_RELEASE",
                }
            )
            continue

        archived_directory_status.append(
            {
                "directory": directory_name,
                "status": "ARCHIVED_AND_COMPARED",
            }
        )

        if not candidate_dir.is_dir():
            raise RuntimeError(
                f"Falta en candidate un directorio "
                f"archivado en la release: {candidate_dir}"
            )

        canonical_csvs = sorted(
            canonical_dir.rglob("*.csv")
        )

        for canonical_file in canonical_csvs:
            relative = canonical_file.relative_to(
                canonical_dir
            )

            candidate_file = (
                candidate_dir
                / relative
            )

            csv_results.append(
                compare_csv(
                    canonical_file,
                    candidate_file,
                    args.numeric_tolerance,
                )
            )

    canonical_sensitivity = (
        release
        / "equilibration_sensitivity"
    )

    candidate_sensitivity = (
        run_root
        / "equilibration_sensitivity"
        / args.system
    )

    if canonical_sensitivity.is_dir():
        if not candidate_sensitivity.is_dir():
            raise RuntimeError(
                "La release contiene sensibilidad "
                "pero el candidate no."
            )

        for canonical_file in sorted(
            canonical_sensitivity.glob("*.csv")
        ):
            candidate_file = (
                candidate_sensitivity
                / canonical_file.name
            )

            csv_results.append(
                compare_csv(
                    canonical_file,
                    candidate_file,
                    args.numeric_tolerance,
                )
            )

    canonical_npz = (
        canonical_processed
        / "prefix_sequences"
        / "prefix_sequences_raw.npz"
    )

    candidate_npz = (
        candidate_processed
        / "prefix_sequences"
        / "prefix_sequences_raw.npz"
    )

    npz_result = compare_npz(
        canonical_npz,
        candidate_npz,
        args.npz_tolerance,
    )

    failed_csvs = [
        record
        for record in csv_results
        if not bool(record["pass"])
    ]

    overall_pass = (
        len(failed_csvs) == 0
        and bool(npz_result["pass"])
    )

    report = {
        "overall_pass": overall_pass,
        "system": args.system,
        "dataset_group": args.dataset_group,
        "canonical_release": str(release),
        "candidate_run": str(run_root),
        "numeric_tolerance": args.numeric_tolerance,
        "npz_tolerance": args.npz_tolerance,
        "n_csv_compared": len(csv_results),
        "n_csv_failed": len(failed_csvs),
        "archived_directory_status": archived_directory_status,
        "csv_results": csv_results,
        "npz": npz_result,
    }

    report_file = (
        output_dir
        / "system_release_regression.json"
    )

    report_file.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    summary_records = []

    for record in csv_results:
        summary_records.append(
            {
                "file": record["file"],
                "pass": record["pass"],
                "max_abs_difference": (
                    record["max_abs_difference"]
                ),
                "errors": "; ".join(
                    record["errors"]
                ),
            }
        )

    pd.DataFrame(
        summary_records
    ).to_csv(
        output_dir
        / "system_release_regression.csv",
        index=False,
    )

    print(
        "=" * 80
    )
    print(
        "REGRESION ORQUESTADOR VS RELEASE CONGELADA"
    )
    print(
        "=" * 80
    )
    print(f"System:              {args.system}")
    print(f"Canonical release:   {release}")
    print(f"Candidate run:       {run_root}")
    print(
        f"CSV comparados:      {len(csv_results)}"
    )
    print(
        f"CSV fallidos:        {len(failed_csvs)}"
    )
    print()

    print("DIRECTORIOS DE REFERENCIA:")
    for record in archived_directory_status:
        print(
            f"  {record['directory']:34s} "
            f"{record['status']}"
        )

    print()
    print("NPZ:")

    for key, record in npz_result["keys"].items():
        exact = record.get(
            "exact_equal",
            "NA",
        )
        max_diff = record.get(
            "max_abs_difference",
            "NA",
        )

        print(
            f"  {key:20s} "
            f"pass={record['pass']} "
            f"exact={exact} "
            f"max_diff={max_diff}"
        )

    if failed_csvs:
        print()
        print("CSV CON DIFERENCIAS:")

        for record in failed_csvs[:50]:
            print(
                f"  {record['file']}"
            )
            for error in record["errors"]:
                print(
                    f"    - {error}"
                )

    print()
    print(
        "REGRESION GLOBAL: "
        + (
            "PASS"
            if overall_pass
            else "FAIL"
        )
    )

    print()
    print(f"Reporte JSON: {report_file}")
    print(
        "Resumen CSV: "
        f"{output_dir / 'system_release_regression.csv'}"
    )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
