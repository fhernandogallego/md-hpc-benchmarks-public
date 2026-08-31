from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit leakage-free standardization parameters "
            "for deterministic nested protein-group CV."
        )
    )

    parser.add_argument(
        "--dataset-npz",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--nested-cv",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def decode_strings(
    array: np.ndarray,
) -> list[str]:
    out = []

    for value in array.tolist():
        if isinstance(value, bytes):
            out.append(
                value.decode("utf-8")
            )
        else:
            out.append(str(value))

    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    args = parse_args()

    dataset_path = (
        args.dataset_npz
        .expanduser()
        .resolve()
    )

    nested_path = (
        args.nested_cv
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            dataset_path
        )

    if not nested_path.is_file():
        raise FileNotFoundError(
            nested_path
        )

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: {output_dir}"
        )

    with np.load(
        dataset_path,
        allow_pickle=False,
    ) as archive:

        required = {
            "X_raw",
            "system_ids",
            "sample_ids",
            "feature_names",
            "target_values",
            "target_mask",
            "target_names",
        }

        missing = required.difference(
            archive.files
        )

        if missing:
            raise RuntimeError(
                f"Missing arrays: {sorted(missing)}"
            )

        X_raw = (
            archive["X_raw"]
            .astype(np.float64)
            .copy()
        )

        system_ids = np.asarray(
            decode_strings(
                archive["system_ids"]
            )
        )

        sample_ids = decode_strings(
            archive["sample_ids"]
        )

        feature_names = decode_strings(
            archive["feature_names"]
        )

        target_values = (
            archive["target_values"]
            .astype(np.float64)
            .copy()
        )

        target_mask = (
            archive["target_mask"]
            .astype(bool)
            .copy()
        )

        target_names = decode_strings(
            archive["target_names"]
        )

    if X_raw.shape != (33, 201, 8):
        raise RuntimeError(
            f"Unexpected X_raw shape: "
            f"{X_raw.shape}"
        )

    if target_values.shape != (33, 5):
        raise RuntimeError(
            f"Unexpected target shape: "
            f"{target_values.shape}"
        )

    if target_mask.shape != (33, 5):
        raise RuntimeError(
            f"Unexpected target mask shape: "
            f"{target_mask.shape}"
        )

    if not np.isfinite(X_raw).all():
        raise RuntimeError(
            "X_raw contains non-finite values."
        )

    if not np.isfinite(
        target_values[target_mask]
    ).all():
        raise RuntimeError(
            "Valid targets contain non-finite values."
        )

    if not np.isnan(
        target_values[~target_mask]
    ).all():
        raise RuntimeError(
            "Invalid targets must remain NaN."
        )

    nested = json.loads(
        nested_path.read_text(
            encoding="utf-8"
        )
    )

    if nested.get(
        "strategy"
    ) != "nested_protein_group_cv":
        raise RuntimeError(
            "Unexpected nested CV strategy."
        )

    if nested.get(
        "n_nested_folds"
    ) != 55:
        raise RuntimeError(
            "Expected 55 nested folds."
        )

    scaler_records = []
    summary_rows = []

    for outer in nested["outer_folds"]:

        outer_fold_id = outer[
            "outer_fold_id"
        ]

        outer_test_system = outer[
            "outer_test_system"
        ]

        for inner in outer[
            "inner_folds"
        ]:

            inner_fold_id = inner[
                "inner_fold_id"
            ]

            nested_fold_id = inner[
                "nested_fold_id"
            ]

            train_indices = np.asarray(
                inner[
                    "inner_train_indices"
                ],
                dtype=np.int64,
            )

            valid_indices = np.asarray(
                inner[
                    "inner_valid_indices"
                ],
                dtype=np.int64,
            )

            test_indices = np.asarray(
                inner[
                    "outer_test_indices"
                ],
                dtype=np.int64,
            )

            if len(train_indices) != 24:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "train size != 24."
                )

            if len(valid_indices) != 6:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "valid size != 6."
                )

            if len(test_indices) != 3:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "test size != 3."
                )

            if (
                set(train_indices)
                & set(valid_indices)
            ):
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "train/valid overlap."
                )

            if (
                set(train_indices)
                & set(test_indices)
            ):
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "train/test overlap."
                )

            if (
                set(valid_indices)
                & set(test_indices)
            ):
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "valid/test overlap."
                )

            train_systems = list(
                dict.fromkeys(
                    system_ids[
                        train_indices
                    ].tolist()
                )
            )

            valid_systems = list(
                dict.fromkeys(
                    system_ids[
                        valid_indices
                    ].tolist()
                )
            )

            test_systems = list(
                dict.fromkeys(
                    system_ids[
                        test_indices
                    ].tolist()
                )
            )

            if len(train_systems) != 8:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "train proteins != 8."
                )

            if len(valid_systems) != 2:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "valid proteins != 2."
                )

            if test_systems != [
                outer_test_system
            ]:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "outer test mismatch."
                )

            X_train = X_raw[
                train_indices
            ]

            feature_mean = (
                X_train.mean(
                    axis=(0, 1)
                )
            )

            feature_std = (
                X_train.std(
                    axis=(0, 1),
                    ddof=0,
                )
            )

            feature_n = (
                X_train.shape[0]
                * X_train.shape[1]
            )

            if not np.isfinite(
                feature_mean
            ).all():
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "invalid feature means."
                )

            if not np.isfinite(
                feature_std
            ).all():
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "invalid feature std."
                )

            if np.any(
                feature_std <= 0.0
            ):
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "zero/non-positive "
                    "feature std."
                )

            target_mean = []
            target_std = []
            target_n = []

            for target_index, target_name in enumerate(
                target_names
            ):

                mask = target_mask[
                    train_indices,
                    target_index,
                ]

                values = target_values[
                    train_indices,
                    target_index,
                ][mask]

                if len(values) == 0:
                    raise RuntimeError(
                        f"{nested_fold_id}: "
                        f"{target_name} has "
                        "no training values."
                    )

                if not np.isfinite(
                    values
                ).all():
                    raise RuntimeError(
                        f"{nested_fold_id}: "
                        f"{target_name} contains "
                        "invalid training values."
                    )

                mean = float(
                    np.mean(values)
                )

                std = float(
                    np.std(
                        values,
                        ddof=0,
                    )
                )

                if not np.isfinite(mean):
                    raise RuntimeError(
                        f"{nested_fold_id}: "
                        f"{target_name} mean "
                        "is invalid."
                    )

                if (
                    not np.isfinite(std)
                    or std <= 0.0
                ):
                    raise RuntimeError(
                        f"{nested_fold_id}: "
                        f"{target_name} std "
                        "is zero/invalid."
                    )

                target_mean.append(mean)
                target_std.append(std)
                target_n.append(
                    int(len(values))
                )

            expected_counts = [
                int(
                    inner[
                        "inner_train_target_counts"
                    ][name]
                )
                for name in target_names
            ]

            if (
                target_n
                != expected_counts
            ):
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "target fit counts differ "
                    "from nested CV."
                )

            record = {
                "nested_fold_id": (
                    nested_fold_id
                ),
                "outer_fold_id": (
                    outer_fold_id
                ),
                "inner_fold_id": (
                    inner_fold_id
                ),
                "outer_test_system": (
                    outer_test_system
                ),
                "fit_role": (
                    "inner_train_only"
                ),
                "fit_sample_indices": (
                    train_indices.tolist()
                ),
                "fit_sample_ids": [
                    sample_ids[i]
                    for i in train_indices
                ],
                "fit_systems": (
                    train_systems
                ),
                "excluded_valid_systems": (
                    valid_systems
                ),
                "excluded_outer_test_systems": (
                    test_systems
                ),
                "n_fit_samples": 24,
                "n_fit_timepoints_per_sample": 201,
                "feature_n_values_per_feature": (
                    int(feature_n)
                ),
                "feature_scaler": {
                    "method": (
                        "standard_score"
                    ),
                    "ddof": 0,
                    "feature_names": (
                        feature_names
                    ),
                    "mean": [
                        float(value)
                        for value
                        in feature_mean
                    ],
                    "std": [
                        float(value)
                        for value
                        in feature_std
                    ],
                },
                "target_scaler": {
                    "method": (
                        "masked_standard_score"
                    ),
                    "ddof": 0,
                    "target_names": (
                        target_names
                    ),
                    "n_valid_fit_values": (
                        target_n
                    ),
                    "mean": (
                        target_mean
                    ),
                    "std": (
                        target_std
                    ),
                },
            }

            scaler_records.append(
                record
            )

            for (
                name,
                mean,
                std,
            ) in zip(
                feature_names,
                feature_mean,
                feature_std,
            ):
                summary_rows.append(
                    {
                        "nested_fold_id": (
                            nested_fold_id
                        ),
                        "kind": "feature",
                        "name": name,
                        "n_fit_values": (
                            feature_n
                        ),
                        "mean": (
                            float(mean)
                        ),
                        "std": (
                            float(std)
                        ),
                    }
                )

            for (
                name,
                n,
                mean,
                std,
            ) in zip(
                target_names,
                target_n,
                target_mean,
                target_std,
            ):
                summary_rows.append(
                    {
                        "nested_fold_id": (
                            nested_fold_id
                        ),
                        "kind": "target",
                        "name": name,
                        "n_fit_values": n,
                        "mean": mean,
                        "std": std,
                    }
                )

    if len(
        scaler_records
    ) != 55:
        raise RuntimeError(
            "Scaler record count != 55."
        )

    if len(
        {
            record["nested_fold_id"]
            for record in scaler_records
        }
    ) != 55:
        raise RuntimeError(
            "Duplicate nested fold IDs."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    scalers_path = (
        output_dir
        / "nested_scalers.json"
    )

    metadata_path = (
        output_dir
        / "nested_scalers_metadata.json"
    )

    summary_path = (
        output_dir
        / "nested_scaler_summary.csv"
    )

    scalers_path.write_text(
        json.dumps(
            {
                "scaler_type": (
                    "fold_specific_"
                    "inner_train_standardization"
                ),
                "n_nested_folds": 55,
                "scalers": scaler_records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_npz": str(
            dataset_path
        ),
        "dataset_npz_sha256": (
            sha256_file(
                dataset_path
            )
        ),
        "nested_cv": str(
            nested_path
        ),
        "nested_cv_sha256": (
            sha256_file(
                nested_path
            )
        ),
        "n_nested_folds": 55,
        "feature_names": (
            feature_names
        ),
        "target_names": (
            target_names
        ),
        "feature_fit_policy": (
            "inner_train samples only; "
            "all 201 timepoints pooled "
            "per feature"
        ),
        "target_fit_policy": (
            "inner_train samples only; "
            "target_mask=True values only"
        ),
        "feature_ddof": 0,
        "target_ddof": 0,
        "transform_formula": (
            "z=(x-mean)/std"
        ),
        "invalid_target_policy": (
            "remain NaN; never imputed; "
            "excluded from scaler fit"
        ),
        "leakage_policy": {
            "inner_valid_used_for_fit": False,
            "outer_test_used_for_fit": False,
            "outer_test_target_values_used": False,
            "fit_role": (
                "inner_train_only"
            ),
        },
        "data_policy": {
            "raw_dataset_modified": False,
            "normalized_arrays_saved": False,
            "imputation_applied": False,
            "target_mask_modified": False,
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    columns = [
        "nested_fold_id",
        "kind",
        "name",
        "n_fit_values",
        "mean",
        "std",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    checksum_files = [
        scalers_path,
        metadata_path,
        summary_path,
    ]

    checksums_path = (
        output_dir
        / "SHA256SUMS.txt"
    )

    with checksums_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            checksum_files,
            key=lambda p: p.name,
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )

    feature_means = np.asarray(
        [
            r["feature_scaler"]["mean"]
            for r in scaler_records
        ],
        dtype=np.float64,
    )

    feature_stds = np.asarray(
        [
            r["feature_scaler"]["std"]
            for r in scaler_records
        ],
        dtype=np.float64,
    )

    target_stds = np.asarray(
        [
            r["target_scaler"]["std"]
            for r in scaler_records
        ],
        dtype=np.float64,
    )

    print(
        "========================================"
    )
    print("NESTED SCALER BUILD: PASS")
    print(
        "========================================"
    )
    print("Nested folds: 55")
    print(
        "Fit samples/fold: 24"
    )
    print(
        "Feature values/feature/fold:",
        24 * 201,
    )
    print(
        "Feature scaler fit role: "
        "inner_train_only"
    )
    print(
        "Target scaler fit role: "
        "inner_train_only + valid mask"
    )
    print(
        "Inner-valid used for fit: NO"
    )
    print(
        "Outer-test used for fit: NO"
    )
    print(
        "All feature std > 0:",
        bool(
            np.all(
                feature_stds > 0
            )
        ),
    )
    print(
        "All target std > 0:",
        bool(
            np.all(
                target_stds > 0
            )
        ),
    )
    print(
        "Output:",
        output_dir,
    )


if __name__ == "__main__":
    main()
