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
            "Materialize leakage-free scaled inner-train "
            "and inner-validation arrays for nested CV."
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
        "--scalers",
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
    result = []

    for value in array.tolist():
        if isinstance(value, bytes):
            result.append(
                value.decode("utf-8")
            )
        else:
            result.append(str(value))

    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def scale_targets(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    scaled = np.full_like(
        values,
        np.nan,
        dtype=np.float64,
    )

    for target_index in range(
        values.shape[1]
    ):
        valid = mask[
            :,
            target_index
        ]

        scaled[
            valid,
            target_index
        ] = (
            values[
                valid,
                target_index
            ]
            - mean[
                target_index
            ]
        ) / std[
            target_index
        ]

    return scaled


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

    scalers_path = (
        args.scalers
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    for path in [
        dataset_path,
        nested_path,
        scalers_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

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
            "time_ns",
            "system_ids",
            "replicas",
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

        time_ns = (
            archive["time_ns"]
            .astype(np.float64)
            .copy()
        )

        system_ids = np.asarray(
            decode_strings(
                archive["system_ids"]
            )
        )

        replicas = (
            archive["replicas"]
            .astype(np.int64)
            .copy()
        )

        sample_ids = np.asarray(
            decode_strings(
                archive["sample_ids"]
            )
        )

        feature_names = np.asarray(
            decode_strings(
                archive["feature_names"]
            )
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

        target_names = np.asarray(
            decode_strings(
                archive["target_names"]
            )
        )

    if X_raw.shape != (33, 201, 8):
        raise RuntimeError(
            f"Unexpected X shape: {X_raw.shape}"
        )

    if target_values.shape != (33, 5):
        raise RuntimeError(
            "Unexpected target shape."
        )

    if target_mask.shape != (33, 5):
        raise RuntimeError(
            "Unexpected mask shape."
        )

    if time_ns.shape != (201,):
        raise RuntimeError(
            "Unexpected time grid."
        )

    if not np.isfinite(X_raw).all():
        raise RuntimeError(
            "X_raw contains non-finite values."
        )

    if not np.isfinite(
        target_values[target_mask]
    ).all():
        raise RuntimeError(
            "Valid targets are non-finite."
        )

    if not np.isnan(
        target_values[~target_mask]
    ).all():
        raise RuntimeError(
            "Invalid targets must be NaN."
        )

    nested = json.loads(
        nested_path.read_text(
            encoding="utf-8"
        )
    )

    scaler_data = json.loads(
        scalers_path.read_text(
            encoding="utf-8"
        )
    )

    if nested.get(
        "n_nested_folds"
    ) != 55:
        raise RuntimeError(
            "Nested CV does not contain 55 folds."
        )

    if scaler_data.get(
        "n_nested_folds"
    ) != 55:
        raise RuntimeError(
            "Scaler package does not contain 55 folds."
        )

    scaler_map = {
        record["nested_fold_id"]: record
        for record
        in scaler_data["scalers"]
    }

    if len(scaler_map) != 55:
        raise RuntimeError(
            "Scaler IDs are not unique."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    folds_dir = (
        output_dir
        / "folds"
    )

    folds_dir.mkdir()

    manifest_rows = []
    n_written = 0

    max_train_feature_mean = 0.0
    max_train_feature_std_error = 0.0

    max_train_target_mean = 0.0
    max_train_target_std_error = 0.0

    for outer in nested[
        "outer_folds"
    ]:

        outer_test_system = outer[
            "outer_test_system"
        ]

        for inner in outer[
            "inner_folds"
        ]:

            nested_fold_id = inner[
                "nested_fold_id"
            ]

            scaler = scaler_map.get(
                nested_fold_id
            )

            if scaler is None:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "missing scaler."
                )

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

            if scaler[
                "fit_sample_indices"
            ] != train_indices.tolist():
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "scaler fit indices differ."
                )

            feature_mean = np.asarray(
                scaler[
                    "feature_scaler"
                ][
                    "mean"
                ],
                dtype=np.float64,
            )

            feature_std = np.asarray(
                scaler[
                    "feature_scaler"
                ][
                    "std"
                ],
                dtype=np.float64,
            )

            target_mean = np.asarray(
                scaler[
                    "target_scaler"
                ][
                    "mean"
                ],
                dtype=np.float64,
            )

            target_std = np.asarray(
                scaler[
                    "target_scaler"
                ][
                    "std"
                ],
                dtype=np.float64,
            )

            if np.any(
                feature_std <= 0
            ):
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "invalid feature std."
                )

            if np.any(
                target_std <= 0
            ):
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "invalid target std."
                )

            X_train = (
                X_raw[
                    train_indices
                ]
                - feature_mean[
                    None,
                    None,
                    :
                ]
            ) / feature_std[
                None,
                None,
                :
            ]

            X_valid = (
                X_raw[
                    valid_indices
                ]
                - feature_mean[
                    None,
                    None,
                    :
                ]
            ) / feature_std[
                None,
                None,
                :
            ]

            y_train = scale_targets(
                target_values[
                    train_indices
                ],
                target_mask[
                    train_indices
                ],
                target_mean,
                target_std,
            )

            y_valid = scale_targets(
                target_values[
                    valid_indices
                ],
                target_mask[
                    valid_indices
                ],
                target_mean,
                target_std,
            )

            mask_train = target_mask[
                train_indices
            ].copy()

            mask_valid = target_mask[
                valid_indices
            ].copy()

            if not np.isfinite(
                X_train
            ).all():
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "non-finite X_train."
                )

            if not np.isfinite(
                X_valid
            ).all():
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "non-finite X_valid."
                )

            if not np.isfinite(
                y_train[
                    mask_train
                ]
            ).all():
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "non-finite valid y_train."
                )

            if not np.isfinite(
                y_valid[
                    mask_valid
                ]
            ).all():
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "non-finite valid y_valid."
                )

            if not np.isnan(
                y_train[
                    ~mask_train
                ]
            ).all():
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "invalid y_train not NaN."
                )

            if not np.isnan(
                y_valid[
                    ~mask_valid
                ]
            ).all():
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "invalid y_valid not NaN."
                )

            train_feature_mean = (
                X_train.mean(
                    axis=(0, 1)
                )
            )

            train_feature_std = (
                X_train.std(
                    axis=(0, 1),
                    ddof=0,
                )
            )

            max_train_feature_mean = max(
                max_train_feature_mean,
                float(
                    np.max(
                        np.abs(
                            train_feature_mean
                        )
                    )
                ),
            )

            max_train_feature_std_error = max(
                max_train_feature_std_error,
                float(
                    np.max(
                        np.abs(
                            train_feature_std
                            - 1.0
                        )
                    )
                ),
            )

            for target_index in range(
                len(target_names)
            ):

                valid = mask_train[
                    :,
                    target_index
                ]

                values = y_train[
                    valid,
                    target_index
                ]

                max_train_target_mean = max(
                    max_train_target_mean,
                    abs(
                        float(
                            np.mean(values)
                        )
                    ),
                )

                max_train_target_std_error = max(
                    max_train_target_std_error,
                    abs(
                        float(
                            np.std(
                                values,
                                ddof=0,
                            )
                        )
                        - 1.0
                    ),
                )

            fold_path = (
                folds_dir
                / f"{nested_fold_id}.npz"
            )

            np.savez(
                fold_path,

                X_train_scaled=(
                    X_train
                ),

                y_train_scaled=(
                    y_train
                ),

                mask_train=(
                    mask_train
                ),

                train_indices=(
                    train_indices
                ),

                train_sample_ids=(
                    sample_ids[
                        train_indices
                    ]
                ),

                train_system_ids=(
                    system_ids[
                        train_indices
                    ]
                ),

                train_replicas=(
                    replicas[
                        train_indices
                    ]
                ),

                X_valid_scaled=(
                    X_valid
                ),

                y_valid_scaled=(
                    y_valid
                ),

                mask_valid=(
                    mask_valid
                ),

                valid_indices=(
                    valid_indices
                ),

                valid_sample_ids=(
                    sample_ids[
                        valid_indices
                    ]
                ),

                valid_system_ids=(
                    system_ids[
                        valid_indices
                    ]
                ),

                valid_replicas=(
                    replicas[
                        valid_indices
                    ]
                ),

                time_ns=(
                    time_ns
                ),

                feature_names=(
                    feature_names
                ),

                target_names=(
                    target_names
                ),
            )

            fold_sha = sha256_file(
                fold_path
            )

            manifest_rows.append(
                {
                    "nested_fold_id": (
                        nested_fold_id
                    ),
                    "outer_fold_id": (
                        inner[
                            "outer_fold_id"
                        ]
                    ),
                    "inner_fold_id": (
                        inner[
                            "inner_fold_id"
                        ]
                    ),
                    "outer_test_system": (
                        outer_test_system
                    ),
                    "n_train": 24,
                    "n_valid": 6,
                    "n_outer_test_materialized": 0,
                    "n_train_valid_targets": int(
                        mask_train.sum()
                    ),
                    "n_valid_valid_targets": int(
                        mask_valid.sum()
                    ),
                    "artifact": str(
                        Path("folds")
                        / fold_path.name
                    ),
                    "artifact_sha256": (
                        fold_sha
                    ),
                }
            )

            n_written += 1

    if n_written != 55:
        raise RuntimeError(
            f"Expected 55 folds, wrote {n_written}."
        )

    manifest_path = (
        output_dir
        / "training_package_manifest.csv"
    )

    columns = [
        "nested_fold_id",
        "outer_fold_id",
        "inner_fold_id",
        "outer_test_system",
        "n_train",
        "n_valid",
        "n_outer_test_materialized",
        "n_train_valid_targets",
        "n_valid_valid_targets",
        "artifact",
        "artifact_sha256",
    ]

    with manifest_path.open(
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
            manifest_rows
        )

    metadata = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "package_type": (
            "nested_scaled_training_validation"
        ),

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

        "scalers": str(
            scalers_path
        ),

        "scalers_sha256": (
            sha256_file(
                scalers_path
            )
        ),

        "n_nested_folds": 55,

        "train_shape": [
            24,
            201,
            8,
        ],

        "valid_shape": [
            6,
            201,
            8,
        ],

        "target_train_shape": [
            24,
            5,
        ],

        "target_valid_shape": [
            6,
            5,
        ],

        "feature_names": (
            feature_names.tolist()
        ),

        "target_names": (
            target_names.tolist()
        ),

        "transform_policy": {
            "feature_transform": (
                "z=(x-inner_train_mean)/"
                "inner_train_std"
            ),
            "target_transform": (
                "masked z-score using "
                "inner_train target statistics"
            ),
            "invalid_targets": (
                "remain NaN"
            ),
        },

        "leakage_policy": {
            "scalers_fit_on": (
                "inner_train_only"
            ),
            "inner_valid_transform_only": True,
            "outer_test_materialized": False,
            "outer_test_used_for_model_selection": False,
        },

        "data_policy": {
            "raw_dataset_modified": False,
            "target_masks_preserved": True,
            "imputation_applied": False,
        },
    }

    metadata_path = (
        output_dir
        / "training_package_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    checksum_paths = (
        list(
            sorted(
                folds_dir.glob(
                    "*.npz"
                )
            )
        )
        + [
            manifest_path,
            metadata_path,
        ]
    )

    checksums_path = (
        output_dir
        / "SHA256SUMS.txt"
    )

    with checksums_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in checksum_paths:

            relative = (
                path.relative_to(
                    output_dir
                )
            )

            handle.write(
                f"{sha256_file(path)}  "
                f"{relative}\n"
            )

    print(
        "========================================"
    )
    print(
        "NESTED TRAINING PACKAGE BUILD: PASS"
    )
    print(
        "========================================"
    )

    print(
        "Nested folds:",
        n_written,
    )

    print(
        "Train shape/fold:",
        (24, 201, 8),
    )

    print(
        "Valid shape/fold:",
        (6, 201, 8),
    )

    print(
        "Target train shape/fold:",
        (24, 5),
    )

    print(
        "Target valid shape/fold:",
        (6, 5),
    )

    print(
        "Outer-test materialized:",
        "NO",
    )

    print(
        "Invalid targets remain NaN:",
        "YES",
    )

    print(
        "Max abs train feature mean:",
        max_train_feature_mean,
    )

    print(
        "Max train feature std error:",
        max_train_feature_std_error,
    )

    print(
        "Max abs train target mean:",
        max_train_target_mean,
    )

    print(
        "Max train target std error:",
        max_train_target_std_error,
    )

    print(
        "Output:",
        output_dir,
    )


if __name__ == "__main__":
    main()
