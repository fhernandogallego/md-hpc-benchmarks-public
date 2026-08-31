#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-npz",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--lopo-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--expected-dataset-sha",
        required=True,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def decode_strings(array):
    values = []

    for value in array.tolist():

        if isinstance(value, bytes):
            values.append(
                value.decode("utf-8")
            )
        else:
            values.append(
                str(value)
            )

    return values


def scale_targets(
    values,
    mask,
    mean,
    std,
):
    scaled = np.full(
        values.shape,
        np.nan,
        dtype=np.float64,
    )

    for target in range(
        values.shape[1]
    ):

        valid = mask[
            :,
            target,
        ]

        scaled[
            valid,
            target,
        ] = (
            values[
                valid,
                target,
            ]
            - mean[target]
        ) / std[target]

    return scaled


def main():
    args = parse_args()

    dataset = (
        args.dataset_npz
        .expanduser()
        .resolve()
    )

    lopo_path = (
        args.lopo_json
        .expanduser()
        .resolve()
    )

    output = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not dataset.is_file():
        raise FileNotFoundError(
            dataset
        )

    if not lopo_path.is_file():
        raise FileNotFoundError(
            lopo_path
        )

    if output.exists():
        raise RuntimeError(
            f"Output exists: {output}"
        )

    actual_dataset_sha = (
        sha256_file(dataset)
    )

    if (
        actual_dataset_sha
        != args.expected_dataset_sha
    ):
        raise RuntimeError(
            "Dataset SHA mismatch"
        )

    with np.load(
        dataset,
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
                f"Missing arrays: "
                f"{sorted(missing)}"
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
            f"Unexpected X_raw: "
            f"{X_raw.shape}"
        )

    if target_values.shape != (33, 5):
        raise RuntimeError(
            "Unexpected target shape"
        )

    if target_mask.shape != (33, 5):
        raise RuntimeError(
            "Unexpected mask shape"
        )

    if time_ns.shape != (201,):
        raise RuntimeError(
            "Unexpected time grid"
        )

    if not np.isfinite(
        X_raw
    ).all():
        raise RuntimeError(
            "Non-finite raw features"
        )

    if not np.isfinite(
        target_values[
            target_mask
        ]
    ).all():
        raise RuntimeError(
            "Non-finite valid targets"
        )

    if not np.isnan(
        target_values[
            ~target_mask
        ]
    ).all():
        raise RuntimeError(
            "Invalid targets must be NaN"
        )

    lopo = json.loads(
        lopo_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        lopo.get("strategy")
        != "leave_one_protein_out"
    ):
        raise RuntimeError(
            "Unexpected split strategy"
        )

    folds = lopo.get(
        "folds",
        []
    )

    if len(folds) != 11:
        raise RuntimeError(
            "Expected 11 LOPO folds"
        )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    folds_dir = (
        output
        / "folds"
    )

    folds_dir.mkdir()

    scaler_records = []
    manifest_rows = []

    max_feature_mean = 0.0
    max_feature_std_error = 0.0

    max_target_mean = 0.0
    max_target_std_error = 0.0

    written_files = []

    for fold in folds:

        fold_id = str(
            fold["fold_id"]
        )

        fold_index = int(
            fold["fold_index"]
        )

        test_system = str(
            fold["test_system"]
        )

        train_indices = np.asarray(
            fold["train_indices"],
            dtype=np.int64,
        )

        test_indices = np.asarray(
            fold["test_indices"],
            dtype=np.int64,
        )

        if len(train_indices) != 30:
            raise RuntimeError(
                f"{fold_id}: "
                "train size != 30"
            )

        if len(test_indices) != 3:
            raise RuntimeError(
                f"{fold_id}: "
                "test size != 3"
            )

        if (
            set(
                train_indices.tolist()
            )
            &
            set(
                test_indices.tolist()
            )
        ):
            raise RuntimeError(
                f"{fold_id}: "
                "train/test overlap"
            )

        if sorted(
            np.concatenate(
                [
                    train_indices,
                    test_indices,
                ]
            ).tolist()
        ) != list(range(33)):
            raise RuntimeError(
                f"{fold_id}: "
                "train/test do not partition "
                "all 33 samples"
            )

        train_systems = list(
            dict.fromkeys(
                system_ids[
                    train_indices
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

        if len(train_systems) != 10:
            raise RuntimeError(
                f"{fold_id}: "
                "train proteins != 10"
            )

        if test_systems != [
            test_system
        ]:
            raise RuntimeError(
                f"{fold_id}: "
                "test-system mismatch"
            )

        if test_system in train_systems:
            raise RuntimeError(
                f"{fold_id}: "
                "outer test leaked into train"
            )

        X_train_raw = X_raw[
            train_indices
        ]

        feature_mean = (
            X_train_raw.mean(
                axis=(0, 1)
            )
        )

        feature_std = (
            X_train_raw.std(
                axis=(0, 1),
                ddof=0,
            )
        )

        if not np.isfinite(
            feature_mean
        ).all():
            raise RuntimeError(
                f"{fold_id}: "
                "invalid feature mean"
            )

        if (
            not np.isfinite(
                feature_std
            ).all()
            or np.any(
                feature_std <= 0.0
            )
        ):
            raise RuntimeError(
                f"{fold_id}: "
                "invalid feature std"
            )

        feature_n = (
            len(train_indices)
            * X_raw.shape[1]
        )

        if feature_n != 6030:
            raise RuntimeError(
                f"{fold_id}: "
                f"feature n={feature_n}"
            )

        target_mean = []
        target_std = []
        target_n = []

        for target_index, target_name in enumerate(
            target_names.tolist()
        ):

            valid = target_mask[
                train_indices,
                target_index,
            ]

            values = target_values[
                train_indices,
                target_index,
            ][valid]

            if len(values) == 0:
                raise RuntimeError(
                    f"{fold_id}: "
                    f"no valid {target_name}"
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

            if (
                not np.isfinite(mean)
                or
                not np.isfinite(std)
                or
                std <= 0.0
            ):
                raise RuntimeError(
                    f"{fold_id}: "
                    f"invalid scaler "
                    f"for {target_name}"
                )

            target_mean.append(
                mean
            )

            target_std.append(
                std
            )

            target_n.append(
                int(len(values))
            )

        expected_target_counts = [
            int(
                fold[
                    "train_target_counts"
                ][name]
            )
            for name
            in target_names.tolist()
        ]

        if (
            target_n
            != expected_target_counts
        ):
            raise RuntimeError(
                f"{fold_id}: "
                "target counts mismatch"
            )

        feature_mean = np.asarray(
            feature_mean,
            dtype=np.float64,
        )

        feature_std = np.asarray(
            feature_std,
            dtype=np.float64,
        )

        target_mean_array = np.asarray(
            target_mean,
            dtype=np.float64,
        )

        target_std_array = np.asarray(
            target_std,
            dtype=np.float64,
        )

        X_train = (
            X_train_raw
            - feature_mean[
                None,
                None,
                :,
            ]
        ) / feature_std[
            None,
            None,
            :,
        ]

        mask_train = target_mask[
            train_indices
        ].copy()

        y_train = scale_targets(
            target_values[
                train_indices
            ],
            mask_train,
            target_mean_array,
            target_std_array,
        )

        if not np.isfinite(
            X_train
        ).all():
            raise RuntimeError(
                f"{fold_id}: "
                "scaled X non-finite"
            )

        if not np.isfinite(
            y_train[
                mask_train
            ]
        ).all():
            raise RuntimeError(
                f"{fold_id}: "
                "scaled valid y non-finite"
            )

        if not np.isnan(
            y_train[
                ~mask_train
            ]
        ).all():
            raise RuntimeError(
                f"{fold_id}: "
                "masked y was imputed"
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

        max_feature_mean = max(
            max_feature_mean,
            float(
                np.max(
                    np.abs(
                        train_feature_mean
                    )
                )
            ),
        )

        max_feature_std_error = max(
            max_feature_std_error,
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
                target_index,
            ]

            values = y_train[
                valid,
                target_index,
            ]

            max_target_mean = max(
                max_target_mean,
                abs(
                    float(
                        np.mean(values)
                    )
                ),
            )

            max_target_std_error = max(
                max_target_std_error,
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
            / f"{fold_id}.npz"
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

            time_ns=(
                time_ns
            ),

            feature_names=(
                feature_names
            ),

            target_names=(
                target_names
            ),

            feature_mean=(
                feature_mean
            ),

            feature_std=(
                feature_std
            ),

            target_mean=(
                target_mean_array
            ),

            target_std=(
                target_std_array
            ),

            target_valid_fit_counts=(
                np.asarray(
                    target_n,
                    dtype=np.int64,
                )
            ),
        )

        fold_sha = sha256_file(
            fold_path
        )

        written_files.append(
            fold_path
        )

        scaler_records.append(
            {
                "fold_id":
                    fold_id,

                "outer_fold":
                    fold_index,

                "outer_test_system":
                    test_system,

                "fit_role":
                    "outer_train_only",

                "n_fit_samples":
                    30,

                "n_fit_systems":
                    10,

                "n_fit_timepoints_per_sample":
                    201,

                "feature_n_values_per_feature":
                    6030,

                "fit_sample_indices":
                    train_indices.tolist(),

                "fit_sample_ids":
                    sample_ids[
                        train_indices
                    ].tolist(),

                "fit_systems":
                    train_systems,

                "excluded_outer_test_indices":
                    test_indices.tolist(),

                "excluded_outer_test_system":
                    test_system,

                "feature_scaler": {
                    "method":
                        "standard_score",
                    "ddof":
                        0,
                    "feature_names":
                        feature_names.tolist(),
                    "mean":
                        feature_mean.tolist(),
                    "std":
                        feature_std.tolist(),
                },

                "target_scaler": {
                    "method":
                        "masked_standard_score",
                    "ddof":
                        0,
                    "target_names":
                        target_names.tolist(),
                    "n_valid_fit_values":
                        target_n,
                    "mean":
                        target_mean_array.tolist(),
                    "std":
                        target_std_array.tolist(),
                },

                "outer_test_materialized":
                    False,
            }
        )

        manifest_rows.append(
            {
                "fold_id":
                    fold_id,

                "outer_fold":
                    fold_index,

                "outer_test_system":
                    test_system,

                "n_train_samples":
                    30,

                "n_train_systems":
                    10,

                "n_outer_test_samples_materialized":
                    0,

                "n_valid_train_targets":
                    int(
                        mask_train.sum()
                    ),

                "package":
                    str(fold_path),

                "package_sha256":
                    fold_sha,
            }
        )

        print(
            f"{fold_id}:",
            f"test={test_system}",
            "train=30",
            "proteins=10",
            "feature_n=6030",
            "outer_test_arrays=0",
            "PASS",
        )

    scalers_path = (
        output
        / "outer_scalers.json"
    )

    scalers_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy":
                    "outer_lopo_train_only_scaling",
                "n_outer_folds":
                    11,
                "scalers":
                    scaler_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = (
        output
        / "outer_training_manifest.csv"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        fieldnames = [
            "fold_id",
            "outer_fold",
            "outer_test_system",
            "n_train_samples",
            "n_train_systems",
            "n_outer_test_samples_materialized",
            "n_valid_train_targets",
            "package",
            "package_sha256",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    metadata_path = (
        output
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            {
                "created_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "analysis_type":
                    "outer_train_preprocessing",

                "dataset":
                    str(dataset),

                "dataset_sha256":
                    actual_dataset_sha,

                "lopo_json":
                    str(lopo_path),

                "lopo_json_sha256":
                    sha256_file(
                        lopo_path
                    ),

                "n_outer_folds":
                    11,

                "samples_per_outer_train":
                    30,

                "systems_per_outer_train":
                    10,

                "outer_test_samples_materialized":
                    0,

                "outer_test_targets_materialized":
                    False,

                "feature_scaler":
                    "standard_score_ddof0",

                "target_scaler":
                    "masked_standard_score_ddof0",

                "missing_target_policy":
                    "NaN_preserved_no_imputation",

                "max_abs_scaled_train_feature_mean":
                    max_feature_mean,

                "max_abs_scaled_train_feature_std_error":
                    max_feature_std_error,

                "max_abs_scaled_train_target_mean":
                    max_target_mean,

                "max_abs_scaled_train_target_std_error":
                    max_target_std_error,

                "outer_test_evaluated":
                    False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    written_files.extend(
        [
            scalers_path,
            manifest_path,
            metadata_path,
        ]
    )

    sums_path = (
        output
        / "SHA256SUMS.txt"
    )

    with sums_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in sorted(
            written_files,
            key=lambda item:
                str(item),
        ):

            relative = (
                path.relative_to(
                    output
                )
            )

            handle.write(
                sha256_file(path)
                + "  "
                + str(relative)
                + "\n"
            )

    print()
    print(
        "outer_packages:",
        len(manifest_rows),
        "/ 11",
    )

    print(
        "max_abs_scaled_feature_mean:",
        max_feature_mean,
    )

    print(
        "max_abs_scaled_feature_std_error:",
        max_feature_std_error,
    )

    print(
        "max_abs_scaled_target_mean:",
        max_target_mean,
    )

    print(
        "max_abs_scaled_target_std_error:",
        max_target_std_error,
    )

    print(
        "outer_test_arrays_materialized:",
        0,
    )

    print(
        "OUTER_TEST_EVALUATED: NO"
    )

    print(
        "OUTER_TRAIN_PREPROCESSING: PASS"
    )


if __name__ == "__main__":
    main()
