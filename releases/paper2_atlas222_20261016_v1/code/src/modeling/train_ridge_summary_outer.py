#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ALPHA = 1.0

SUMMARY_STATS = (
    "mean",
    "std",
    "min",
    "max",
    "last",
    "slope",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def decode_strings(array):
    out = []

    for value in array.tolist():
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))

    return out


def summarize_sequence(X):
    if X.ndim != 3:
        raise RuntimeError("Expected 3-D sequence input")

    n, timepoints, n_features = X.shape

    if timepoints != 201 or n_features != 8:
        raise RuntimeError(
            f"Unexpected sequence shape: {X.shape}"
        )

    mean = X.mean(axis=1)
    std = X.std(axis=1, ddof=0)
    minimum = X.min(axis=1)
    maximum = X.max(axis=1)
    last = X[:, -1, :]

    # Time is normalized from -1 to +1.
    # This makes the slope a simple measure
    # of whether a variable rises or falls
    # during the 20-ns prefix.
    t = np.linspace(
        -1.0,
        1.0,
        timepoints,
        dtype=np.float64,
    )

    denom = float(
        np.sum(t * t)
    )

    slope = np.einsum(
        "t,ntf->nf",
        t,
        X,
    ) / denom

    return np.concatenate(
        [
            mean,
            std,
            minimum,
            maximum,
            last,
            slope,
        ],
        axis=1,
    )


def ridge_fit(
    X,
    y,
    alpha,
):
    if X.ndim != 2:
        raise RuntimeError("X must be 2-D")

    if y.ndim != 1:
        raise RuntimeError("y must be 1-D")

    if len(X) != len(y):
        raise RuntimeError("X/y length mismatch")

    design = np.column_stack(
        [
            np.ones(
                len(X),
                dtype=np.float64,
            ),
            X,
        ]
    )

    penalty = np.eye(
        design.shape[1],
        dtype=np.float64,
    )

    # Do not penalize intercept.
    penalty[0, 0] = 0.0

    system = (
        design.T @ design
        + alpha * penalty
    )

    rhs = design.T @ y

    beta = np.linalg.solve(
        system,
        rhs,
    )

    intercept = float(beta[0])
    coefficients = beta[1:]

    return intercept, coefficients


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preproc-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    preproc = (
        args.preproc_root
        .expanduser()
        .resolve()
    )

    output = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    work = Path(
        str(output) + ".work"
    )

    if output.exists() or work.exists():
        raise RuntimeError(
            "Output already exists"
        )

    manifest_path = (
        preproc
        / "outer_training_manifest.csv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )

    with manifest_path.open(
        encoding="utf-8",
        newline="",
    ) as handle:

        manifest_rows = list(
            csv.DictReader(handle)
        )

    if len(manifest_rows) != 11:
        raise RuntimeError(
            "Expected 11 outer folds"
        )

    test_system_map = {
        int(row["outer_fold"]):
            row["outer_test_system"]
        for row in manifest_rows
    }

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    models_dir = (
        work
        / "models"
    )

    models_dir.mkdir()

    freeze_rows = []

    for outer in range(1, 12):

        package = (
            preproc
            / "folds"
            / f"fold_{outer:02d}.npz"
        )

        if not package.is_file():
            raise FileNotFoundError(
                package
            )

        with np.load(
            package,
            allow_pickle=False,
        ) as z:

            validation_arrays = {
                "X_valid_scaled",
                "y_valid_scaled",
                "mask_valid",
                "valid_indices",
                "valid_sample_ids",
                "valid_system_ids",
                "valid_replicas",
            }

            forbidden = [
                key
                for key in z.files
                if (
                    "test" in key.lower()
                    or key in validation_arrays
                )
            ]

            if forbidden:
                raise RuntimeError(
                    f"outer={outer:02d}: "
                    f"forbidden arrays {forbidden}"
                )

            X = (
                z["X_train_scaled"]
                .astype(np.float64)
            )

            y = (
                z["y_train_scaled"]
                .astype(np.float64)
            )

            mask = (
                z["mask_train"]
                .astype(bool)
            )

            train_indices = (
                z["train_indices"]
                .astype(np.int64)
            )

            train_system_ids = (
                decode_strings(
                    z["train_system_ids"]
                )
            )

            feature_names = (
                decode_strings(
                    z["feature_names"]
                )
            )

            target_names = (
                decode_strings(
                    z["target_names"]
                )
            )

        if X.shape != (30, 201, 8):
            raise RuntimeError(
                f"outer={outer:02d}: "
                f"bad X shape {X.shape}"
            )

        if y.shape != (30, 5):
            raise RuntimeError(
                f"outer={outer:02d}: "
                "bad y shape"
            )

        if mask.shape != (30, 5):
            raise RuntimeError(
                f"outer={outer:02d}: "
                "bad mask shape"
            )

        if len(set(train_system_ids)) != 10:
            raise RuntimeError(
                f"outer={outer:02d}: "
                "expected 10 train proteins"
            )

        if not np.isfinite(X).all():
            raise RuntimeError(
                "Non-finite features"
            )

        if not np.isfinite(
            y[mask]
        ).all():
            raise RuntimeError(
                "Non-finite valid targets"
            )

        summary = summarize_sequence(
            X
        )

        if summary.shape != (30, 48):
            raise RuntimeError(
                f"Bad summary shape: "
                f"{summary.shape}"
            )

        summary_mean = (
            summary.mean(axis=0)
        )

        summary_std = (
            summary.std(
                axis=0,
                ddof=0,
            )
        )

        zero_variance = (
            summary_std <= 0.0
        )

        summary_std_safe = (
            summary_std.copy()
        )

        summary_std_safe[
            zero_variance
        ] = 1.0

        Z = (
            summary
            - summary_mean
        ) / summary_std_safe

        if not np.isfinite(Z).all():
            raise RuntimeError(
                "Non-finite summary features"
            )

        coefficients = np.empty(
            (5, 48),
            dtype=np.float64,
        )

        intercepts = np.empty(
            5,
            dtype=np.float64,
        )

        predictions = np.empty(
            (30, 5),
            dtype=np.float64,
        )

        train_mse = []

        train_counts = []

        for target in range(5):

            valid = mask[
                :,
                target,
            ]

            if not valid.any():
                raise RuntimeError(
                    f"outer={outer:02d}: "
                    f"target {target} absent"
                )

            intercept, coef = ridge_fit(
                Z[valid],
                y[valid, target],
                ALPHA,
            )

            intercepts[target] = (
                intercept
            )

            coefficients[
                target
            ] = coef

            predictions[
                :,
                target,
            ] = (
                intercept
                + Z @ coef
            )

            error = (
                predictions[
                    valid,
                    target,
                ]
                - y[
                    valid,
                    target,
                ]
            )

            train_mse.append(
                float(
                    np.mean(
                        error * error
                    )
                )
            )

            train_counts.append(
                int(valid.sum())
            )

        if not np.isfinite(
            coefficients
        ).all():
            raise RuntimeError(
                "Non-finite coefficients"
            )

        if not np.isfinite(
            intercepts
        ).all():
            raise RuntimeError(
                "Non-finite intercepts"
            )

        summary_feature_names = []

        for stat in SUMMARY_STATS:
            for name in feature_names:
                summary_feature_names.append(
                    f"{name}__{stat}"
                )

        outer_dir = (
            models_dir
            / f"outer_{outer:02d}"
        )

        outer_dir.mkdir()

        model_path = (
            outer_dir
            / "ridge_summary_model.npz"
        )

        np.savez(
            model_path,

            coefficients=(
                coefficients
            ),

            intercepts=(
                intercepts
            ),

            alpha=np.asarray(
                ALPHA,
                dtype=np.float64,
            ),

            summary_mean=(
                summary_mean
            ),

            summary_std=(
                summary_std_safe
            ),

            summary_zero_variance=(
                zero_variance
            ),

            summary_feature_names=(
                np.asarray(
                    summary_feature_names
                )
            ),

            source_feature_names=(
                np.asarray(
                    feature_names
                )
            ),

            target_names=(
                np.asarray(
                    target_names
                )
            ),
        )

        predictions_path = (
            outer_dir
            / "train_predictions.npz"
        )

        np.savez(
            predictions_path,

            prediction_scaled=(
                predictions
            ),

            target_scaled=(
                y
            ),

            target_mask=(
                mask
            ),

            train_indices=(
                train_indices
            ),
        )

        metadata = {
            "analysis_type":
                "classical_summary_ridge_baseline",

            "outer_fold":
                outer,

            "outer_test_system":
                test_system_map[outer],

            "input":
                "0_to_20ns_prefix_summary",

            "sequence_points":
                201,

            "source_features":
                8,

            "summary_statistics":
                list(
                    SUMMARY_STATS
                ),

            "summary_feature_count":
                48,

            "ridge_alpha":
                ALPHA,

            "alpha_selection":
                "fixed_a_priori_no_tuning",

            "training_samples":
                30,

            "training_proteins":
                10,

            "target_models":
                5,

            "target_valid_counts":
                train_counts,

            "train_mse_scaled_per_target":
                {
                    name: value
                    for name, value
                    in zip(
                        target_names,
                        train_mse,
                    )
                },

            "train_equal_target_mse_scaled":
                float(
                    np.mean(
                        train_mse
                    )
                ),

            "zero_variance_summary_features":
                int(
                    zero_variance.sum()
                ),

            "source_package_sha256":
                sha256_file(
                    package
                ),

            "validation_used":
                False,

            "hyperparameter_tuning":
                False,

            "outer_test_arrays_present":
                False,

            "outer_test_targets_used":
                False,

            "outer_test_evaluated":
                False,
        }

        metadata_path = (
            outer_dir
            / "run_metadata.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        sums = (
            outer_dir
            / "SHA256SUMS.txt"
        )

        with sums.open(
            "w",
            encoding="utf-8",
        ) as handle:

            for path in [
                model_path,
                predictions_path,
                metadata_path,
            ]:

                handle.write(
                    sha256_file(path)
                    + "  "
                    + path.name
                    + "\n"
                )

        freeze_rows.append(
            {
                "outer_fold":
                    outer,

                "outer_test_system":
                    test_system_map[
                        outer
                    ],

                "model_path":
                    str(model_path),

                "model_sha256":
                    sha256_file(
                        model_path
                    ),

                "metadata_sha256":
                    sha256_file(
                        metadata_path
                    ),

                "ridge_alpha":
                    ALPHA,

                "summary_features":
                    48,

                "train_equal_target_mse_scaled":
                    float(
                        np.mean(
                            train_mse
                        )
                    ),

                "outer_test_evaluated":
                    False,
            }
        )

        print(
            f"outer={outer:02d}",
            f"train=30",
            f"features=48",
            f"alpha={ALPHA}",
            f"train_loss={np.mean(train_mse):.6f}",
            "outer_test=NO",
            "PASS",
        )

    freeze_manifest = (
        work
        / "ridge_checkpoint_manifest.csv"
    )

    with freeze_manifest.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        fieldnames = list(
            freeze_rows[0].keys()
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            freeze_rows
        )

    freeze_metadata = {
        "freeze_type":
            "classical_summary_ridge_baseline",

        "n_outer_models":
            11,

        "input_prefix":
            "0_to_20ns",

        "source_sequence_shape":
            [201, 8],

        "summary_statistics":
            list(
                SUMMARY_STATS
            ),

        "summary_feature_count":
            48,

        "ridge_alpha":
            ALPHA,

        "alpha_policy":
            "fixed_a_priori_no_tuning",

        "training_samples_per_outer":
            30,

        "training_proteins_per_outer":
            10,

        "validation_used":
            False,

        "hyperparameter_tuning":
            False,

        "outer_test_targets_used":
            False,

        "outer_test_evaluated":
            False,

        "preproc_manifest_sha256":
            sha256_file(
                manifest_path
            ),

        "checkpoint_manifest_sha256":
            sha256_file(
                freeze_manifest
            ),
    }

    freeze_metadata_path = (
        work
        / "freeze_metadata.json"
    )

    freeze_metadata_path.write_text(
        json.dumps(
            freeze_metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    root_sums = (
        work
        / "SHA256SUMS.txt"
    )

    with root_sums.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in [
            freeze_manifest,
            freeze_metadata_path,
        ]:

            handle.write(
                sha256_file(path)
                + "  "
                + path.name
                + "\n"
            )

    work.rename(
        output
    )

    print()
    print(
        "RIDGE_MODELS_FROZEN: 11 / 11"
    )

    print(
        "SUMMARY_FEATURES: 48"
    )

    print(
        "RIDGE_ALPHA_FIXED: 1.0"
    )

    print(
        "HYPERPARAMETER_TUNING: NO"
    )

    print(
        "OUTER_TEST_EVALUATED: NO"
    )

    print(
        "RIDGE_BASELINE_FREEZE: PASS"
    )


if __name__ == "__main__":
    main()
