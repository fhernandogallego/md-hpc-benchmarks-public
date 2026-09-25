#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


MODELING = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "modeling"
)

sys.path.insert(
    0,
    str(MODELING),
)

from train_gru_tunable import GRUTunable
from train_gru_baseline import GRUBaseline
from train_ridge_summary_outer import summarize_sequence


MODEL_ORDER = [
    "ridge_summary",
    "gru_fixed_h32",
    "gru_improved_gwo",
]


def sha256_file(path):
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
    result = []

    for value in array.tolist():
        if isinstance(value, bytes):
            result.append(
                value.decode("utf-8")
            )
        else:
            result.append(
                str(value)
            )

    return result


def read_csv(path):
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def metric_block(
    true,
    predicted,
):
    true = np.asarray(
        true,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    if true.shape != predicted.shape:
        raise RuntimeError(
            "Metric shape mismatch"
        )

    if true.ndim != 1:
        raise RuntimeError(
            "Metrics expect vectors"
        )

    if len(true) == 0:
        raise RuntimeError(
            "Empty metric input"
        )

    error = predicted - true

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error * error
            )
        )
    )

    r2 = math.nan
    pearson = math.nan

    if len(true) >= 2:

        denominator = float(
            np.sum(
                (
                    true
                    - np.mean(true)
                )
                ** 2
            )
        )

        if denominator > 0.0:
            r2 = float(
                1.0
                - (
                    np.sum(
                        error * error
                    )
                    / denominator
                )
            )

        true_std = float(
            np.std(
                true,
                ddof=0,
            )
        )

        pred_std = float(
            np.std(
                predicted,
                ddof=0,
            )
        )

        if (
            true_std > 0.0
            and pred_std > 0.0
        ):
            pearson = float(
                np.corrcoef(
                    true,
                    predicted,
                )[0, 1]
            )

    return {
        "n": int(len(true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
    }


def bootstrap_metric_ci(
    true,
    predicted,
    n_boot,
    seed,
):
    true = np.asarray(
        true,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    n = len(true)

    if n == 0:
        raise RuntimeError(
            "Empty bootstrap input"
        )

    rng = np.random.default_rng(
        seed
    )

    mae_values = np.empty(
        n_boot,
        dtype=np.float64,
    )

    rmse_values = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for i in range(n_boot):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        error = (
            predicted[indices]
            - true[indices]
        )

        mae_values[i] = np.mean(
            np.abs(error)
        )

        rmse_values[i] = np.sqrt(
            np.mean(
                error * error
            )
        )

    return {
        "mae_ci_low": float(
            np.quantile(
                mae_values,
                0.025,
            )
        ),
        "mae_ci_high": float(
            np.quantile(
                mae_values,
                0.975,
            )
        ),
        "rmse_ci_low": float(
            np.quantile(
                rmse_values,
                0.025,
            )
        ),
        "rmse_ci_high": float(
            np.quantile(
                rmse_values,
                0.975,
            )
        ),
    }


def bootstrap_paired_delta(
    proposed_error,
    comparator_error,
    n_boot,
    seed,
):
    proposed_error = np.asarray(
        proposed_error,
        dtype=np.float64,
    )

    comparator_error = np.asarray(
        comparator_error,
        dtype=np.float64,
    )

    if (
        proposed_error.shape
        != comparator_error.shape
    ):
        raise RuntimeError(
            "Paired comparison mismatch"
        )

    n = len(proposed_error)

    if n == 0:
        raise RuntimeError(
            "Empty paired comparison"
        )

    delta = (
        proposed_error
        - comparator_error
    )

    observed = float(
        np.mean(delta)
    )

    rng = np.random.default_rng(
        seed
    )

    boot = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for i in range(n_boot):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = np.mean(
            delta[indices]
        )

    return {
        "mean_delta":
            observed,

        "ci_low":
            float(
                np.quantile(
                    boot,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    boot,
                    0.975,
                )
            ),

        "proposed_better_fraction":
            float(
                np.mean(
                    proposed_error
                    < comparator_error
                )
            ),
    }


def load_manifest_map(
    path,
    key,
):
    rows = read_csv(path)

    mapping = {
        int(row["outer_fold"]):
            row
        for row in rows
    }

    if len(mapping) != 20:
        raise RuntimeError(
            f"{key}: expected 20 rows"
        )

    return mapping


def verify_sha(
    path,
    expected,
    label,
):
    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} SHA mismatch: "
            f"{actual} != {expected}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-npz",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--lopo-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--preproc-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--gwo-freeze",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--baseline-freeze",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ridge-freeze",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--model-lock",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--protocol",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    raw_path = (
        args.raw_npz
        .expanduser()
        .resolve()
    )

    lopo_path = (
        args.lopo_json
        .expanduser()
        .resolve()
    )

    preproc = (
        args.preproc_root
        .expanduser()
        .resolve()
    )

    gwo_freeze = (
        args.gwo_freeze
        .expanduser()
        .resolve()
    )

    baseline_freeze = (
        args.baseline_freeze
        .expanduser()
        .resolve()
    )

    ridge_freeze = (
        args.ridge_freeze
        .expanduser()
        .resolve()
    )

    model_lock = (
        args.model_lock
        .expanduser()
        .resolve()
    )

    protocol_path = (
        args.protocol
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
            "Evaluation output already exists"
        )

    protocol = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        protocol["status"]
        != "frozen_before_outer_test"
    ):
        raise RuntimeError(
            "Protocol is not frozen"
        )

    identities = protocol[
        "identities"
    ]

    verify_sha(
        raw_path,
        identities[
            "raw_dataset_sha256"
        ],
        "raw dataset",
    )

    verify_sha(
        lopo_path,
        identities[
            "lopo_split_sha256"
        ],
        "LOPO split",
    )

    verify_sha(
        preproc
        / "outer_training_manifest.csv",
        identities[
            "preproc_manifest_sha256"
        ],
        "preproc manifest",
    )

    verify_sha(
        preproc
        / "outer_scalers.json",
        identities[
            "preproc_scalers_sha256"
        ],
        "preproc scalers",
    )

    verify_sha(
        model_lock
        / "model_family_lock.csv",
        identities[
            "model_lock_sha256"
        ],
        "model lock",
    )

    verify_sha(
        model_lock
        / "lock_metadata.json",
        identities[
            "model_lock_metadata_sha256"
        ],
        "model lock metadata",
    )

    verify_sha(
        gwo_freeze
        / "final_checkpoint_manifest.csv",
        identities[
            "gwo_manifest_sha256"
        ],
        "GWO manifest",
    )

    verify_sha(
        gwo_freeze
        / "freeze_metadata.json",
        identities[
            "gwo_metadata_sha256"
        ],
        "GWO metadata",
    )

    verify_sha(
        baseline_freeze
        / "final_checkpoint_manifest.csv",
        identities[
            "fixed_gru_manifest_sha256"
        ],
        "fixed GRU manifest",
    )

    verify_sha(
        baseline_freeze
        / "freeze_metadata.json",
        identities[
            "fixed_gru_metadata_sha256"
        ],
        "fixed GRU metadata",
    )

    verify_sha(
        ridge_freeze
        / "ridge_checkpoint_manifest.csv",
        identities[
            "ridge_manifest_sha256"
        ],
        "Ridge manifest",
    )

    verify_sha(
        ridge_freeze
        / "freeze_metadata.json",
        identities[
            "ridge_metadata_sha256"
        ],
        "Ridge metadata",
    )

    with np.load(
        raw_path,
        allow_pickle=False,
    ) as z:

        X_raw = (
            z["X_raw"]
            .astype(np.float64)
            .copy()
        )

        target_values = (
            z["target_values"]
            .astype(np.float64)
            .copy()
        )

        target_mask = (
            z["target_mask"]
            .astype(bool)
            .copy()
        )

        system_ids = np.asarray(
            decode_strings(
                z["system_ids"]
            )
        )

        replicas = (
            z["replicas"]
            .astype(np.int64)
            .copy()
        )

        sample_ids = np.asarray(
            decode_strings(
                z["sample_ids"]
            )
        )

        target_names = (
            decode_strings(
                z["target_names"]
            )
        )

    if X_raw.shape != (
        60,
        201,
        8,
    ):
        raise RuntimeError(
            "Unexpected raw X shape"
        )

    if target_values.shape != (
        60,
        5,
    ):
        raise RuntimeError(
            "Unexpected target shape"
        )

    if target_mask.shape != (
        60,
        5,
    ):
        raise RuntimeError(
            "Unexpected target mask shape"
        )

    expected_targets = [
        item["name"]
        for item in protocol[
            "targets"
        ]
    ]

    if target_names != expected_targets:
        raise RuntimeError(
            "Target order differs "
            "from frozen protocol"
        )

    units = {
        item["name"]:
            item["unit"]
        for item in protocol[
            "targets"
        ]
    }

    lopo = json.loads(
        lopo_path.read_text(
            encoding="utf-8"
        )
    )

    folds = lopo.get(
        "folds",
        []
    )

    if len(folds) != 20:
        raise RuntimeError(
            "Expected 20 outer folds"
        )

    gwo_map = load_manifest_map(
        gwo_freeze
        / "final_checkpoint_manifest.csv",
        "GWO",
    )

    baseline_map = load_manifest_map(
        baseline_freeze
        / "final_checkpoint_manifest.csv",
        "baseline",
    )

    ridge_map = load_manifest_map(
        ridge_freeze
        / "ridge_checkpoint_manifest.csv",
        "Ridge",
    )

    replica_rows = []

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    for fold in folds:

        outer = int(
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

        if len(train_indices) != 57:
            raise RuntimeError(
                f"outer={outer}: "
                "train != 57"
            )

        if len(test_indices) != 3:
            raise RuntimeError(
                f"outer={outer}: "
                "test != 3"
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
                f"outer={outer}: "
                "train/test overlap"
            )

        observed_systems = set(
            system_ids[
                test_indices
            ].tolist()
        )

        if observed_systems != {
            test_system
        }:
            raise RuntimeError(
                f"outer={outer}: "
                "test protein mismatch"
            )

        package = (
            preproc
            / "folds"
            / f"fold_{outer:02d}.npz"
        )

        with np.load(
            package,
            allow_pickle=False,
        ) as z:

            package_train_indices = (
                z["train_indices"]
                .astype(np.int64)
            )

            feature_mean = (
                z["feature_mean"]
                .astype(np.float64)
            )

            feature_std = (
                z["feature_std"]
                .astype(np.float64)
            )

            target_mean = (
                z["target_mean"]
                .astype(np.float64)
            )

            target_std = (
                z["target_std"]
                .astype(np.float64)
            )

        if not np.array_equal(
            package_train_indices,
            train_indices,
        ):
            raise RuntimeError(
                f"outer={outer}: "
                "train package/split mismatch"
            )

        X_test = (
            X_raw[
                test_indices
            ]
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

        if not np.isfinite(
            X_test
        ).all():
            raise RuntimeError(
                f"outer={outer}: "
                "scaled test X non-finite"
            )

        X_test_f32 = (
            X_test.astype(
                np.float32
            )
        )

        # -------------------------------------------------
        # Improved-GWO GRU
        # -------------------------------------------------

        gwo_row = gwo_map[
            outer
        ]

        gwo_checkpoint = Path(
            gwo_row[
                "checkpoint_path"
            ]
        )

        verify_sha(
            gwo_checkpoint,
            gwo_row[
                "checkpoint_sha256"
            ],
            f"GWO checkpoint outer {outer}",
        )

        gwo_data = torch.load(
            gwo_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if gwo_data[
            "outer_test_used"
        ] is not False:
            raise RuntimeError(
                "GWO checkpoint test flag"
            )

        if not np.allclose(
            np.asarray(
                gwo_data[
                    "feature_mean"
                ],
                dtype=np.float64,
            ),
            feature_mean,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                "GWO feature scaler mismatch"
            )

        if not np.allclose(
            np.asarray(
                gwo_data[
                    "target_mean"
                ],
                dtype=np.float64,
            ),
            target_mean,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                "GWO target scaler mismatch"
            )

        gwo_model = GRUTunable(
            input_size=8,
            hidden_size=int(
                gwo_data[
                    "hidden_size"
                ]
            ),
            num_layers=int(
                gwo_data[
                    "num_layers"
                ]
            ),
            dropout=float(
                gwo_data[
                    "dropout_effective"
                ]
            ),
            output_size=5,
        )

        gwo_model.load_state_dict(
            gwo_data[
                "model_state_dict"
            ]
        )

        gwo_model.eval()

        with torch.no_grad():

            gwo_scaled = (
                gwo_model(
                    torch.from_numpy(
                        X_test_f32
                    )
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )

        gwo_prediction = (
            gwo_scaled
            * target_std[
                None,
                :,
            ]
            + target_mean[
                None,
                :,
            ]
        )

        # -------------------------------------------------
        # Fixed GRU baseline
        # -------------------------------------------------

        base_row = baseline_map[
            outer
        ]

        base_checkpoint = Path(
            base_row[
                "checkpoint_path"
            ]
        )

        verify_sha(
            base_checkpoint,
            base_row[
                "checkpoint_sha256"
            ],
            f"baseline checkpoint outer {outer}",
        )

        base_data = torch.load(
            base_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if base_data[
            "outer_test_used"
        ] is not False:
            raise RuntimeError(
                "Baseline checkpoint test flag"
            )

        if not np.allclose(
            np.asarray(
                base_data[
                    "feature_mean"
                ],
                dtype=np.float64,
            ),
            feature_mean,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                "Baseline feature scaler mismatch"
            )

        base_model = GRUBaseline(
            input_size=8,
            hidden_size=32,
            output_size=5,
        )

        base_model.load_state_dict(
            base_data[
                "model_state_dict"
            ]
        )

        base_model.eval()

        with torch.no_grad():

            base_scaled = (
                base_model(
                    torch.from_numpy(
                        X_test_f32
                    )
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )

        base_prediction = (
            base_scaled
            * target_std[
                None,
                :,
            ]
            + target_mean[
                None,
                :,
            ]
        )

        # -------------------------------------------------
        # Classical Ridge summary baseline
        # -------------------------------------------------

        ridge_row = ridge_map[
            outer
        ]

        ridge_path = Path(
            ridge_row[
                "model_path"
            ]
        )

        verify_sha(
            ridge_path,
            ridge_row[
                "model_sha256"
            ],
            f"Ridge model outer {outer}",
        )

        with np.load(
            ridge_path,
            allow_pickle=False,
        ) as ridge:

            coefficients = (
                ridge[
                    "coefficients"
                ].astype(np.float64)
            )

            intercepts = (
                ridge[
                    "intercepts"
                ].astype(np.float64)
            )

            summary_mean = (
                ridge[
                    "summary_mean"
                ].astype(np.float64)
            )

            summary_std = (
                ridge[
                    "summary_std"
                ].astype(np.float64)
            )

        summary = summarize_sequence(
            X_test
        )

        summary_scaled = (
            summary
            - summary_mean[
                None,
                :,
            ]
        ) / summary_std[
            None,
            :,
        ]

        ridge_scaled = (
            intercepts[
                None,
                :,
            ]
            + (
                summary_scaled
                @ coefficients.T
            )
        )

        ridge_prediction = (
            ridge_scaled
            * target_std[
                None,
                :,
            ]
            + target_mean[
                None,
                :,
            ]
        )

        predictions = {
            "ridge_summary":
                ridge_prediction,

            "gru_fixed_h32":
                base_prediction,

            "gru_improved_gwo":
                gwo_prediction,
        }

        for local_index, sample_index in enumerate(
            test_indices
        ):

            for target_index, target_name in enumerate(
                target_names
            ):

                valid = bool(
                    target_mask[
                        sample_index,
                        target_index,
                    ]
                )

                true_value = (
                    float(
                        target_values[
                            sample_index,
                            target_index,
                        ]
                    )
                    if valid
                    else math.nan
                )

                row = {
                    "outer_fold":
                        outer,

                    "system_id":
                        str(
                            system_ids[
                                sample_index
                            ]
                        ),

                    "replica":
                        int(
                            replicas[
                                sample_index
                            ]
                        ),

                    "sample_id":
                        str(
                            sample_ids[
                                sample_index
                            ]
                        ),

                    "target":
                        target_name,

                    "unit":
                        units[
                            target_name
                        ],

                    "target_valid":
                        valid,

                    "true_value":
                        true_value,
                }

                for model_name in MODEL_ORDER:

                    row[
                        f"pred_{model_name}"
                    ] = float(
                        predictions[
                            model_name
                        ][
                            local_index,
                            target_index,
                        ]
                    )

                replica_rows.append(
                    row
                )

    # -----------------------------------------------------
    # Protein-level primary aggregation
    # -----------------------------------------------------

    protein_rows = []

    for outer in range(1, 21):

        outer_rows = [
            row
            for row in replica_rows
            if row[
                "outer_fold"
            ] == outer
        ]

        systems = {
            row["system_id"]
            for row in outer_rows
        }

        if len(systems) != 1:
            raise RuntimeError(
                "Outer fold has !=1 protein"
            )

        system_id = next(
            iter(systems)
        )

        for target_name in target_names:

            valid_rows = [
                row
                for row in outer_rows
                if (
                    row["target"]
                    == target_name
                    and row[
                        "target_valid"
                    ]
                )
            ]

            if not valid_rows:
                continue

            protein_row = {
                "outer_fold":
                    outer,

                "system_id":
                    system_id,

                "target":
                    target_name,

                "unit":
                    units[
                        target_name
                    ],

                "n_valid_replicas":
                    len(
                        valid_rows
                    ),

                "true_mean":
                    float(
                        np.mean(
                            [
                                row[
                                    "true_value"
                                ]
                                for row
                                in valid_rows
                            ]
                        )
                    ),
            }

            for model_name in MODEL_ORDER:

                predicted_mean = float(
                    np.mean(
                        [
                            row[
                                f"pred_{model_name}"
                            ]
                            for row
                            in valid_rows
                        ]
                    )
                )

                protein_row[
                    f"pred_mean_{model_name}"
                ] = predicted_mean

                protein_row[
                    f"abs_error_{model_name}"
                ] = abs(
                    predicted_mean
                    - protein_row[
                        "true_mean"
                    ]
                )

            protein_rows.append(
                protein_row
            )

    # -----------------------------------------------------
    # Primary metrics
    # -----------------------------------------------------

    n_boot = int(
        protocol[
            "uncertainty"
        ][
            "bootstrap_replicates"
        ]
    )

    bootstrap_seed = int(
        protocol[
            "uncertainty"
        ][
            "seed"
        ]
    )

    primary_rows = []

    seed_counter = 0

    for target_name in target_names:

        target_rows = [
            row
            for row in protein_rows
            if row[
                "target"
            ] == target_name
        ]

        true = np.asarray(
            [
                row["true_mean"]
                for row in target_rows
            ],
            dtype=np.float64,
        )

        for model_name in MODEL_ORDER:

            predicted = np.asarray(
                [
                    row[
                        f"pred_mean_"
                        f"{model_name}"
                    ]
                    for row
                    in target_rows
                ],
                dtype=np.float64,
            )

            metrics = metric_block(
                true,
                predicted,
            )

            ci = bootstrap_metric_ci(
                true,
                predicted,
                n_boot,
                bootstrap_seed
                + seed_counter,
            )

            seed_counter += 1

            primary_rows.append(
                {
                    "model":
                        model_name,

                    "target":
                        target_name,

                    "unit":
                        units[
                            target_name
                        ],

                    "n_proteins":
                        metrics["n"],

                    "mae":
                        metrics["mae"],

                    "mae_ci_low":
                        ci[
                            "mae_ci_low"
                        ],

                    "mae_ci_high":
                        ci[
                            "mae_ci_high"
                        ],

                    "rmse":
                        metrics[
                            "rmse"
                        ],

                    "rmse_ci_low":
                        ci[
                            "rmse_ci_low"
                        ],

                    "rmse_ci_high":
                        ci[
                            "rmse_ci_high"
                        ],

                    "r2":
                        metrics["r2"],

                    "pearson":
                        metrics[
                            "pearson"
                        ],
                }
            )

    # -----------------------------------------------------
    # Secondary pooled-replica metrics
    # -----------------------------------------------------

    pooled_rows = []

    for target_name in target_names:

        valid_rows = [
            row
            for row in replica_rows
            if (
                row["target"]
                == target_name
                and row[
                    "target_valid"
                ]
            )
        ]

        true = np.asarray(
            [
                row["true_value"]
                for row in valid_rows
            ],
            dtype=np.float64,
        )

        for model_name in MODEL_ORDER:

            predicted = np.asarray(
                [
                    row[
                        f"pred_{model_name}"
                    ]
                    for row
                    in valid_rows
                ],
                dtype=np.float64,
            )

            metrics = metric_block(
                true,
                predicted,
            )

            pooled_rows.append(
                {
                    "model":
                        model_name,

                    "target":
                        target_name,

                    "unit":
                        units[
                            target_name
                        ],

                    "n_replicas":
                        metrics["n"],

                    "mae":
                        metrics["mae"],

                    "rmse":
                        metrics["rmse"],

                    "r2":
                        metrics["r2"],

                    "pearson":
                        metrics[
                            "pearson"
                        ],
                }
            )

    # -----------------------------------------------------
    # Paired proposed-vs-baseline comparisons
    # -----------------------------------------------------

    comparison_rows = []

    comparisons = [
        (
            "gru_improved_gwo",
            "gru_fixed_h32",
        ),
        (
            "gru_improved_gwo",
            "ridge_summary",
        ),
    ]

    for target_index, target_name in enumerate(
        target_names
    ):

        target_rows = [
            row
            for row in protein_rows
            if row[
                "target"
            ] == target_name
        ]

        for comparison_index, (
            proposed,
            comparator,
        ) in enumerate(
            comparisons
        ):

            proposed_error = np.asarray(
                [
                    row[
                        f"abs_error_"
                        f"{proposed}"
                    ]
                    for row
                    in target_rows
                ],
                dtype=np.float64,
            )

            comparator_error = np.asarray(
                [
                    row[
                        f"abs_error_"
                        f"{comparator}"
                    ]
                    for row
                    in target_rows
                ],
                dtype=np.float64,
            )

            result = bootstrap_paired_delta(
                proposed_error,
                comparator_error,
                n_boot,
                bootstrap_seed
                + 1000
                + target_index * 10
                + comparison_index,
            )

            comparison_rows.append(
                {
                    "target":
                        target_name,

                    "unit":
                        units[
                            target_name
                        ],

                    "proposed":
                        proposed,

                    "comparator":
                        comparator,

                    "n_proteins":
                        len(
                            target_rows
                        ),

                    "mean_abs_error_delta":
                        result[
                            "mean_delta"
                        ],

                    "ci_low":
                        result[
                            "ci_low"
                        ],

                    "ci_high":
                        result[
                            "ci_high"
                        ],

                    "negative_favors_proposed":
                        True,

                    "proposed_better_fraction":
                        result[
                            "proposed_better_fraction"
                        ],
                }
            )

    # -----------------------------------------------------
    # Write once, after all evaluation succeeded
    # -----------------------------------------------------

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    def write_rows(
        path,
        rows,
    ):
        if not rows:
            raise RuntimeError(
                f"No rows for {path}"
            )

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(rows)

    replica_path = (
        work
        / "replica_predictions.csv"
    )

    protein_path = (
        work
        / "protein_predictions.csv"
    )

    primary_path = (
        work
        / "primary_protein_metrics.csv"
    )

    pooled_path = (
        work
        / "secondary_pooled_replica_metrics.csv"
    )

    comparison_path = (
        work
        / "paired_model_comparisons.csv"
    )

    write_rows(
        replica_path,
        replica_rows,
    )

    write_rows(
        protein_path,
        protein_rows,
    )

    write_rows(
        primary_path,
        primary_rows,
    )

    write_rows(
        pooled_path,
        pooled_rows,
    )

    write_rows(
        comparison_path,
        comparison_rows,
    )

    metadata = {
        "evaluation":
            "atlas20_outer_test_v1",

        "protocol_sha256":
            sha256_file(
                protocol_path
            ),

        "outer_folds_evaluated":
            20,

        "test_replicas_total":
            60,

        "model_families":
            MODEL_ORDER,

        "primary_unit":
            "held_out_protein",

        "primary_metrics": [
            "MAE",
            "RMSE",
            "R2",
            "Pearson",
        ],

        "bootstrap_replicates":
            n_boot,

        "bootstrap_seed":
            bootstrap_seed,

        "model_selection_after_test":
            False,

        "hyperparameter_tuning_after_test":
            False,

        "outer_test_evaluated":
            True,
    }

    metadata_path = (
        work
        / "evaluation_metadata.json"
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

    files = [
        replica_path,
        protein_path,
        primary_path,
        pooled_path,
        comparison_path,
        metadata_path,
    ]

    sums_path = (
        work
        / "SHA256SUMS.txt"
    )

    with sums_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in files:

            handle.write(
                sha256_file(path)
                + "  "
                + path.name
                + "\n"
            )

    work.rename(
        output
    )

    print(
        "OUTER_FOLDS_EVALUATED: 20 / 20"
    )

    print(
        "TEST_REPLICAS_EVALUATED: 60 / 60"
    )

    print(
        "MODEL_FAMILIES_EVALUATED: 3 / 3"
    )

    print(
        "PRIMARY_UNIT: HELD_OUT_PROTEIN"
    )

    print(
        "PRIMARY_METRICS: MAE RMSE R2 PEARSON"
    )

    print(
        "BOOTSTRAP_CI: 95%"
    )

    print(
        "OUTER_TEST_EVALUATED: YES"
    )

    print(
        "FINAL_OUTER_TEST_EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()
