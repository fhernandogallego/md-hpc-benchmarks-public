#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGET_NAMES = [
    "rmsd",
    "rg",
    "sasa",
    "helix",
    "strand",
]

TARGET_UNITS = {
    "rmsd": "A",
    "rg": "A",
    "sasa": "A2",
    "helix": "fraction",
    "strand": "fraction",
}

EXPECTED_UNIQUE_VALID_COUNTS = {
    "rmsd": 21,
    "rg": 14,
    "sasa": 20,
    "helix": 26,
    "strand": 26,
}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_key_value_file(path):
    result = {}

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value

    return result


def metric_block(rows, true_key, pred_key):
    true = np.asarray(
        [row[true_key] for row in rows],
        dtype=np.float64,
    )

    pred = np.asarray(
        [row[pred_key] for row in rows],
        dtype=np.float64,
    )

    error = pred - true

    mse = float(
        np.mean(error ** 2)
    )

    mae = float(
        np.mean(np.abs(error))
    )

    rmse = float(
        np.sqrt(mse)
    )

    bias = float(
        np.mean(error)
    )

    r2 = math.nan
    pearson = math.nan

    if true.size >= 2:

        denominator = float(
            np.sum(
                (
                    true
                    - np.mean(true)
                ) ** 2
            )
        )

        if denominator > 0.0:
            r2 = float(
                1.0
                - np.sum(error ** 2)
                / denominator
            )

        if (
            np.std(true) > 0.0
            and np.std(pred) > 0.0
        ):
            pearson = float(
                np.corrcoef(
                    true,
                    pred,
                )[0, 1]
            )

    return {
        "n": int(true.size),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mse": mse,
        "r2": r2,
        "pearson": pearson,
    }


def system_macro_mae(
    rows,
    true_key,
    pred_key,
):
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["system_id"]
        ].append(row)

    per_system = []

    for system_rows in grouped.values():

        true = np.asarray(
            [
                row[true_key]
                for row in system_rows
            ],
            dtype=np.float64,
        )

        pred = np.asarray(
            [
                row[pred_key]
                for row in system_rows
            ],
            dtype=np.float64,
        )

        per_system.append(
            float(
                np.mean(
                    np.abs(
                        pred - true
                    )
                )
            )
        )

    return (
        len(per_system),
        float(np.mean(per_system)),
    )


def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values):
    values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if not values:
        return math.nan

    return statistics.mean(values)


def safe_median(values):
    values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if not values:
        return math.nan

    return statistics.median(values)


def safe_stdev(values):
    values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if len(values) < 2:
        return math.nan

    return statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-root",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        required=True,
    )

    parser.add_argument(
        "--scaler-json",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    run_root = Path(args.run_root)
    data_root = Path(args.data_root)
    scaler_json = Path(
        args.scaler_json
    )
    output_dir = Path(
        args.output_dir
    )

    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    fold_ids = [
        f"fold_{outer:02d}_inner_{inner:02d}"
        for outer in range(1, 12)
        for inner in range(1, 6)
    ]

    scaler_document = json.loads(
        scaler_json.read_text(
            encoding="utf-8"
        )
    )

    scaler_list = scaler_document[
        "scalers"
    ]

    scaler_map = {
        item["nested_fold_id"]: item
        for item in scaler_list
    }

    require(
        len(scaler_list) == 55,
        "Scaler list does not contain 55 entries.",
    )

    require(
        len(scaler_map) == 55,
        "Scaler fold IDs are not unique.",
    )

    require(
        sorted(scaler_map)
        == fold_ids,
        "Scaler fold set differs from expected folds.",
    )

    run_config = read_key_value_file(
        run_root
        / "provenance"
        / "run_config.txt"
    )

    require(
        run_config.get(
            "outer_test_used"
        ) == "false",
        "Run provenance claims outer test was used.",
    )

    require(
        run_config.get(
            "max_epochs"
        ) == "5000",
        "Expected max_epochs=5000.",
    )

    long_rows = []
    fold_target_rows = []
    fold_summary_rows = []

    pair_true_values = defaultdict(list)
    pair_occurrences = defaultdict(int)

    outer_validation_indices = defaultdict(list)

    max_objective_diff = 0.0
    max_per_target_mse_diff = 0.0

    for fold_id in fold_ids:

        outer_number = int(
            fold_id[5:7]
        )

        inner_number = int(
            fold_id[-2:]
        )

        outer_fold = (
            f"fold_{outer_number:02d}"
        )

        inner_fold = (
            f"inner_{inner_number:02d}"
        )

        pred_path = (
            run_root
            / "folds"
            / fold_id
            / "best_predictions.npz"
        )

        source_path = (
            data_root
            / "folds"
            / f"{fold_id}.npz"
        )

        metadata_path = (
            run_root
            / "folds"
            / fold_id
            / "run_metadata.json"
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        scaler = scaler_map[
            fold_id
        ]

        target_scaler = scaler[
            "target_scaler"
        ]

        target_names = list(
            target_scaler[
                "target_names"
            ]
        )

        require(
            target_names
            == TARGET_NAMES,
            f"{fold_id}: scaler target order mismatch.",
        )

        target_mean = np.asarray(
            target_scaler["mean"],
            dtype=np.float64,
        )

        target_std = np.asarray(
            target_scaler["std"],
            dtype=np.float64,
        )

        require(
            target_mean.shape == (5,),
            f"{fold_id}: target mean shape mismatch.",
        )

        require(
            target_std.shape == (5,),
            f"{fold_id}: target std shape mismatch.",
        )

        require(
            bool(
                np.all(
                    np.isfinite(
                        target_mean
                    )
                )
            ),
            f"{fold_id}: non-finite target mean.",
        )

        require(
            bool(
                np.all(
                    np.isfinite(
                        target_std
                    )
                )
            ),
            f"{fold_id}: non-finite target std.",
        )

        require(
            bool(
                np.all(
                    target_std > 0.0
                )
            ),
            f"{fold_id}: non-positive target std.",
        )

        with np.load(
            pred_path,
            allow_pickle=False,
        ) as pred, np.load(
            source_path,
            allow_pickle=False,
        ) as source:

            pred_targets = (
                pred["target_names"]
                .astype(str)
                .tolist()
            )

            source_targets = (
                source["target_names"]
                .astype(str)
                .tolist()
            )

            require(
                pred_targets
                == TARGET_NAMES,
                f"{fold_id}: prediction target order mismatch.",
            )

            require(
                source_targets
                == TARGET_NAMES,
                f"{fold_id}: source target order mismatch.",
            )

            source_train_indices = (
                source["train_indices"]
                .astype(int)
                .tolist()
            )

            source_train_ids = (
                source["train_sample_ids"]
                .astype(str)
                .tolist()
            )

            require(
                source_train_indices
                == [
                    int(value)
                    for value
                    in scaler[
                        "fit_sample_indices"
                    ]
                ],
                f"{fold_id}: scaler fit indices mismatch.",
            )

            require(
                source_train_ids
                == [
                    str(value)
                    for value
                    in scaler[
                        "fit_sample_ids"
                    ]
                ],
                f"{fold_id}: scaler fit sample IDs mismatch.",
            )

            require(
                int(
                    scaler["n_fit_samples"]
                ) == 24,
                f"{fold_id}: scaler fit sample count != 24.",
            )

            y_true = np.asarray(
                pred["y_valid_scaled"],
                dtype=np.float64,
            )

            y_pred = np.asarray(
                pred[
                    "valid_prediction_scaled"
                ],
                dtype=np.float64,
            )

            mask = np.asarray(
                pred["mask_valid"],
                dtype=bool,
            )

            valid_indices = np.asarray(
                pred["valid_indices"],
                dtype=np.int64,
            )

            valid_sample_ids = (
                pred["valid_sample_ids"]
                .astype(str)
            )

            valid_system_ids = (
                source["valid_system_ids"]
                .astype(str)
            )

            valid_replicas = np.asarray(
                source["valid_replicas"],
                dtype=np.int64,
            )

            require(
                y_true.shape == (6, 5),
                f"{fold_id}: y_valid shape mismatch.",
            )

            require(
                y_pred.shape == (6, 5),
                f"{fold_id}: prediction shape mismatch.",
            )

            require(
                mask.shape == (6, 5),
                f"{fold_id}: mask shape mismatch.",
            )

            require(
                np.array_equal(
                    valid_indices,
                    source["valid_indices"],
                ),
                f"{fold_id}: valid indices mismatch.",
            )

            require(
                np.array_equal(
                    valid_sample_ids,
                    source[
                        "valid_sample_ids"
                    ].astype(str),
                ),
                f"{fold_id}: valid IDs mismatch.",
            )

            outer_validation_indices[
                outer_fold
            ].extend(
                valid_indices.tolist()
            )

            per_target_mse = []

            for target_index, target in enumerate(
                TARGET_NAMES
            ):

                current_mask = mask[
                    :,
                    target_index,
                ]

                require(
                    int(
                        current_mask.sum()
                    ) > 0,
                    f"{fold_id}: no valid {target} targets.",
                )

                true_scaled = y_true[
                    current_mask,
                    target_index,
                ]

                pred_scaled = y_pred[
                    current_mask,
                    target_index,
                ]

                error_scaled = (
                    pred_scaled
                    - true_scaled
                )

                current_mse = float(
                    np.mean(
                        error_scaled ** 2
                    )
                )

                per_target_mse.append(
                    current_mse
                )

                true_physical = (
                    true_scaled
                    * target_std[target_index]
                    + target_mean[target_index]
                )

                pred_physical = (
                    pred_scaled
                    * target_std[target_index]
                    + target_mean[target_index]
                )

                error_physical = (
                    pred_physical
                    - true_physical
                )

                target_rows_for_fold = []

                valid_positions = np.flatnonzero(
                    current_mask
                )

                for local_position, array_position in enumerate(
                    valid_positions
                ):

                    row = {
                        "fold_id": fold_id,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "sample_index": int(
                            valid_indices[
                                array_position
                            ]
                        ),
                        "sample_id": str(
                            valid_sample_ids[
                                array_position
                            ]
                        ),
                        "system_id": str(
                            valid_system_ids[
                                array_position
                            ]
                        ),
                        "replica": int(
                            valid_replicas[
                                array_position
                            ]
                        ),
                        "target": target,
                        "unit": TARGET_UNITS[
                            target
                        ],
                        "target_mean": float(
                            target_mean[
                                target_index
                            ]
                        ),
                        "target_std": float(
                            target_std[
                                target_index
                            ]
                        ),
                        "y_true_scaled": float(
                            true_scaled[
                                local_position
                            ]
                        ),
                        "y_pred_scaled": float(
                            pred_scaled[
                                local_position
                            ]
                        ),
                        "error_scaled": float(
                            error_scaled[
                                local_position
                            ]
                        ),
                        "y_true_physical": float(
                            true_physical[
                                local_position
                            ]
                        ),
                        "y_pred_physical": float(
                            pred_physical[
                                local_position
                            ]
                        ),
                        "error_physical": float(
                            error_physical[
                                local_position
                            ]
                        ),
                    }

                    long_rows.append(row)
                    target_rows_for_fold.append(
                        row
                    )

                    pair_key = (
                        row["sample_index"],
                        target,
                    )

                    pair_occurrences[
                        pair_key
                    ] += 1

                    pair_true_values[
                        pair_key
                    ].append(
                        row[
                            "y_true_physical"
                        ]
                    )

                scaled_metrics = metric_block(
                    target_rows_for_fold,
                    "y_true_scaled",
                    "y_pred_scaled",
                )

                physical_metrics = metric_block(
                    target_rows_for_fold,
                    "y_true_physical",
                    "y_pred_physical",
                )

                fold_target_rows.append(
                    {
                        "fold_id": fold_id,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "target": target,
                        "unit": TARGET_UNITS[target],
                        "n_valid": scaled_metrics["n"],
                        "mae_scaled": scaled_metrics["mae"],
                        "rmse_scaled": scaled_metrics["rmse"],
                        "bias_scaled": scaled_metrics["bias"],
                        "mse_scaled": scaled_metrics["mse"],
                        "mae_physical": physical_metrics["mae"],
                        "rmse_physical": physical_metrics["rmse"],
                        "bias_physical": physical_metrics["bias"],
                        "mse_physical": physical_metrics["mse"],
                    }
                )

            recomputed_objective = float(
                np.mean(
                    per_target_mse
                )
            )

            reported_objective = float(
                metadata[
                    "result"
                ][
                    "best_valid_loss"
                ]
            )

            objective_diff = abs(
                recomputed_objective
                - reported_objective
            )

            max_objective_diff = max(
                max_objective_diff,
                objective_diff,
            )

            reported_per_target = np.asarray(
                metadata[
                    "result"
                ][
                    "best_valid_per_target_mse"
                ],
                dtype=np.float64,
            )

            per_target_diff = float(
                np.max(
                    np.abs(
                        np.asarray(
                            per_target_mse
                        )
                        - reported_per_target
                    )
                )
            )

            max_per_target_mse_diff = max(
                max_per_target_mse_diff,
                per_target_diff,
            )

            require(
                objective_diff <= 2e-6,
                f"{fold_id}: recomputed objective mismatch "
                f"{objective_diff}.",
            )

            require(
                per_target_diff <= 2e-6,
                f"{fold_id}: per-target MSE mismatch "
                f"{per_target_diff}.",
            )

            fold_summary_rows.append(
                {
                    "fold_id": fold_id,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "epochs_executed": int(
                        metadata[
                            "result"
                        ][
                            "epochs_executed"
                        ]
                    ),
                    "best_epoch": int(
                        metadata[
                            "result"
                        ][
                            "best_epoch"
                        ]
                    ),
                    "reported_best_valid_loss": reported_objective,
                    "recomputed_equal_target_mse_scaled": recomputed_objective,
                    "objective_abs_diff": objective_diff,
                }
            )

    for outer_fold in [
        f"fold_{outer:02d}"
        for outer in range(1, 12)
    ]:

        indices = (
            outer_validation_indices[
                outer_fold
            ]
        )

        require(
            len(indices) == 30,
            f"{outer_fold}: expected 30 validation sample occurrences.",
        )

        require(
            len(set(indices)) == 30,
            f"{outer_fold}: validation samples not unique across inner folds.",
        )

    require(
        len(long_rows) == 1070,
        f"Expected 1070 valid repeated-context rows, got {len(long_rows)}.",
    )

    require(
        len(pair_occurrences) == 107,
        f"Expected 107 unique sample-target pairs, got {len(pair_occurrences)}.",
    )

    occurrence_values = list(
        pair_occurrences.values()
    )

    require(
        min(occurrence_values) == 10
        and max(occurrence_values) == 10,
        "Every valid sample-target pair must occur exactly 10 times.",
    )

    unique_counts = {
        target: sum(
            1
            for (
                _sample_index,
                current_target,
            ) in pair_occurrences
            if current_target == target
        )
        for target in TARGET_NAMES
    }

    require(
        unique_counts
        == EXPECTED_UNIQUE_VALID_COUNTS,
        "Unique valid target counts differ from canonical counts.",
    )

    max_true_span = {
        target: 0.0
        for target in TARGET_NAMES
    }

    for (
        sample_index,
        target,
    ), values in pair_true_values.items():

        values_array = np.asarray(
            values,
            dtype=np.float64,
        )

        span = float(
            np.max(values_array)
            - np.min(values_array)
        )

        max_true_span[target] = max(
            max_true_span[target],
            span,
        )

        require(
            bool(
                np.allclose(
                    values_array,
                    values_array[0],
                    rtol=1e-10,
                    atol=1e-8,
                )
            ),
            f"Physical target reconstruction inconsistent for "
            f"sample={sample_index} target={target}.",
        )

    outer_target_rows = []

    for outer in range(1, 12):

        outer_fold = (
            f"fold_{outer:02d}"
        )

        for target in TARGET_NAMES:

            rows = [
                row
                for row in long_rows
                if (
                    row["outer_fold"]
                    == outer_fold
                    and row["target"]
                    == target
                )
            ]

            require(
                len(rows) > 0,
                f"{outer_fold}: no rows for target {target}.",
            )

            scaled = metric_block(
                rows,
                "y_true_scaled",
                "y_pred_scaled",
            )

            physical = metric_block(
                rows,
                "y_true_physical",
                "y_pred_physical",
            )

            (
                n_systems,
                macro_mae_scaled,
            ) = system_macro_mae(
                rows,
                "y_true_scaled",
                "y_pred_scaled",
            )

            (
                n_systems_physical,
                macro_mae_physical,
            ) = system_macro_mae(
                rows,
                "y_true_physical",
                "y_pred_physical",
            )

            require(
                n_systems
                == n_systems_physical,
                "System macro count mismatch.",
            )

            outer_target_rows.append(
                {
                    "outer_fold": outer_fold,
                    "target": target,
                    "unit": TARGET_UNITS[target],
                    "n_valid_pairs": scaled["n"],
                    "n_systems_valid": n_systems,
                    "mae_scaled": scaled["mae"],
                    "rmse_scaled": scaled["rmse"],
                    "bias_scaled": scaled["bias"],
                    "r2": scaled["r2"],
                    "pearson": scaled["pearson"],
                    "system_macro_mae_scaled": macro_mae_scaled,
                    "mae_physical": physical["mae"],
                    "rmse_physical": physical["rmse"],
                    "bias_physical": physical["bias"],
                    "system_macro_mae_physical": macro_mae_physical,
                }
            )

    pooled_rows = []

    for target in TARGET_NAMES:

        rows = [
            row
            for row in long_rows
            if row["target"] == target
        ]

        scaled = metric_block(
            rows,
            "y_true_scaled",
            "y_pred_scaled",
        )

        physical = metric_block(
            rows,
            "y_true_physical",
            "y_pred_physical",
        )

        pooled_rows.append(
            {
                "target": target,
                "unit": TARGET_UNITS[target],
                "n_repeated_context_pairs": scaled["n"],
                "n_unique_sample_target_pairs": unique_counts[target],
                "contexts_per_unique_pair": 10,
                "mae_scaled": scaled["mae"],
                "rmse_scaled": scaled["rmse"],
                "bias_scaled": scaled["bias"],
                "r2": scaled["r2"],
                "pearson": scaled["pearson"],
                "mae_physical": physical["mae"],
                "rmse_physical": physical["rmse"],
                "bias_physical": physical["bias"],
            }
        )

    outer_macro_rows = []

    for target in TARGET_NAMES:

        rows = [
            row
            for row in outer_target_rows
            if row["target"] == target
        ]

        outer_macro_rows.append(
            {
                "target": target,
                "unit": TARGET_UNITS[target],
                "n_outer_contexts": len(rows),
                "mean_mae_scaled": safe_mean(
                    [
                        row["mae_scaled"]
                        for row in rows
                    ]
                ),
                "median_mae_scaled": safe_median(
                    [
                        row["mae_scaled"]
                        for row in rows
                    ]
                ),
                "sd_mae_scaled": safe_stdev(
                    [
                        row["mae_scaled"]
                        for row in rows
                    ]
                ),
                "mean_rmse_scaled": safe_mean(
                    [
                        row["rmse_scaled"]
                        for row in rows
                    ]
                ),
                "mean_mae_physical": safe_mean(
                    [
                        row["mae_physical"]
                        for row in rows
                    ]
                ),
                "median_mae_physical": safe_median(
                    [
                        row["mae_physical"]
                        for row in rows
                    ]
                ),
                "sd_mae_physical": safe_stdev(
                    [
                        row["mae_physical"]
                        for row in rows
                    ]
                ),
                "mean_rmse_physical": safe_mean(
                    [
                        row["rmse_physical"]
                        for row in rows
                    ]
                ),
                "mean_system_macro_mae_physical": safe_mean(
                    [
                        row[
                            "system_macro_mae_physical"
                        ]
                        for row in rows
                    ]
                ),
            }
        )

    long_fields = [
        "fold_id",
        "outer_fold",
        "inner_fold",
        "sample_index",
        "sample_id",
        "system_id",
        "replica",
        "target",
        "unit",
        "target_mean",
        "target_std",
        "y_true_scaled",
        "y_pred_scaled",
        "error_scaled",
        "y_true_physical",
        "y_pred_physical",
        "error_physical",
    ]

    fold_target_fields = [
        "fold_id",
        "outer_fold",
        "inner_fold",
        "target",
        "unit",
        "n_valid",
        "mae_scaled",
        "rmse_scaled",
        "bias_scaled",
        "mse_scaled",
        "mae_physical",
        "rmse_physical",
        "bias_physical",
        "mse_physical",
    ]

    fold_summary_fields = [
        "fold_id",
        "outer_fold",
        "inner_fold",
        "epochs_executed",
        "best_epoch",
        "reported_best_valid_loss",
        "recomputed_equal_target_mse_scaled",
        "objective_abs_diff",
    ]

    outer_target_fields = [
        "outer_fold",
        "target",
        "unit",
        "n_valid_pairs",
        "n_systems_valid",
        "mae_scaled",
        "rmse_scaled",
        "bias_scaled",
        "r2",
        "pearson",
        "system_macro_mae_scaled",
        "mae_physical",
        "rmse_physical",
        "bias_physical",
        "system_macro_mae_physical",
    ]

    pooled_fields = [
        "target",
        "unit",
        "n_repeated_context_pairs",
        "n_unique_sample_target_pairs",
        "contexts_per_unique_pair",
        "mae_scaled",
        "rmse_scaled",
        "bias_scaled",
        "r2",
        "pearson",
        "mae_physical",
        "rmse_physical",
        "bias_physical",
    ]

    outer_macro_fields = [
        "target",
        "unit",
        "n_outer_contexts",
        "mean_mae_scaled",
        "median_mae_scaled",
        "sd_mae_scaled",
        "mean_rmse_scaled",
        "mean_mae_physical",
        "median_mae_physical",
        "sd_mae_physical",
        "mean_rmse_physical",
        "mean_system_macro_mae_physical",
    ]

    write_csv(
        output_dir
        / "validation_predictions_long.csv",
        long_rows,
        long_fields,
    )

    write_csv(
        output_dir
        / "fold_target_metrics.csv",
        fold_target_rows,
        fold_target_fields,
    )

    write_csv(
        output_dir
        / "fold_summary.csv",
        fold_summary_rows,
        fold_summary_fields,
    )

    write_csv(
        output_dir
        / "outer_context_target_metrics.csv",
        outer_target_rows,
        outer_target_fields,
    )

    write_csv(
        output_dir
        / "target_pooled_repeated_context_metrics.csv",
        pooled_rows,
        pooled_fields,
    )

    write_csv(
        output_dir
        / "target_outer_macro_summary.csv",
        outer_macro_rows,
        outer_macro_fields,
    )

    summary = {
        "analysis_scope": "inner_validation_only",
        "outer_test_evaluated": False,
        "repeated_context_warning": (
            "Pooled rows are repeated-context inner-validation "
            "predictions and are not independent observations."
        ),
        "run_root": str(run_root),
        "data_root": str(data_root),
        "scaler_json": str(scaler_json),
        "scaler_json_sha256": sha256(
            scaler_json
        ),
        "run_provenance": run_config,
        "n_nested_folds": 55,
        "n_outer_contexts": 11,
        "n_inner_folds_per_outer": 5,
        "n_valid_prediction_rows": len(
            long_rows
        ),
        "n_unique_valid_sample_target_pairs": len(
            pair_occurrences
        ),
        "contexts_per_unique_valid_pair_min": min(
            occurrence_values
        ),
        "contexts_per_unique_valid_pair_max": max(
            occurrence_values
        ),
        "unique_valid_target_counts": unique_counts,
        "max_objective_abs_diff": max_objective_diff,
        "max_per_target_mse_abs_diff": max_per_target_mse_diff,
        "max_true_physical_reconstruction_span": max_true_span,
        "target_outer_macro_summary": outer_macro_rows,
        "target_pooled_repeated_context_metrics": pooled_rows,
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result_files = [
        "validation_predictions_long.csv",
        "fold_target_metrics.csv",
        "fold_summary.csv",
        "outer_context_target_metrics.csv",
        "target_pooled_repeated_context_metrics.csv",
        "target_outer_macro_summary.csv",
        "summary.json",
    ]

    with (
        output_dir
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for name in result_files:

            handle.write(
                f"{sha256(output_dir / name)}  {name}\n"
            )

    print("===== INTEGRITY SUMMARY =====")
    print(
        "nested_folds:",
        len(fold_ids),
    )
    print(
        "validation_rows_repeated_context:",
        len(long_rows),
    )
    print(
        "unique_valid_sample_target_pairs:",
        len(pair_occurrences),
    )
    print(
        "contexts_per_pair_min:",
        min(occurrence_values),
    )
    print(
        "contexts_per_pair_max:",
        max(occurrence_values),
    )
    print(
        "unique_valid_target_counts:",
        unique_counts,
    )
    print(
        "max_objective_abs_diff:",
        max_objective_diff,
    )
    print(
        "max_per_target_mse_abs_diff:",
        max_per_target_mse_diff,
    )
    print(
        "max_true_physical_reconstruction_span:",
        max_true_span,
    )

    print()
    print("===== OUTER-MACRO TARGET SUMMARY =====")

    for row in outer_macro_rows:
        print(
            row["target"],
            f"unit={row['unit']}",
            f"mean_MAE={row['mean_mae_physical']:.9g}",
            f"median_MAE={row['median_mae_physical']:.9g}",
            f"mean_RMSE={row['mean_rmse_physical']:.9g}",
            f"protein_macro_MAE={row['mean_system_macro_mae_physical']:.9g}",
        )

    print()
    print("===== POOLED REPEATED-CONTEXT SUMMARY =====")

    for row in pooled_rows:
        print(
            row["target"],
            f"n={row['n_repeated_context_pairs']}",
            f"unique={row['n_unique_sample_target_pairs']}",
            f"MAE_scaled={row['mae_scaled']:.9g}",
            f"RMSE_scaled={row['rmse_scaled']:.9g}",
            f"MAE_physical={row['mae_physical']:.9g}",
            f"RMSE_physical={row['rmse_physical']:.9g}",
            f"R2={row['r2']:.9g}",
            f"Pearson={row['pearson']:.9g}",
        )

    print()
    print("===== FINAL =====")

    if errors:

        print(
            "BASELINE_INNER_AGGREGATION: FAIL"
        )

        for error in errors:
            print(
                "ERROR:",
                error,
            )

        sys.exit(1)

    print("SCALER_MAPPING_55: PASS")
    print("SCALER_FIT_TRAIN_ONLY_MATCH: PASS")
    print("OUTER_CONTEXT_VALIDATION_COVERAGE: PASS")
    print("VALID_PAIR_MULTIPLICITY_10: PASS")
    print("CANONICAL_TARGET_COUNTS: PASS")
    print("PHYSICAL_TARGET_RECONSTRUCTION: PASS")
    print("FOLD_OBJECTIVE_RECOMPUTATION: PASS")
    print("INNER_VALIDATION_ONLY: PASS")
    print("OUTER_TEST_EVALUATED: NO")
    print("BASELINE_INNER_AGGREGATION: PASS")
    print("PASO 142.3 AGGREGATOR: PASS")


if __name__ == "__main__":
    main()
