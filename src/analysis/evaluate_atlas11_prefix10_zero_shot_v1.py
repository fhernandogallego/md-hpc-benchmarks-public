#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_gru_tunable import GRUTunable


TARGET_UNITS = {
    "rmsd": "A",
    "rg": "A",
    "sasa": "A2",
    "helix": "fraction",
    "strand": "fraction",
}

PREFIXES = {
    "10ns": 101,
    "20ns": 201,
}

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260818


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--preproc", required=True, type=Path)
    parser.add_argument("--gwo-freeze", required=True, type=Path)
    parser.add_argument("--frozen-outer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def decode_strings(values):
    result = []

    for value in values:
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
        return list(csv.DictReader(handle))


def write_csv(path, rows):
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


def metric_block(true, predicted):
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
            "Metric arrays must be 1D"
        )

    if len(true) < 2:
        raise RuntimeError(
            "Need >=2 observations"
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
                error ** 2
            )
        )
    )

    denominator = float(
        np.sum(
            (
                true
                - np.mean(true)
            ) ** 2
        )
    )

    if denominator == 0.0:
        r2 = float("nan")
    else:
        r2 = float(
            1.0
            - np.sum(error ** 2)
            / denominator
        )

    if (
        np.std(true) == 0.0
        or np.std(predicted) == 0.0
    ):
        pearson = float("nan")
    else:
        pearson = float(
            np.corrcoef(
                true,
                predicted,
            )[0, 1]
        )

    return {
        "n": len(true),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
    }


def bootstrap_mae_ci(
    true,
    predicted,
    *,
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

    rng = np.random.default_rng(
        seed
    )

    values = np.empty(
        BOOTSTRAP_REPLICATES,
        dtype=np.float64,
    )

    for i in range(
        BOOTSTRAP_REPLICATES
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        values[i] = np.mean(
            np.abs(
                predicted[idx]
                - true[idx]
            )
        )

    low, high = np.quantile(
        values,
        [0.025, 0.975],
    )

    return (
        float(low),
        float(high),
    )


def bootstrap_paired_error_delta(
    error10,
    error20,
    *,
    seed,
):
    error10 = np.asarray(
        error10,
        dtype=np.float64,
    )

    error20 = np.asarray(
        error20,
        dtype=np.float64,
    )

    if error10.shape != error20.shape:
        raise RuntimeError(
            "Paired error shape mismatch"
        )

    delta = (
        error10
        - error20
    )

    n = len(delta)

    rng = np.random.default_rng(
        seed
    )

    boot = np.empty(
        BOOTSTRAP_REPLICATES,
        dtype=np.float64,
    )

    for i in range(
        BOOTSTRAP_REPLICATES
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = np.mean(
            delta[idx]
        )

    low, high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return {
        "n": n,

        "mean_error_delta_10_minus_20":
            float(
                np.mean(delta)
            ),

        "median_error_delta_10_minus_20":
            float(
                np.median(delta)
            ),

        "ci95_low":
            float(low),

        "ci95_high":
            float(high),

        "ten_ns_better_fraction":
            float(
                np.mean(
                    delta < 0.0
                )
            ),

        "equal_fraction":
            float(
                np.mean(
                    delta == 0.0
                )
            ),

        "positive_delta_means_10ns_worse":
            True,
    }


def main():
    args = parse_args()

    raw_path = (
        args.raw
        .expanduser()
        .resolve()
    )

    preproc = (
        args.preproc
        .expanduser()
        .resolve()
    )

    gwo_freeze = (
        args.gwo_freeze
        .expanduser()
        .resolve()
    )

    frozen_outer = (
        args.frozen_outer
        .expanduser()
        .resolve()
    )

    output = (
        args.output
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
        gwo_freeze
        / "final_checkpoint_manifest.csv"
    )

    frozen_replica_path = (
        frozen_outer
        / "replica_predictions.csv"
    )

    # --------------------------------------------------
    # Frozen raw dataset
    # --------------------------------------------------

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

        sample_ids = np.asarray(
            decode_strings(
                z["sample_ids"]
            )
        )

        replicas = (
            z["replicas"]
            .astype(np.int64)
            .copy()
        )

        target_names = (
            decode_strings(
                z["target_names"]
            )
        )

    if X_raw.shape != (
        33,
        201,
        8,
    ):
        raise RuntimeError(
            f"Unexpected X shape: "
            f"{X_raw.shape}"
        )

    if target_values.shape != (
        33,
        5,
    ):
        raise RuntimeError(
            "Unexpected target shape"
        )

    if target_mask.shape != (
        33,
        5,
    ):
        raise RuntimeError(
            "Unexpected mask shape"
        )

    if len(set(system_ids)) != 11:
        raise RuntimeError(
            "Expected 11 proteins"
        )

    # --------------------------------------------------
    # Frozen final GWO manifest
    # --------------------------------------------------

    manifest_rows = read_csv(
        manifest_path
    )

    if len(manifest_rows) != 11:
        raise RuntimeError(
            "Expected 11 GWO checkpoints"
        )

    fold_map = {}

    for row in manifest_rows:

        outer = int(
            row["outer_fold"]
        )

        if outer in fold_map:
            raise RuntimeError(
                "Duplicate outer fold"
            )

        checkpoint = Path(
            row["checkpoint_path"]
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                checkpoint
            )

        actual = sha256_file(
            checkpoint
        )

        expected = (
            row["checkpoint_sha256"]
            .strip()
        )

        if actual != expected:
            raise RuntimeError(
                f"Checkpoint hash mismatch "
                f"outer={outer}"
            )

        fold_map[outer] = row

    if set(fold_map) != set(
        range(1, 12)
    ):
        raise RuntimeError(
            "Outer folds != 1..11"
        )

    # --------------------------------------------------
    # Frozen official 20-ns predictions
    # --------------------------------------------------

    frozen_rows = read_csv(
        frozen_replica_path
    )

    frozen_lookup = {}

    for row in frozen_rows:

        key = (
            int(row["outer_fold"]),
            row["sample_id"],
            row["target"],
        )

        if key in frozen_lookup:
            raise RuntimeError(
                "Duplicate frozen prediction key"
            )

        frozen_lookup[key] = float(
            row[
                "pred_gru_improved_gwo"
            ]
        )

    if len(frozen_lookup) != 165:
        raise RuntimeError(
            f"Expected 165 frozen "
            f"GWO prediction cells, "
            f"got {len(frozen_lookup)}"
        )

    # --------------------------------------------------
    # Predict 10 ns and independently reproduce 20 ns
    # --------------------------------------------------

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(
            1
        )
    except RuntimeError:
        pass

    prediction_rows = []

    reproduction_diffs = []

    for outer in range(1, 12):

        manifest = fold_map[outer]

        test_system = str(
            manifest[
                "outer_test_system"
            ]
        )

        test_indices = np.flatnonzero(
            system_ids
            == test_system
        )

        if len(test_indices) != 3:
            raise RuntimeError(
                f"outer={outer}: "
                f"test protein has "
                f"{len(test_indices)} replicas"
            )

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

            time_ns = (
                z["time_ns"]
                .astype(np.float64)
            )

        if time_ns.shape != (201,):
            raise RuntimeError(
                "Unexpected time grid"
            )

        if (
            abs(float(time_ns[0])) > 1e-12
            or abs(
                float(time_ns[100])
                - 10.0
            ) > 1e-8
            or abs(
                float(time_ns[-1])
                - 20.0
            ) > 1e-8
        ):
            raise RuntimeError(
                "Unexpected 0/10/20 ns grid"
            )

        checkpoint = Path(
            manifest[
                "checkpoint_path"
            ]
        )

        data = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if data[
            "outer_test_used"
        ] is not False:
            raise RuntimeError(
                "Checkpoint outer-test flag "
                "is not False"
            )

        # Exact scaler identity guards.

        for key, expected in [
            (
                "feature_mean",
                feature_mean,
            ),
            (
                "feature_std",
                feature_std,
            ),
            (
                "target_mean",
                target_mean,
            ),
            (
                "target_std",
                target_std,
            ),
        ]:

            observed = np.asarray(
                data[key],
                dtype=np.float64,
            )

            if not np.allclose(
                observed,
                expected,
                rtol=0.0,
                atol=0.0,
            ):
                raise RuntimeError(
                    f"Scaler mismatch: "
                    f"{key}, outer={outer}"
                )

        model = GRUTunable(
            input_size=8,

            hidden_size=int(
                data[
                    "hidden_size"
                ]
            ),

            num_layers=int(
                data[
                    "num_layers"
                ]
            ),

            dropout=float(
                data[
                    "dropout_effective"
                ]
            ),

            output_size=5,
        )

        model.load_state_dict(
            data[
                "model_state_dict"
            ]
        )

        model.eval()

        fold_predictions = {}

        for prefix_name, frames in (
            ("10ns", 101),
            ("20ns", 201),
        ):

            X = (
                X_raw[
                    test_indices,
                    :frames,
                    :,
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

            if X.shape != (
                3,
                frames,
                8,
            ):
                raise RuntimeError(
                    "Unexpected truncated shape"
                )

            if not np.isfinite(
                X
            ).all():
                raise RuntimeError(
                    "Non-finite scaled input"
                )

            with torch.no_grad():

                scaled = (
                    model(
                        torch.from_numpy(
                            X.astype(
                                np.float32
                            )
                        )
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

            prediction = (
                scaled
                * target_std[
                    None,
                    :,
                ]
                + target_mean[
                    None,
                    :,
                ]
            )

            fold_predictions[
                prefix_name
            ] = prediction

        # Reproduce frozen 20-ns prediction BEFORE
        # accepting any 10-ns result.

        pred20 = fold_predictions[
            "20ns"
        ]

        pred10 = fold_predictions[
            "10ns"
        ]

        for local_index, raw_index in enumerate(
            test_indices
        ):

            sample_id = str(
                sample_ids[
                    raw_index
                ]
            )

            for target_index, target in enumerate(
                target_names
            ):

                key = (
                    outer,
                    sample_id,
                    target,
                )

                if key not in frozen_lookup:
                    raise RuntimeError(
                        f"Missing frozen key: "
                        f"{key}"
                    )

                frozen_value = (
                    frozen_lookup[key]
                )

                reproduced = float(
                    pred20[
                        local_index,
                        target_index,
                    ]
                )

                diff = abs(
                    reproduced
                    - frozen_value
                )

                reproduction_diffs.append(
                    diff
                )

                prediction_rows.append(
                    {
                        "outer_fold":
                            outer,

                        "system_id":
                            str(
                                system_ids[
                                    raw_index
                                ]
                            ),

                        "sample_id":
                            sample_id,

                        "replica":
                            int(
                                replicas[
                                    raw_index
                                ]
                            ),

                        "target":
                            target,

                        "unit":
                            TARGET_UNITS.get(
                                target,
                                "",
                            ),

                        "target_valid":
                            bool(
                                target_mask[
                                    raw_index,
                                    target_index,
                                ]
                            ),

                        "true_value":
                            float(
                                target_values[
                                    raw_index,
                                    target_index,
                                ]
                            ),

                        "pred_10ns":
                            float(
                                pred10[
                                    local_index,
                                    target_index,
                                ]
                            ),

                        "pred_20ns_reproduced":
                            reproduced,

                        "pred_20ns_frozen":
                            frozen_value,

                        "abs_reproduction_diff":
                            diff,
                    }
                )

    max_reproduction_diff = float(
        max(
            reproduction_diffs
        )
    )

    if max_reproduction_diff > 1e-10:
        raise RuntimeError(
            "20-ns reproduction failed: "
            f"max_abs_diff="
            f"{max_reproduction_diff:.17g}"
        )

    # --------------------------------------------------
    # Protein-level aggregation
    # --------------------------------------------------

    protein_rows = []

    for outer in range(1, 12):

        fold_rows = [
            row
            for row in prediction_rows
            if row["outer_fold"]
            == outer
        ]

        systems = {
            row["system_id"]
            for row in fold_rows
        }

        if len(systems) != 1:
            raise RuntimeError(
                "Fold != one protein"
            )

        system_id = next(
            iter(systems)
        )

        for target in target_names:

            rows = [
                row
                for row in fold_rows
                if (
                    row["target"]
                    == target
                    and row[
                        "target_valid"
                    ]
                )
            ]

            if not rows:
                continue

            true_mean = float(
                np.mean(
                    [
                        row[
                            "true_value"
                        ]
                        for row in rows
                    ]
                )
            )

            pred10_mean = float(
                np.mean(
                    [
                        row[
                            "pred_10ns"
                        ]
                        for row in rows
                    ]
                )
            )

            pred20_mean = float(
                np.mean(
                    [
                        row[
                            "pred_20ns_reproduced"
                        ]
                        for row in rows
                    ]
                )
            )

            error10 = abs(
                pred10_mean
                - true_mean
            )

            error20 = abs(
                pred20_mean
                - true_mean
            )

            protein_rows.append(
                {
                    "outer_fold":
                        outer,

                    "system_id":
                        system_id,

                    "target":
                        target,

                    "unit":
                        TARGET_UNITS.get(
                            target,
                            "",
                        ),

                    "n_valid_replicas":
                        len(rows),

                    "true_mean":
                        true_mean,

                    "pred_mean_10ns":
                        pred10_mean,

                    "pred_mean_20ns":
                        pred20_mean,

                    "abs_error_10ns":
                        error10,

                    "abs_error_20ns":
                        error20,

                    "error_delta_10_minus_20":
                        error10
                        - error20,
                }
            )

    # --------------------------------------------------
    # Per-target metrics
    # --------------------------------------------------

    metrics_rows = []
    paired_rows = []

    for target_index, target in enumerate(
        target_names
    ):

        rows = [
            row
            for row in protein_rows
            if row["target"]
            == target
        ]

        true = np.asarray(
            [
                row["true_mean"]
                for row in rows
            ],
            dtype=np.float64,
        )

        pred10 = np.asarray(
            [
                row["pred_mean_10ns"]
                for row in rows
            ],
            dtype=np.float64,
        )

        pred20 = np.asarray(
            [
                row["pred_mean_20ns"]
                for row in rows
            ],
            dtype=np.float64,
        )

        error10 = np.abs(
            pred10 - true
        )

        error20 = np.abs(
            pred20 - true
        )

        metric10 = metric_block(
            true,
            pred10,
        )

        metric20 = metric_block(
            true,
            pred20,
        )

        for prefix, predicted, metrics in [
            (
                "10ns",
                pred10,
                metric10,
            ),
            (
                "20ns",
                pred20,
                metric20,
            ),
        ]:

            ci_low, ci_high = (
                bootstrap_mae_ci(
                    true,
                    predicted,
                    seed=(
                        BOOTSTRAP_SEED
                        + target_index * 10
                        + (
                            0
                            if prefix
                            == "10ns"
                            else 1
                        )
                    ),
                )
            )

            metrics_rows.append(
                {
                    "prefix":
                        prefix,

                    "target":
                        target,

                    "unit":
                        TARGET_UNITS.get(
                            target,
                            "",
                        ),

                    "n_proteins":
                        metrics["n"],

                    "mae":
                        metrics["mae"],

                    "mae_ci_low":
                        ci_low,

                    "mae_ci_high":
                        ci_high,

                    "rmse":
                        metrics["rmse"],

                    "r2":
                        metrics["r2"],

                    "pearson":
                        metrics["pearson"],
                }
            )

        paired = (
            bootstrap_paired_error_delta(
                error10,
                error20,
                seed=(
                    BOOTSTRAP_SEED
                    + 1000
                    + target_index
                ),
            )
        )

        mae10 = metric10["mae"]
        mae20 = metric20["mae"]

        paired_rows.append(
            {
                "target":
                    target,

                "unit":
                    TARGET_UNITS.get(
                        target,
                        "",
                    ),

                "n_proteins":
                    len(rows),

                "mae_10ns":
                    mae10,

                "mae_20ns":
                    mae20,

                "mae_change_10_minus_20":
                    mae10
                    - mae20,

                "mae_percent_change_10_vs_20":
                    (
                        100.0
                        * (
                            mae10
                            - mae20
                        )
                        / mae20
                    ),

                **paired,
            }
        )

    # --------------------------------------------------
    # Trajectory-length implications
    # --------------------------------------------------

    trajectory_rows = [
        {
            "prefix_ns": 10,
            "reference_ns": 100,
            "trajectory_reduction_percent": 90.0,
            "full_to_prefix_length_ratio": 10.0,
            "n_trajectories": 33,
            "aggregate_prefix_us": 0.33,
            "aggregate_reference_us": 3.30,
            "potentially_avoidable_us": 2.97,
            "measured_wallclock_speedup": False,
        },
        {
            "prefix_ns": 20,
            "reference_ns": 100,
            "trajectory_reduction_percent": 80.0,
            "full_to_prefix_length_ratio": 5.0,
            "n_trajectories": 33,
            "aggregate_prefix_us": 0.66,
            "aggregate_reference_us": 3.30,
            "potentially_avoidable_us": 2.64,
            "measured_wallclock_speedup": False,
        },
    ]

    # --------------------------------------------------
    # Freeze output
    # --------------------------------------------------

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        work
        / "replica_predictions_10vs20.csv",
        prediction_rows,
    )

    write_csv(
        work
        / "protein_predictions_10vs20.csv",
        protein_rows,
    )

    write_csv(
        work
        / "protein_metrics_10vs20.csv",
        metrics_rows,
    )

    write_csv(
        work
        / "paired_error_10vs20.csv",
        paired_rows,
    )

    write_csv(
        work
        / "trajectory_length_10vs20.csv",
        trajectory_rows,
    )

    metadata = {
        "analysis":
            "atlas11_10ns_zero_shot_prefix_sensitivity",

        "status":
            "post_hoc_exploratory",

        "primary_frozen_analysis":
            "20ns outer-test remains primary",

        "model_family":
            "gru_improved_gwo",

        "model_retraining":
            False,

        "hyperparameter_selection":
            False,

        "epoch_selection":
            False,

        "scaler_refit":
            False,

        "model_replacement":
            False,

        "outer_test_reopened_for_selection":
            False,

        "prefixes_frames": {
            "10ns": 101,
            "20ns": 201,
        },

        "time_grid":
            "0.1 ns",

        "20ns_reproduction_required":
            True,

        "20ns_max_abs_prediction_difference":
            max_reproduction_diff,

        "bootstrap_replicates":
            BOOTSTRAP_REPLICATES,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "paired_definition":
            (
                "absolute_error_10ns "
                "minus absolute_error_20ns"
            ),

        "paired_primary_unit":
            "held-out protein",

        "positive_paired_delta":
            "10ns worse than 20ns",

        "wallclock_speedup_measured":
            False,

        "scientific_warning":
            (
                "10ns results are zero-shot "
                "post-hoc prefix truncation of "
                "models trained and selected "
                "under the frozen 20ns protocol. "
                "They are not independently "
                "optimized 10ns models."
            ),

        "source_sha256": {
            "raw_dataset":
                sha256_file(raw_path),

            "gwo_manifest":
                sha256_file(
                    manifest_path
                ),

            "frozen_replica_predictions":
                sha256_file(
                    frozen_replica_path
                ),
        },
    }

    metadata_path = (
        work
        / "metadata.json"
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

    artifacts = [
        work
        / "replica_predictions_10vs20.csv",

        work
        / "protein_predictions_10vs20.csv",

        work
        / "protein_metrics_10vs20.csv",

        work
        / "paired_error_10vs20.csv",

        work
        / "trajectory_length_10vs20.csv",

        metadata_path,
    ]

    with (
        work
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in artifacts:

            handle.write(
                sha256_file(path)
                + "  "
                + path.name
                + "\n"
            )

    work.rename(
        output
    )

    # --------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------

    print(
        "20NS_REPRODUCTION_MAX_ABS_DIFF:",
        f"{max_reproduction_diff:.17g}",
    )

    print(
        "20NS_REPRODUCTION:",
        "PASS",
    )

    print()
    print(
        "===== PRIMARY 10NS VS 20NS ====="
    )

    for row in paired_rows:

        print(
            f"{row['target']}: "
            f"n={row['n_proteins']} "
            f"MAE10={row['mae_10ns']:.9g} "
            f"MAE20={row['mae_20ns']:.9g} "
            f"change={row['mae_percent_change_10_vs_20']:+.2f}% "
            f"paired_delta="
            f"{row['mean_error_delta_10_minus_20']:.9g} "
            f"CI95=["
            f"{row['ci95_low']:.9g},"
            f"{row['ci95_high']:.9g}] "
            f"10ns_better="
            f"{100.0 * row['ten_ns_better_fraction']:.1f}%"
        )

    print()
    print(
        "10NS_TRAJECTORY_REDUCTION_PERCENT: 90.0"
    )

    print(
        "10NS_AGGREGATE_PREFIX_US: 0.33"
    )

    print(
        "10NS_POTENTIALLY_AVOIDABLE_US: 2.97"
    )

    print(
        "20NS_TRAJECTORY_REDUCTION_PERCENT: 80.0"
    )

    print(
        "20NS_POTENTIALLY_AVOIDABLE_US: 2.64"
    )

    print(
        "EXTRA_POTENTIALLY_AVOIDABLE_US_10VS20: 0.33"
    )

    print()
    print(
        "TEN_NS_ANALYSIS: POST_HOC_ZERO_SHOT"
    )

    print(
        "MODEL_RETRAINING: NO"
    )

    print(
        "HPO_RERUN: NO"
    )

    print(
        "SCALER_REFIT: NO"
    )

    print(
        "PRIMARY_20NS_RESULT_CHANGED: NO"
    )

    print(
        "PREFIX_SENSITIVITY_EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()
