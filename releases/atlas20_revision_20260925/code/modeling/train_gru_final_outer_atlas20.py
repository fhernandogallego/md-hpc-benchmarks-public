#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from train_gru_tunable_atlas20 import (
    GRUTunable,
    masked_equal_target_mse,
    set_deterministic,
)


TRAINER_SEED = 20260811


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

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


def read_csv_row(
    path: Path,
    outer: int,
):
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:

        rows = list(
            csv.DictReader(handle)
        )

    matches = [
        row
        for row in rows
        if int(
            row["outer_fold"]
        ) == outer
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one row for "
            f"outer={outer}"
        )

    return matches[0]


def same_float(a, b):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer-fold",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--package",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--epoch-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--selected-models",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    outer = int(
        args.outer_fold
    )

    if not 1 <= outer <= 20:
        raise RuntimeError(
            "outer must be 1..20"
        )

    package_path = (
        args.package
        .expanduser()
        .resolve()
    )

    epoch_plan_path = (
        args.epoch_plan
        .expanduser()
        .resolve()
    )

    selected_path = (
        args.selected_models
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

    for path in [
        package_path,
        epoch_plan_path,
        selected_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    if output.exists():
        raise RuntimeError(
            f"Output exists: {output}"
        )

    if work.exists():
        raise RuntimeError(
            f"Work exists: {work}"
        )

    epoch_row = read_csv_row(
        epoch_plan_path,
        outer,
    )

    selected_row = read_csv_row(
        selected_path,
        outer,
    )

    integer_fields = [
        "hidden_size",
        "num_layers",
        "parameter_count",
    ]

    for field in integer_fields:
        if (
            int(epoch_row[field])
            != int(selected_row[field])
        ):
            raise RuntimeError(
                f"{field} differs between "
                "selection and epoch plan"
            )

    float_fields = [
        "dropout",
        "learning_rate",
        "weight_decay",
    ]

    for field in float_fields:
        if not same_float(
            epoch_row[field],
            selected_row[field],
        ):
            raise RuntimeError(
                f"{field} differs between "
                "selection and epoch plan"
            )

    hidden_size = int(
        epoch_row["hidden_size"]
    )

    num_layers = int(
        epoch_row["num_layers"]
    )

    dropout = float(
        epoch_row["dropout"]
    )

    learning_rate = float(
        epoch_row["learning_rate"]
    )

    weight_decay = float(
        epoch_row["weight_decay"]
    )

    parameter_count_expected = int(
        epoch_row["parameter_count"]
    )

    final_epochs = int(
        epoch_row["final_epochs"]
    )

    if final_epochs <= 0:
        raise RuntimeError(
            "final_epochs must be positive"
        )

    with np.load(
        package_path,
        allow_pickle=False,
    ) as z:

        # target_valid_fit_counts is TRAIN metadata:
        # it stores the number of observed target values
        # used to fit the training target scalers.
        #
        # Block real validation/test arrays only.
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
                "Test/validation arrays found: "
                f"{forbidden}"
            )

        required = {
            "X_train_scaled",
            "y_train_scaled",
            "mask_train",
            "train_indices",
            "train_sample_ids",
            "train_system_ids",
            "train_replicas",
            "feature_names",
            "target_names",
            "feature_mean",
            "feature_std",
            "target_mean",
            "target_std",
        }

        missing = required.difference(
            z.files
        )

        if missing:
            raise RuntimeError(
                f"Missing arrays: "
                f"{sorted(missing)}"
            )

        X_np = (
            z["X_train_scaled"]
            .astype(np.float32)
        )

        y_np = (
            z["y_train_scaled"]
            .astype(np.float32)
        )

        mask_np = (
            z["mask_train"]
            .astype(bool)
        )

        train_indices = (
            z["train_indices"]
            .astype(np.int64)
        )

        train_sample_ids = (
            decode_strings(
                z["train_sample_ids"]
            )
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

    if X_np.shape != (
        57,
        201,
        8,
    ):
        raise RuntimeError(
            f"Unexpected X shape: "
            f"{X_np.shape}"
        )

    if y_np.shape != (
        57,
        5,
    ):
        raise RuntimeError(
            f"Unexpected y shape: "
            f"{y_np.shape}"
        )

    if mask_np.shape != (
        57,
        5,
    ):
        raise RuntimeError(
            "Unexpected mask shape"
        )

    if len(
        set(train_system_ids)
    ) != 19:
        raise RuntimeError(
            "Expected 19 training proteins"
        )

    if not np.isfinite(
        X_np
    ).all():
        raise RuntimeError(
            "Non-finite X"
        )

    if not np.isfinite(
        y_np[
            mask_np
        ]
    ).all():
        raise RuntimeError(
            "Non-finite valid targets"
        )

    if not np.isnan(
        y_np[
            ~mask_np
        ]
    ).all():
        raise RuntimeError(
            "Masked targets were imputed"
        )

    if not np.all(
        mask_np.sum(
            axis=0
        ) > 0
    ):
        raise RuntimeError(
            "A target is absent from train"
        )

    if np.any(
        feature_std <= 0.0
    ):
        raise RuntimeError(
            "Invalid feature scaler"
        )

    if np.any(
        target_std <= 0.0
    ):
        raise RuntimeError(
            "Invalid target scaler"
        )

    set_deterministic(
        TRAINER_SEED
    )

    effective_dropout = (
        dropout
        if num_layers > 1
        else 0.0
    )

    X = torch.from_numpy(
        X_np
    )

    y = torch.from_numpy(
        y_np
    )

    mask = torch.from_numpy(
        mask_np
    )

    model = GRUTunable(
        input_size=8,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=effective_dropout,
        output_size=5,
    )

    actual_parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    if (
        actual_parameter_count
        != parameter_count_expected
    ):
        raise RuntimeError(
            "Parameter-count mismatch: "
            f"{actual_parameter_count} != "
            f"{parameter_count_expected}"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []

    for epoch in range(
        1,
        final_epochs + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            X
        )

        (
            loss,
            _,
            _,
        ) = masked_equal_target_mse(
            prediction,
            y,
            mask,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Epoch {epoch}: "
                "non-finite train loss"
            )

        loss.backward()

        for parameter in model.parameters():

            if (
                parameter.grad is not None
                and not torch.isfinite(
                    parameter.grad
                ).all()
            ):
                raise RuntimeError(
                    f"Epoch {epoch}: "
                    "non-finite gradient"
                )

        optimizer.step()

        model.eval()

        with torch.no_grad():

            logged_prediction = model(
                X
            )

            (
                logged_loss,
                logged_per_target,
                logged_counts,
            ) = masked_equal_target_mse(
                logged_prediction,
                y,
                mask,
            )

        if not torch.isfinite(
            logged_loss
        ):
            raise RuntimeError(
                f"Epoch {epoch}: "
                "non-finite logged train loss"
            )

        row = {
            "epoch":
                epoch,
            "train_loss":
                float(
                    logged_loss
                ),
        }

        for index, name in enumerate(
            target_names
        ):

            row[
                f"train_mse_{name}"
            ] = float(
                logged_per_target[
                    index
                ]
            )

            row[
                f"train_n_{name}"
            ] = int(
                logged_counts[
                    index
                ]
            )

        history.append(
            row
        )

    model.eval()

    with torch.no_grad():

        final_prediction = model(
            X
        )

        (
            final_loss,
            final_per_target,
            final_counts,
        ) = masked_equal_target_mse(
            final_prediction,
            y,
            mask,
        )

    if not torch.isfinite(
        final_loss
    ):
        raise RuntimeError(
            "Final training loss "
            "is non-finite"
        )

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    history_path = (
        work
        / "history.csv"
    )

    with history_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                history[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            history
        )

    checkpoint_path = (
        work
        / "final_model.pt"
    )

    torch.save(
        {
            "model_name":
                "gru_final_outer_v1",

            "outer_fold":
                outer,

            "model_state_dict":
                model.state_dict(),

            "input_size":
                8,

            "hidden_size":
                hidden_size,

            "num_layers":
                num_layers,

            "dropout_requested":
                dropout,

            "dropout_effective":
                effective_dropout,

            "output_size":
                5,

            "learning_rate":
                learning_rate,

            "weight_decay":
                weight_decay,

            "seed":
                TRAINER_SEED,

            "final_epochs":
                final_epochs,

            "parameter_count":
                actual_parameter_count,

            "feature_names":
                feature_names,

            "target_names":
                target_names,

            "feature_mean":
                feature_mean,

            "feature_std":
                feature_std,

            "target_mean":
                target_mean,

            "target_std":
                target_std,

            "source_package_sha256":
                sha256_file(
                    package_path
                ),

            "epoch_plan_sha256":
                sha256_file(
                    epoch_plan_path
                ),

            "selected_models_sha256":
                sha256_file(
                    selected_path
                ),

            "outer_test_used":
                False,
        },
        checkpoint_path,
    )

    predictions_path = (
        work
        / "train_predictions.npz"
    )

    np.savez(
        predictions_path,

        train_prediction_scaled=(
            final_prediction
            .cpu()
            .numpy()
            .astype(np.float64)
        ),

        train_indices=(
            train_indices
        ),

        train_sample_ids=(
            np.asarray(
                train_sample_ids
            )
        ),

        train_system_ids=(
            np.asarray(
                train_system_ids
            )
        ),

        target_mask=(
            mask_np
        ),

        target_names=(
            np.asarray(
                target_names
            )
        ),
    )

    metadata = {
        "analysis_type":
            "final_outer_training",

        "outer_fold":
            outer,

        "training_samples":
            57,

        "training_systems":
            19,

        "trainer_seed":
            TRAINER_SEED,

        "hidden_size":
            hidden_size,

        "num_layers":
            num_layers,

        "dropout_requested":
            dropout,

        "dropout_effective":
            effective_dropout,

        "learning_rate":
            learning_rate,

        "weight_decay":
            weight_decay,

        "parameter_count":
            actual_parameter_count,

        "epochs_planned":
            final_epochs,

        "epochs_completed":
            final_epochs,

        "epoch_policy":
            "median_of_five_inner_best_epochs",

        "validation_used":
            False,

        "early_stopping_used":
            False,

        "checkpoint_policy":
            "state_after_exact_final_epoch",

        "outer_test_arrays_present":
            False,

        "outer_test_targets_used":
            False,

        "outer_test_evaluated":
            False,

        "source_package":
            str(package_path),

        "source_package_sha256":
            sha256_file(
                package_path
            ),

        "epoch_plan_sha256":
            sha256_file(
                epoch_plan_path
            ),

        "selected_models_sha256":
            sha256_file(
                selected_path
            ),

        "final_train_loss_scaled":
            float(
                final_loss
            ),

        "final_train_mse_per_target_scaled":
            {
                name: float(
                    final_per_target[
                        index
                    ]
                )
                for index, name
                in enumerate(
                    target_names
                )
            },

        "train_valid_counts":
            {
                name: int(
                    final_counts[
                        index
                    ]
                )
                for index, name
                in enumerate(
                    target_names
                )
            },
    }

    metadata_path = (
        work
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

    written = [
        checkpoint_path,
        predictions_path,
        history_path,
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

        for path in written:

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
        "========================================"
    )

    print(
        "FINAL OUTER GRU TRAINING"
    )

    print(
        "========================================"
    )

    print(
        "outer_fold:",
        outer,
    )

    print(
        "hidden_size:",
        hidden_size,
    )

    print(
        "num_layers:",
        num_layers,
    )

    print(
        "learning_rate:",
        learning_rate,
    )

    print(
        "weight_decay:",
        weight_decay,
    )

    print(
        "final_epochs:",
        final_epochs,
    )

    print(
        "training_samples:",
        30,
    )

    print(
        "training_systems:",
        10,
    )

    print(
        "final_train_loss_scaled:",
        float(
            final_loss
        ),
    )

    print(
        "validation_used: NO"
    )

    print(
        "early_stopping_used: NO"
    )

    print(
        "outer_test_evaluated: NO"
    )

    print(
        "FINAL_OUTER_TRAINING: PASS"
    )


if __name__ == "__main__":
    main()
