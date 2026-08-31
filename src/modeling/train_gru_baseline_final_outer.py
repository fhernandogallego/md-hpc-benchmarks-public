#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_gru_baseline import (
    GRUBaseline,
    masked_equal_target_mse,
    set_deterministic,
)


SEED = 20260811
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


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


def get_plan_row(
    path,
    outer,
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
            "Expected exactly one "
            "epoch-plan row"
        )

    return matches[0]


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
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    outer = int(
        args.outer_fold
    )

    if not 1 <= outer <= 11:
        raise RuntimeError(
            "outer must be 1..11"
        )

    package = (
        args.package
        .expanduser()
        .resolve()
    )

    plan = (
        args.epoch_plan
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

    if not package.is_file():
        raise FileNotFoundError(
            package
        )

    if not plan.is_file():
        raise FileNotFoundError(
            plan
        )

    if output.exists() or work.exists():
        raise RuntimeError(
            "Output already exists"
        )

    row = get_plan_row(
        plan,
        outer,
    )

    if int(
        row["hidden_size"]
    ) != HIDDEN_SIZE:
        raise RuntimeError(
            "hidden mismatch"
        )

    if int(
        row["num_layers"]
    ) != 1:
        raise RuntimeError(
            "layers mismatch"
        )

    if float(
        row["learning_rate"]
    ) != LEARNING_RATE:
        raise RuntimeError(
            "lr mismatch"
        )

    if float(
        row["weight_decay"]
    ) != WEIGHT_DECAY:
        raise RuntimeError(
            "weight decay mismatch"
        )

    if int(
        row["trainer_seed"]
    ) != SEED:
        raise RuntimeError(
            "seed mismatch"
        )

    final_epochs = int(
        row["final_epochs"]
    )

    if final_epochs <= 0:
        raise RuntimeError(
            "Invalid final_epochs"
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
                f"Forbidden arrays: "
                f"{forbidden}"
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
        30,
        201,
        8,
    ):
        raise RuntimeError(
            f"Bad X shape: {X_np.shape}"
        )

    if y_np.shape != (
        30,
        5,
    ):
        raise RuntimeError(
            "Bad y shape"
        )

    if mask_np.shape != (
        30,
        5,
    ):
        raise RuntimeError(
            "Bad mask shape"
        )

    if len(
        set(
            train_system_ids
        )
    ) != 10:
        raise RuntimeError(
            "Expected 10 train proteins"
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
            "Non-finite target"
        )

    if not np.isnan(
        y_np[
            ~mask_np
        ]
    ).all():
        raise RuntimeError(
            "Masked target was imputed"
        )

    set_deterministic(
        SEED
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

    model = GRUBaseline(
        input_size=8,
        hidden_size=HIDDEN_SIZE,
        output_size=5,
    )

    n_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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

        loss, _, _ = (
            masked_equal_target_mse(
                prediction,
                y,
                mask,
            )
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Epoch {epoch}: "
                "non-finite train loss"
            )

        loss.backward()

        for parameter in (
            model.parameters()
        ):

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
                "non-finite logged loss"
            )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    float(
                        logged_loss
                    ),
            }
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
            fieldnames=[
                "epoch",
                "train_loss",
            ],
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
                "gru_baseline_final_outer_v1",

            "outer_fold":
                outer,

            "model_state_dict":
                model.state_dict(),

            "input_size":
                8,

            "hidden_size":
                HIDDEN_SIZE,

            "num_layers":
                1,

            "output_size":
                5,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "seed":
                SEED,

            "final_epochs":
                final_epochs,

            "parameter_count":
                n_parameters,

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

        prediction_scaled=(
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
    )

    metadata = {
        "analysis_type":
            "fixed_gru_baseline_final_outer",

        "outer_fold":
            outer,

        "training_samples":
            30,

        "training_proteins":
            10,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            1,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "seed":
            SEED,

        "parameter_count":
            n_parameters,

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

        "hyperparameter_tuning":
            False,

        "outer_test_arrays_present":
            False,

        "outer_test_targets_used":
            False,

        "outer_test_evaluated":
            False,

        "source_package_sha256":
            sha256_file(
                package
            ),

        "epoch_plan_sha256":
            sha256_file(
                plan
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

        "target_valid_counts":
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

    sums = (
        work
        / "SHA256SUMS.txt"
    )

    files = [
        history_path,
        checkpoint_path,
        predictions_path,
        metadata_path,
    ]

    with sums.open(
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
        "outer:",
        outer,
    )

    print(
        "hidden_size: 32"
    )

    print(
        "final_epochs:",
        final_epochs,
    )

    print(
        "train_loss:",
        float(
            final_loss
        ),
    )

    print(
        "validation_used: NO"
    )

    print(
        "outer_test_evaluated: NO"
    )

    print(
        "FINAL_BASELINE_TRAINING: PASS"
    )


if __name__ == "__main__":
    main()
