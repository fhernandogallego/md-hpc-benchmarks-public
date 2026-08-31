from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train deterministic leakage-free GRU baseline "
            "on one nested inner train/validation fold."
        )
    )

    parser.add_argument(
        "--fold-npz",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260811,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-5,
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def decode_strings(
    array: np.ndarray,
) -> list[str]:
    output = []

    for value in array.tolist():
        if isinstance(value, bytes):
            output.append(
                value.decode("utf-8")
            )
        else:
            output.append(str(value))

    return output


def set_deterministic(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.use_deterministic_algorithms(
        True
    )


def masked_equal_target_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if prediction.shape != target.shape:
        raise RuntimeError(
            "Prediction/target shape mismatch."
        )

    if mask.shape != target.shape:
        raise RuntimeError(
            "Mask/target shape mismatch."
        )

    if mask.dtype != torch.bool:
        raise RuntimeError(
            "Mask must be bool."
        )

    safe_target = torch.where(
        mask,
        target,
        torch.zeros_like(target),
    )

    error2 = (
        prediction
        - safe_target
    ).square()

    error2 = torch.where(
        mask,
        error2,
        torch.zeros_like(error2),
    )

    counts = mask.sum(
        dim=0
    )

    active = counts > 0

    if not bool(
        active.all()
    ):
        raise RuntimeError(
            "All five targets must be "
            "represented in this nested fold."
        )

    per_target = (
        error2.sum(
            dim=0
        )
        / counts
    )

    loss = per_target.mean()

    return (
        loss,
        per_target,
        counts,
    )


class GRUBaseline(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        _, hidden = self.gru(x)

        representation = hidden[-1]

        return self.output(
            representation
        )


def main() -> None:
    args = parse_args()

    fold_path = (
        args.fold_npz
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not fold_path.is_file():
        raise FileNotFoundError(
            fold_path
        )

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: "
            f"{output_dir}"
        )

    if args.hidden_size <= 0:
        raise RuntimeError(
            "hidden_size must be positive."
        )

    if args.max_epochs <= 0:
        raise RuntimeError(
            "max_epochs must be positive."
        )

    if args.patience <= 0:
        raise RuntimeError(
            "patience must be positive."
        )

    set_deterministic(
        args.seed
    )

    with np.load(
        fold_path,
        allow_pickle=False,
    ) as z:

        forbidden = [
            key
            for key in z.files
            if (
                "outer" in key.lower()
                or "test" in key.lower()
            )
        ]

        if forbidden:
            raise RuntimeError(
                "Outer-test-like arrays "
                f"found: {forbidden}"
            )

        X_train_np = (
            z["X_train_scaled"]
            .astype(np.float32)
        )

        y_train_np = (
            z["y_train_scaled"]
            .astype(np.float32)
        )

        mask_train_np = (
            z["mask_train"]
            .astype(bool)
        )

        X_valid_np = (
            z["X_valid_scaled"]
            .astype(np.float32)
        )

        y_valid_np = (
            z["y_valid_scaled"]
            .astype(np.float32)
        )

        mask_valid_np = (
            z["mask_valid"]
            .astype(bool)
        )

        train_indices = (
            z["train_indices"]
            .astype(np.int64)
        )

        valid_indices = (
            z["valid_indices"]
            .astype(np.int64)
        )

        train_sample_ids = decode_strings(
            z["train_sample_ids"]
        )

        valid_sample_ids = decode_strings(
            z["valid_sample_ids"]
        )

        train_system_ids = decode_strings(
            z["train_system_ids"]
        )

        valid_system_ids = decode_strings(
            z["valid_system_ids"]
        )

        target_names = decode_strings(
            z["target_names"]
        )

        feature_names = decode_strings(
            z["feature_names"]
        )

    if X_train_np.shape != (
        24,
        201,
        8,
    ):
        raise RuntimeError(
            "Unexpected X_train shape."
        )

    if X_valid_np.shape != (
        6,
        201,
        8,
    ):
        raise RuntimeError(
            "Unexpected X_valid shape."
        )

    if y_train_np.shape != (
        24,
        5,
    ):
        raise RuntimeError(
            "Unexpected y_train shape."
        )

    if y_valid_np.shape != (
        6,
        5,
    ):
        raise RuntimeError(
            "Unexpected y_valid shape."
        )

    if not np.isfinite(
        X_train_np
    ).all():
        raise RuntimeError(
            "X_train contains non-finite values."
        )

    if not np.isfinite(
        X_valid_np
    ).all():
        raise RuntimeError(
            "X_valid contains non-finite values."
        )

    if not np.isnan(
        y_train_np[
            ~mask_train_np
        ]
    ).all():
        raise RuntimeError(
            "Invalid y_train values are not NaN."
        )

    if not np.isnan(
        y_valid_np[
            ~mask_valid_np
        ]
    ).all():
        raise RuntimeError(
            "Invalid y_valid values are not NaN."
        )

    if not np.all(
        mask_train_np.sum(
            axis=0
        ) > 0
    ):
        raise RuntimeError(
            "Training is missing a target."
        )

    if not np.all(
        mask_valid_np.sum(
            axis=0
        ) > 0
    ):
        raise RuntimeError(
            "Validation is missing a target."
        )

    if set(
        train_indices.tolist()
    ) & set(
        valid_indices.tolist()
    ):
        raise RuntimeError(
            "Train/validation index leakage."
        )

    if set(
        train_system_ids
    ) & set(
        valid_system_ids
    ):
        raise RuntimeError(
            "Protein-group leakage."
        )

    X_train = torch.from_numpy(
        X_train_np
    )

    y_train = torch.from_numpy(
        y_train_np
    )

    mask_train = torch.from_numpy(
        mask_train_np
    )

    X_valid = torch.from_numpy(
        X_valid_np
    )

    y_valid = torch.from_numpy(
        y_valid_np
    )

    mask_valid = torch.from_numpy(
        mask_valid_np
    )

    model = GRUBaseline(
        input_size=8,
        hidden_size=args.hidden_size,
        output_size=5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    n_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    best_epoch = None
    best_valid_loss = float("inf")
    best_state = None
    best_valid_per_target = None

    # Separate checkpoint selection from the patience criterion.
    #
    # best_valid_loss:
    #     exact lowest validation loss observed; any strict
    #     improvement updates the checkpoint.
    #
    # patience_reference_loss:
    #     reference used only to decide whether an improvement is
    #     large enough (min_delta) to reset early-stopping patience.
    patience_reference_loss = float("inf")
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        args.max_epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        train_prediction = model(
            X_train
        )

        (
            train_loss,
            train_per_target,
            train_counts,
        ) = masked_equal_target_mse(
            train_prediction,
            y_train,
            mask_train,
        )

        if not torch.isfinite(
            train_loss
        ):
            raise RuntimeError(
                f"Epoch {epoch}: "
                "non-finite train loss."
            )

        train_loss.backward()

        for parameter in model.parameters():
            if (
                parameter.grad is not None
                and not torch.isfinite(
                    parameter.grad
                ).all()
            ):
                raise RuntimeError(
                    f"Epoch {epoch}: "
                    "non-finite gradient."
                )

        optimizer.step()

        # Metrics recorded in history must describe the same
        # post-update model state for both train and validation.
        # The pre-update train_loss above remains the objective
        # used for backpropagation.
        model.eval()

        with torch.no_grad():
            logged_train_prediction = model(
                X_train
            )

            (
                logged_train_loss,
                logged_train_per_target,
                _,
            ) = masked_equal_target_mse(
                logged_train_prediction,
                y_train,
                mask_train,
            )

            valid_prediction = model(
                X_valid
            )

            (
                valid_loss,
                valid_per_target,
                valid_counts,
            ) = masked_equal_target_mse(
                valid_prediction,
                y_valid,
                mask_valid,
            )

        if not torch.isfinite(
            logged_train_loss
        ):
            raise RuntimeError(
                f"Epoch {epoch}: "
                "non-finite logged train loss."
            )

        if not torch.isfinite(
            valid_loss
        ):
            raise RuntimeError(
                f"Epoch {epoch}: "
                "non-finite validation loss."
            )

        train_loss_value = float(
            logged_train_loss.detach()
        )

        valid_loss_value = float(
            valid_loss.detach()
        )

        row = {
            "epoch": epoch,
            "train_loss": (
                train_loss_value
            ),
            "valid_loss": (
                valid_loss_value
            ),
        }

        for index, name in enumerate(
            target_names
        ):
            row[
                f"train_mse_{name}"
            ] = float(
                logged_train_per_target[
                    index
                ].detach()
            )

            row[
                f"valid_mse_{name}"
            ] = float(
                valid_per_target[
                    index
                ].detach()
            )

        history.append(row)

        # Checkpoint policy:
        # save the true lowest validation loss observed.
        if valid_loss_value < best_valid_loss:
            best_valid_loss = (
                valid_loss_value
            )

            best_epoch = epoch

            best_state = deepcopy(
                model.state_dict()
            )

            best_valid_per_target = [
                float(value)
                for value
                in valid_per_target.detach()
            ]

        # Early-stopping policy:
        # min_delta affects only patience, not checkpoint selection.
        significant_improvement = (
            valid_loss_value
            < (
                patience_reference_loss
                - args.min_delta
            )
        )

        if significant_improvement:
            patience_reference_loss = (
                valid_loss_value
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint selected."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():
        final_train_prediction = model(
            X_train
        )

        (
            final_train_loss,
            final_train_per_target,
            _,
        ) = masked_equal_target_mse(
            final_train_prediction,
            y_train,
            mask_train,
        )

        final_valid_prediction = model(
            X_valid
        )

        (
            final_valid_loss,
            final_valid_per_target,
            _,
        ) = masked_equal_target_mse(
            final_valid_prediction,
            y_valid,
            mask_valid,
        )

    if abs(
        float(final_valid_loss)
        - best_valid_loss
    ) > 1e-7:
        raise RuntimeError(
            "Restored checkpoint does not "
            "match best validation loss."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    history_path = (
        output_dir
        / "history.csv"
    )

    history_fields = list(
        history[0].keys()
    )

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=history_fields,
        )

        writer.writeheader()
        writer.writerows(
            history
        )

    checkpoint_path = (
        output_dir
        / "best_model.pt"
    )

    torch.save(
        {
            "model_name": (
                "gru_baseline_v1"
            ),
            "model_state_dict": (
                best_state
            ),
            "input_size": 8,
            "hidden_size": (
                args.hidden_size
            ),
            "output_size": 5,
            "seed": args.seed,
            "best_epoch": (
                best_epoch
            ),
            "best_valid_loss": (
                best_valid_loss
            ),
            "target_names": (
                target_names
            ),
            "feature_names": (
                feature_names
            ),
        },
        checkpoint_path,
    )

    predictions_path = (
        output_dir
        / "best_predictions.npz"
    )

    np.savez(
        predictions_path,

        train_prediction_scaled=(
            final_train_prediction
            .cpu()
            .numpy()
            .astype(np.float64)
        ),

        valid_prediction_scaled=(
            final_valid_prediction
            .cpu()
            .numpy()
            .astype(np.float64)
        ),

        y_train_scaled=(
            y_train_np.astype(
                np.float64
            )
        ),

        y_valid_scaled=(
            y_valid_np.astype(
                np.float64
            )
        ),

        mask_train=(
            mask_train_np
        ),

        mask_valid=(
            mask_valid_np
        ),

        train_indices=(
            train_indices
        ),

        valid_indices=(
            valid_indices
        ),

        train_sample_ids=np.asarray(
            train_sample_ids
        ),

        valid_sample_ids=np.asarray(
            valid_sample_ids
        ),

        target_names=np.asarray(
            target_names
        ),
    )

    metadata = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "run_type": (
            "single_nested_fold_"
            "gru_baseline"
        ),

        "model": {
            "name": "gru_baseline_v1",
            "input_size": 8,
            "hidden_size": (
                args.hidden_size
            ),
            "num_layers": 1,
            "output_size": 5,
            "n_parameters": (
                n_parameters
            ),
        },

        "training": {
            "optimizer": "Adam",
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "batching": (
                "full_batch"
            ),
            "max_epochs": (
                args.max_epochs
            ),
            "patience": (
                args.patience
            ),
            "min_delta": (
                args.min_delta
            ),
            "seed": (
                args.seed
            ),
            "deterministic_algorithms": (
                True
            ),
            "torch_num_threads": 1,
        },

        "objective": {
            "name": (
                "masked_equal_target_mse"
            ),
            "target_scale": (
                "inner_train_standardized"
            ),
            "aggregation": (
                "per-target masked MSE, "
                "then equal mean over 5 targets"
            ),
        },

        "data": {
            "fold_npz": str(
                fold_path
            ),
            "fold_npz_sha256": (
                sha256_file(
                    fold_path
                )
            ),
            "n_train": 24,
            "n_valid": 6,
            "train_target_counts": (
                mask_train_np.sum(
                    axis=0
                ).astype(int).tolist()
            ),
            "valid_target_counts": (
                mask_valid_np.sum(
                    axis=0
                ).astype(int).tolist()
            ),
            "target_names": (
                target_names
            ),
        },

        "leakage_policy": {
            "inner_train_used_for_fit": (
                True
            ),
            "inner_valid_used_for_early_stopping": (
                True
            ),
            "outer_test_available_to_trainer": (
                False
            ),
            "outer_test_used": (
                False
            ),
        },

        "result": {
            "epochs_executed": (
                len(history)
            ),
            "best_epoch": (
                best_epoch
            ),
            "best_valid_loss": (
                best_valid_loss
            ),
            "best_valid_per_target_mse": (
                best_valid_per_target
            ),
            "restored_train_loss": float(
                final_train_loss
            ),
            "restored_valid_loss": float(
                final_valid_loss
            ),
            "restored_train_per_target_mse": [
                float(value)
                for value
                in final_train_per_target
            ],
            "restored_valid_per_target_mse": [
                float(value)
                for value
                in final_valid_per_target
            ],
        },
    }

    metadata_path = (
        output_dir
        / "run_metadata.json"
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

    files = [
        history_path,
        checkpoint_path,
        predictions_path,
        metadata_path,
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
            files,
            key=lambda value: value.name,
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )

    print(
        "========================================"
    )
    print(
        "GRU BASELINE TRAINING: PASS"
    )
    print(
        "========================================"
    )

    print(
        "model_parameters:",
        n_parameters,
    )

    print(
        "seed:",
        args.seed,
    )

    print(
        "epochs_executed:",
        len(history),
    )

    print(
        "best_epoch:",
        best_epoch,
    )

    print(
        "best_valid_loss:",
        best_valid_loss,
    )

    print(
        "restored_train_loss:",
        float(
            final_train_loss
        ),
    )

    print(
        "restored_valid_loss:",
        float(
            final_valid_loss
        ),
    )

    print(
        "train_target_counts:",
        mask_train_np.sum(
            axis=0
        ).tolist(),
    )

    print(
        "valid_target_counts:",
        mask_valid_np.sum(
            axis=0
        ).tolist(),
    )

    print(
        "outer_test_available:",
        "NO",
    )

    print(
        "Output:",
        output_dir,
    )


if __name__ == "__main__":
    main()
