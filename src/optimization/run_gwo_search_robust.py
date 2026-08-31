#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import run_gwo_search as base


NUMERICAL_FAILURE_FOLD_LOSS = 1_000_000.0

NUMERICAL_MARKERS = (
    "non-finite gradient",
    "non-finite train loss",
    "non-finite logged train loss",
    "non-finite validation loss",
)


def is_numerical_training_failure(
    text: str,
) -> bool:
    lower = text.lower()

    return any(
        marker in lower
        for marker in NUMERICAL_MARKERS
    )


class RobustRealEvaluator(
    base.RealEvaluator
):
    def _run_fold(
        self,
        fold_id,
        candidate,
        temp_root,
    ):
        try:
            return super()._run_fold(
                fold_id,
                candidate,
                temp_root,
            )

        except RuntimeError as exc:
            message = str(exc)

            if "see " not in message:
                raise

            diagnostic_path = Path(
                message.rsplit(
                    "see ",
                    1,
                )[1].strip()
            )

            if not diagnostic_path.is_file():
                raise

            diagnostic_text = (
                diagnostic_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            if not (
                is_numerical_training_failure(
                    diagnostic_text
                )
            ):
                raise

            fold_npz = (
                self.data_root
                / "folds"
                / f"{fold_id}.npz"
            )

            expected_parameters = (
                base.gru_parameter_count(
                    candidate[
                        "hidden_size"
                    ],
                    candidate[
                        "num_layers"
                    ],
                )
            )

            epoch_match = re.search(
                r"Epoch\s+(\d+)",
                diagnostic_text,
            )

            failure_epoch = (
                int(
                    epoch_match.group(1)
                )
                if epoch_match
                else -1
            )

            return {
                "fold_id": fold_id,
                "fold_npz": str(
                    fold_npz
                ),
                "fold_npz_sha256": (
                    base.sha256_file(
                        fold_npz
                    )
                ),
                "best_valid_loss": (
                    NUMERICAL_FAILURE_FOLD_LOSS
                ),
                "best_epoch": -1,
                "epochs_executed": (
                    failure_epoch
                ),
                "runtime_seconds": 0.0,
                "n_parameters": int(
                    expected_parameters
                ),
                "outer_test_used": False,
                "numerically_invalid": True,
                "failure_kind": (
                    "non_finite_training"
                ),
                "failure_epoch": (
                    failure_epoch
                ),
                "failure_diagnostic_sha256": (
                    base.sha256_file(
                        diagnostic_path
                    )
                ),
            }


def main():
    # Monkey-patch only the evaluator class.
    #
    # All GWO logic, decoding, ranking, M1/M2/M3,
    # budget accounting and successful training
    # behavior remain in the frozen base engine.
    base.RealEvaluator = (
        RobustRealEvaluator
    )

    base.main()


if __name__ == "__main__":
    main()
