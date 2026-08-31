#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_selected(
    path: Path,
    outer_fold: int,
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
        if int(row["outer_fold"])
        == outer_fold
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one selected "
            "Improved-GWO candidate."
        )

    row = matches[0]

    candidate = {
        "hidden_size": int(
            row["hidden_size"]
        ),
        "num_layers": int(
            row["num_layers"]
        ),
        "dropout": float(
            row["dropout"]
        ),
        "learning_rate": float(
            row["learning_rate"]
        ),
        "weight_decay": float(
            row["weight_decay"]
        ),
    }

    return row, candidate


def import_space_module(project: Path):
    optimization = (
        project
        / "src"
        / "optimization"
    )

    sys.path.insert(
        0,
        str(optimization),
    )

    from gwo_space import (  # noqa
        gru_parameter_count,
        load_spec,
        trainer_arguments,
    )

    return (
        gru_parameter_count,
        load_spec,
        trainer_arguments,
    )


def run_fold(
    *,
    fold_id,
    fold_npz,
    fold_output,
    candidate,
    spec,
    trainer,
    trainer_arguments,
    expected_parameters,
):
    env = os.environ.copy()

    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        env[key] = "1"

    env["PYTHONNOUSERSITE"] = "1"

    command = [
        sys.executable,
        str(trainer),
        "--fold-npz",
        str(fold_npz),
        "--output-dir",
        str(fold_output),
    ]

    command += trainer_arguments(
        candidate,
        spec,
    )

    start = time.perf_counter()

    completed = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{fold_id} failed\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    metadata_path = (
        fold_output
        / "run_metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        metadata[
            "leakage_policy"
        ][
            "outer_test_available_to_trainer"
        ]
        is not False
    ):
        raise RuntimeError(
            f"{fold_id}: outer test available"
        )

    if (
        metadata[
            "leakage_policy"
        ][
            "outer_test_used"
        ]
        is not False
    ):
        raise RuntimeError(
            f"{fold_id}: outer test used"
        )

    actual_parameters = int(
        metadata[
            "model"
        ][
            "n_parameters"
        ]
    )

    if (
        actual_parameters
        != expected_parameters
    ):
        raise RuntimeError(
            f"{fold_id}: parameter count "
            f"{actual_parameters} != "
            f"{expected_parameters}"
        )

    return {
        "fold_id": fold_id,
        "fold_npz": str(fold_npz),
        "fold_npz_sha256": (
            sha256_file(fold_npz)
        ),
        "runtime_seconds": runtime,
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
        "best_valid_loss": float(
            metadata[
                "result"
            ][
                "best_valid_loss"
            ]
        ),
        "n_parameters": (
            actual_parameters
        ),
        "outer_test_used": False,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--selected-csv",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--spec",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--trainer",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--outer-fold",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    project = (
        args.project
        .expanduser()
        .resolve()
    )

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    selected_csv = (
        args.selected_csv
        .expanduser()
        .resolve()
    )

    spec_path = (
        args.spec
        .expanduser()
        .resolve()
    )

    trainer = (
        args.trainer
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    outer = int(
        args.outer_fold
    )

    if not 1 <= outer <= 11:
        raise RuntimeError(
            "outer_fold must be 1..11"
        )

    if output_dir.exists():
        raise RuntimeError(
            f"Output exists: {output_dir}"
        )

    (
        gru_parameter_count,
        load_spec,
        trainer_arguments,
    ) = import_space_module(
        project
    )

    spec = load_spec(
        spec_path
    )

    selected_row, selected = (
        load_selected(
            selected_csv,
            outer,
        )
    )

    if (
        selected[
            "hidden_size"
        ]
        != 96
    ):
        raise RuntimeError(
            "Expected selected boundary "
            "candidate hidden_size=96."
        )

    outer_dir = (
        data_root
        / "folds"
    )

    fold_paths = []

    for inner in range(
        1,
        6,
    ):
        fold = (
            outer_dir
            / (
                f"fold_{outer:02d}"
                f"_inner_{inner:02d}.npz"
            )
        )

        if not fold.is_file():
            raise FileNotFoundError(
                fold
            )

        fold_paths.append(
            (
                f"fold_{outer:02d}"
                f"_inner_{inner:02d}",
                fold,
            )
        )

    temp_root = Path(
        str(output_dir)
        + ".work"
    )

    if temp_root.exists():
        raise RuntimeError(
            f"Work directory exists: "
            f"{temp_root}"
        )

    temp_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    hidden_sizes = [
        96,
        128,
        160,
    ]

    candidate_results = []

    for hidden_size in hidden_sizes:

        candidate = dict(
            selected
        )

        candidate[
            "hidden_size"
        ] = hidden_size

        expected_parameters = int(
            gru_parameter_count(
                hidden_size,
                candidate[
                    "num_layers"
                ],
            )
        )

        candidate_dir = (
            temp_root
            / f"hidden_{hidden_size:03d}"
        )

        candidate_dir.mkdir()

        futures = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(
                int(args.workers),
                5,
            )
        ) as executor:

            for fold_id, fold_npz in (
                fold_paths
            ):
                fold_output = (
                    candidate_dir
                    / fold_id
                )

                futures.append(
                    executor.submit(
                        run_fold,
                        fold_id=fold_id,
                        fold_npz=fold_npz,
                        fold_output=fold_output,
                        candidate=candidate,
                        spec=spec,
                        trainer=trainer,
                        trainer_arguments=(
                            trainer_arguments
                        ),
                        expected_parameters=(
                            expected_parameters
                        ),
                    )
                )

            fold_records = [
                future.result()
                for future in futures
            ]

        fold_records.sort(
            key=lambda row:
            row["fold_id"]
        )

        losses = [
            float(
                row[
                    "best_valid_loss"
                ]
            )
            for row in fold_records
        ]

        if len(losses) != 5:
            raise RuntimeError(
                "Expected five fold losses."
            )

        if not all(
            math.isfinite(value)
            for value in losses
        ):
            raise RuntimeError(
                "Non-finite fitness component."
            )

        fitness = (
            sum(losses)
            / 5.0
        )

        candidate_results.append(
            {
                "hidden_size": (
                    hidden_size
                ),
                "candidate": (
                    candidate
                ),
                "parameter_count": (
                    expected_parameters
                ),
                "fitness": (
                    fitness
                ),
                "fold_losses": (
                    losses
                ),
                "fold_records": (
                    fold_records
                ),
                "outer_test_used": (
                    False
                ),
            }
        )

    reference_original = float(
        selected_row[
            "inner_fitness"
        ]
    )

    rerun96 = next(
        item
        for item in candidate_results
        if item["hidden_size"] == 96
    )

    drift = (
        float(
            rerun96["fitness"]
        )
        - reference_original
    )

    ranked = sorted(
        candidate_results,
        key=lambda row: (
            float(
                row["fitness"]
            ),
            int(
                row[
                    "parameter_count"
                ]
            ),
            int(
                row[
                    "hidden_size"
                ]
            ),
        ),
    )

    best = ranked[0]

    summary = {
        "analysis_type": (
            "hidden_size_boundary_sensitivity"
        ),
        "outer_fold": outer,
        "hidden_sizes_evaluated": (
            hidden_sizes
        ),
        "fixed_from_selected_improved_gwo": {
            "num_layers": (
                selected[
                    "num_layers"
                ]
            ),
            "dropout": (
                selected[
                    "dropout"
                ]
            ),
            "learning_rate": (
                selected[
                    "learning_rate"
                ]
            ),
            "weight_decay": (
                selected[
                    "weight_decay"
                ]
            ),
            "selected_optimizer_seed": int(
                selected_row[
                    "selected_optimizer_seed"
                ]
            ),
        },
        "original_selected_96_fitness": (
            reference_original
        ),
        "rerun_96_fitness": float(
            rerun96[
                "fitness"
            ]
        ),
        "rerun_96_minus_original": (
            drift
        ),
        "best_hidden_size": int(
            best[
                "hidden_size"
            ]
        ),
        "best_fitness": float(
            best[
                "fitness"
            ]
        ),
        "outer_test_evaluated": False,
        "sensitivity_outside_frozen_search_space": (
            True
        ),
        "note": (
            "Only hidden_size is varied. "
            "All other selected Improved-GWO "
            "hyperparameters and the frozen "
            "training protocol are unchanged."
        ),
        "candidates": (
            candidate_results
        ),
        "provenance": {
            "selected_csv": str(
                selected_csv
            ),
            "selected_csv_sha256": (
                sha256_file(
                    selected_csv
                )
            ),
            "spec": str(
                spec_path
            ),
            "spec_sha256": (
                sha256_file(
                    spec_path
                )
            ),
            "trainer": str(
                trainer
            ),
            "trainer_sha256": (
                sha256_file(
                    trainer
                )
            ),
        },
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_json(
        output_dir
        / "summary.json",
        summary,
    )

    with (
        output_dir
        / "candidate_results.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        fieldnames = [
            "outer_fold",
            "hidden_size",
            "fitness",
            "parameter_count",
            "num_layers",
            "dropout",
            "learning_rate",
            "weight_decay",
            "original_selected_96_fitness",
            "outer_test_used",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in candidate_results:
            candidate = item[
                "candidate"
            ]

            writer.writerow(
                {
                    "outer_fold": (
                        outer
                    ),
                    "hidden_size": (
                        item[
                            "hidden_size"
                        ]
                    ),
                    "fitness": (
                        item[
                            "fitness"
                        ]
                    ),
                    "parameter_count": (
                        item[
                            "parameter_count"
                        ]
                    ),
                    "num_layers": (
                        candidate[
                            "num_layers"
                        ]
                    ),
                    "dropout": (
                        candidate[
                            "dropout"
                        ]
                    ),
                    "learning_rate": (
                        candidate[
                            "learning_rate"
                        ]
                    ),
                    "weight_decay": (
                        candidate[
                            "weight_decay"
                        ]
                    ),
                    "original_selected_96_fitness": (
                        reference_original
                    ),
                    "outer_test_used": (
                        False
                    ),
                }
            )

    files = [
        "candidate_results.csv",
        "summary.json",
    ]

    with (
        output_dir
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for name in files:
            handle.write(
                sha256_file(
                    output_dir
                    / name
                )
                + "  "
                + name
                + "\n"
            )

    # Keep full fold-level trainer artifacts.
    final_folds = (
        output_dir
        / "fold_runs"
    )

    temp_root.rename(
        final_folds
    )

    print(
        "========================================"
    )
    print(
        "HIDDEN BOUNDARY SENSITIVITY"
    )
    print(
        "========================================"
    )

    print(
        "outer_fold:",
        outer,
    )

    print(
        "original_96_fitness:",
        reference_original,
    )

    for item in candidate_results:
        print(
            "hidden=",
            item[
                "hidden_size"
            ],
            "fitness=",
            item[
                "fitness"
            ],
            "parameters=",
            item[
                "parameter_count"
            ],
        )

    print(
        "rerun96_minus_original:",
        drift,
    )

    print(
        "best_hidden_size:",
        best[
            "hidden_size"
        ],
    )

    print(
        "outer_test_evaluated: NO"
    )

    print(
        "HIDDEN_BOUNDARY_OUTER: PASS"
    )


if __name__ == "__main__":
    main()
