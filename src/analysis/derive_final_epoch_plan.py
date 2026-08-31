#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


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


def read_selected(
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
            "Expected one selected model"
        )

    row = matches[0]

    candidate = {
        "hidden_size":
            int(row["hidden_size"]),
        "num_layers":
            int(row["num_layers"]),
        "dropout":
            float(row["dropout"]),
        "learning_rate":
            float(
                row["learning_rate"]
            ),
        "weight_decay":
            float(
                row["weight_decay"]
            ),
    }

    return row, candidate


def run_fold(
    *,
    python,
    trainer,
    fold_npz,
    fold_id,
    output,
    candidate,
    trainer_arguments,
    spec,
):

    command = [
        python,
        str(trainer),
        "--fold-npz",
        str(fold_npz),
        "--output-dir",
        str(output),
    ]

    command += trainer_arguments(
        candidate,
        spec,
    )

    env = os.environ.copy()

    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        env[key] = "1"

    env[
        "PYTHONNOUSERSITE"
    ] = "1"

    started = time.monotonic()

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )

    runtime = (
        time.monotonic()
        - started
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"{fold_id} failed\n"
            f"STDOUT:\n"
            f"{process.stdout}\n"
            f"STDERR:\n"
            f"{process.stderr}"
        )

    metadata = json.loads(
        (
            output
            / "run_metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    leakage = metadata[
        "leakage_policy"
    ]

    if (
        leakage[
            "outer_test_used"
        ]
        is not False
    ):
        raise RuntimeError(
            "Outer test used"
        )

    result = metadata["result"]

    return {
        "fold_id": fold_id,
        "best_epoch":
            int(
                result[
                    "best_epoch"
                ]
            ),
        "best_valid_loss":
            float(
                result[
                    "best_valid_loss"
                ]
            ),
        "epochs_executed":
            int(
                result[
                    "epochs_executed"
                ]
            ),
        "runtime_seconds":
            runtime,
        "outer_test_used":
            False,
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

    output = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    outer = int(
        args.outer_fold
    )

    if not 1 <= outer <= 11:
        raise RuntimeError(
            "outer must be 1..11"
        )

    if output.exists():
        raise RuntimeError(
            f"Output exists: {output}"
        )

    sys.path.insert(
        0,
        str(
            project
            / "src"
            / "optimization"
        ),
    )

    from gwo_space import (
        gru_parameter_count,
        load_spec,
        trainer_arguments,
    )

    spec = load_spec(
        spec_path
    )

    selected_row, candidate = (
        read_selected(
            selected_csv,
            outer,
        )
    )

    expected_params = (
        gru_parameter_count(
            candidate["hidden_size"],
            candidate["num_layers"],
        )
    )

    if (
        expected_params
        != int(
            selected_row[
                "parameter_count"
            ]
        )
    ):
        raise RuntimeError(
            "Parameter count mismatch"
        )

    temp = Path(
        str(output)
        + ".work"
    )

    if temp.exists():
        raise RuntimeError(
            f"Work exists: {temp}"
        )

    temp.mkdir(
        parents=True,
        exist_ok=False,
    )

    futures = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(
            args.workers,
            5,
        )
    ) as executor:

        for inner in range(1, 6):

            fold_id = (
                f"fold_{outer:02d}"
                f"_inner_{inner:02d}"
            )

            fold_npz = (
                data_root
                / "folds"
                / f"{fold_id}.npz"
            )

            if not fold_npz.is_file():
                raise FileNotFoundError(
                    fold_npz
                )

            fold_output = (
                temp
                / fold_id
            )

            futures.append(
                executor.submit(
                    run_fold,
                    python=sys.executable,
                    trainer=trainer,
                    fold_npz=fold_npz,
                    fold_id=fold_id,
                    output=fold_output,
                    candidate=candidate,
                    trainer_arguments=(
                        trainer_arguments
                    ),
                    spec=spec,
                )
            )

        records = [
            future.result()
            for future in futures
        ]

    records.sort(
        key=lambda row:
            row["fold_id"]
    )

    losses = [
        row["best_valid_loss"]
        for row in records
    ]

    best_epochs = [
        row["best_epoch"]
        for row in records
    ]

    if not all(
        epoch >= 1
        for epoch in best_epochs
    ):
        raise RuntimeError(
            "Invalid best_epoch"
        )

    rerun_fitness = (
        sum(losses)
        / 5.0
    )

    selected_fitness = float(
        selected_row[
            "inner_fitness"
        ]
    )

    drift = (
        rerun_fitness
        - selected_fitness
    )

    if not math.isfinite(
        rerun_fitness
    ):
        raise RuntimeError(
            "Non-finite rerun fitness"
        )

    if abs(drift) > 1e-5:
        raise RuntimeError(
            f"Fitness reproduction drift "
            f"too large: {drift}"
        )

    # Predeclared final training-duration rule:
    #
    # median of five inner-fold best epochs.
    #
    # Five values -> exact observed integer.
    final_epochs = int(
        statistics.median(
            best_epochs
        )
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary = {
        "analysis_type":
            "final_epoch_derivation",
        "outer_fold":
            outer,
        "selected_optimizer_seed":
            int(
                selected_row[
                    "selected_optimizer_seed"
                ]
            ),
        "candidate":
            candidate,
        "parameter_count":
            expected_params,
        "selected_inner_fitness":
            selected_fitness,
        "rerun_inner_fitness":
            rerun_fitness,
        "fitness_drift":
            drift,
        "best_epochs":
            best_epochs,
        "final_epoch_policy":
            "median_of_five_inner_best_epochs",
        "final_epochs":
            final_epochs,
        "outer_test_evaluated":
            False,
        "provenance": {
            "selected_csv":
                str(selected_csv),
            "selected_csv_sha256":
                sha256_file(
                    selected_csv
                ),
            "spec":
                str(spec_path),
            "spec_sha256":
                sha256_file(
                    spec_path
                ),
            "trainer":
                str(trainer),
            "trainer_sha256":
                sha256_file(
                    trainer
                ),
        },
    }

    summary_path = (
        output
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = (
        output
        / "inner_epoch_results.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        fieldnames = [
            "fold_id",
            "best_epoch",
            "best_valid_loss",
            "epochs_executed",
            "runtime_seconds",
            "outer_test_used",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    with (
        output
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in [
            summary_path,
            csv_path,
        ]:
            handle.write(
                sha256_file(path)
                + "  "
                + path.name
                + "\n"
            )

    temp.rename(
        output
        / "fold_runs"
    )

    print(
        "========================================"
    )
    print(
        "FINAL EPOCH DERIVATION"
    )
    print(
        "========================================"
    )

    print(
        "outer_fold:",
        outer,
    )

    print(
        "candidate:",
        candidate,
    )

    print(
        "selected_fitness:",
        selected_fitness,
    )

    print(
        "rerun_fitness:",
        rerun_fitness,
    )

    print(
        "fitness_drift:",
        drift,
    )

    print(
        "best_epochs:",
        best_epochs,
    )

    print(
        "final_epochs:",
        final_epochs,
    )

    print(
        "final_epoch_policy:",
        "median_of_five_inner_best_epochs",
    )

    print(
        "outer_test_evaluated: NO"
    )

    print(
        "FINAL_EPOCH_DERIVATION: PASS"
    )


if __name__ == "__main__":
    main()
