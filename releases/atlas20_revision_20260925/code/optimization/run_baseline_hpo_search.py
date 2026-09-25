#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr

import run_gwo_search as base
import run_gwo_search_robust as robust


METHODS = (
    "random_search",
    "bayesian_optimization",
)

BAYES_INITIAL_RANDOM = 5
BAYES_POOL_SIZE = 8192
BAYES_XI = 0.01
BAYES_LENGTH_SCALES = (
    0.10,
    0.20,
    0.35,
    0.50,
    0.80,
    1.20,
)
GP_NOISE = 1e-6


class Atlas20RobustEvaluator(
    robust.RobustRealEvaluator
):
    """
    The frozen ATLAS11 evaluator validates outer_fold <= 11 in __init__.
    All subsequent logic is generic and obtains fold names from
    self.outer_fold. Initialize through a valid legacy value, then restore
    the requested ATLAS20 fold after independently validating 1..20.
    """

    def __init__(
        self,
        *,
        outer_fold,
        data_root,
        trainer,
        output_dir,
        spec,
        workers,
    ):
        actual_outer = int(
            outer_fold
        )

        if not (
            1 <= actual_outer <= 20
        ):
            raise ValueError(
                "outer_fold must be 1..20"
            )

        super().__init__(
            outer_fold=min(
                actual_outer,
                11,
            ),
            data_root=data_root,
            trainer=trainer,
            output_dir=output_dir,
            spec=spec,
            workers=workers,
        )

        self.outer_fold = (
            actual_outer
        )


def sha256_file(
    path: Path,
) -> str:
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


def write_json(
    path: Path,
    obj,
):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def make_record(
    *,
    position,
    result,
    evaluation_index,
    phase,
    generation,
):
    candidate = result[
        "candidate"
    ] if "candidate" in result else None

    if candidate is None:
        raise RuntimeError(
            "Evaluator result lacks candidate."
        )

    return {
        "evaluation_index": int(
            evaluation_index
        ),
        "phase": str(
            phase
        ),
        "generation": int(
            generation
        ),
        "position": [
            float(value)
            for value in position
        ],
        "candidate": candidate,
        "candidate_key": str(
            result[
                "candidate_key"
            ]
        ),
        "fitness": float(
            result[
                "fitness"
            ]
        ),
        "parameter_count": int(
            result[
                "parameter_count"
            ]
        ),
        "cache_hit": bool(
            result.get(
                "cache_hit",
                False,
            )
        ),
        "fold_losses": [
            float(value)
            for value
            in result[
                "fold_losses"
            ]
        ],
    }


def evaluate_position(
    *,
    position,
    evaluator,
    spec,
    evaluation_index,
    phase,
    generation,
):
    candidate = base.decode_position(
        position,
        spec,
    )

    result = evaluator.evaluate(
        candidate
    )

    return make_record(
        position=position,
        result=result,
        evaluation_index=(
            evaluation_index
        ),
        phase=phase,
        generation=generation,
    )


def random_position(
    rng: random.Random,
    dimension: int,
):
    return [
        rng.random()
        for _ in range(
            dimension
        )
    ]


def run_random_search(
    *,
    optimizer_seed,
    spec,
    evaluator,
    budget,
):
    rng = random.Random(
        int(
            optimizer_seed
        )
    )

    dimension = int(
        spec[
            "position_dimension"
        ]
    )

    records = []

    for index in range(
        1,
        int(budget) + 1,
    ):
        position = random_position(
            rng,
            dimension,
        )

        records.append(
            evaluate_position(
                position=position,
                evaluator=evaluator,
                spec=spec,
                evaluation_index=index,
                phase="random",
                generation=index,
            )
        )

    return (
        records,
        [],
    )


def matern52_kernel(
    xa: np.ndarray,
    xb: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    difference = (
        xa[:, None, :]
        - xb[None, :, :]
    )

    distance = np.sqrt(
        np.sum(
            difference
            * difference,
            axis=2,
        )
    )

    scaled = (
        math.sqrt(5.0)
        * distance
        / float(
            length_scale
        )
    )

    return (
        1.0
        + scaled
        + (
            scaled
            * scaled
            / 3.0
        )
    ) * np.exp(
        -scaled
    )


def fit_gp(
    x: np.ndarray,
    y: np.ndarray,
):
    y_mean = float(
        np.mean(y)
    )

    y_std = float(
        np.std(y)
    )

    if not math.isfinite(
        y_std
    ) or y_std < 1e-12:
        y_std = 1.0

    ys = (
        y - y_mean
    ) / y_std

    best = None

    for length_scale in (
        BAYES_LENGTH_SCALES
    ):
        kernel = matern52_kernel(
            x,
            x,
            length_scale,
        )

        kernel = (
            kernel
            + GP_NOISE
            * np.eye(
                len(x),
                dtype=np.float64,
            )
        )

        try:
            factor = cho_factor(
                kernel,
                lower=True,
                check_finite=True,
            )

            alpha = cho_solve(
                factor,
                ys,
                check_finite=True,
            )

        except Exception:
            continue

        log_det_half = float(
            np.sum(
                np.log(
                    np.diag(
                        factor[0]
                    )
                )
            )
        )

        lml = (
            -0.5
            * float(
                ys @ alpha
            )
            - log_det_half
            - 0.5
            * len(x)
            * math.log(
                2.0
                * math.pi
            )
        )

        candidate = {
            "length_scale": float(
                length_scale
            ),
            "factor": factor,
            "alpha": alpha,
            "lml": float(
                lml
            ),
            "y_mean": y_mean,
            "y_std": y_std,
            "ys": ys,
        }

        if (
            best is None
            or candidate["lml"]
            > best["lml"]
        ):
            best = candidate

    if best is None:
        raise RuntimeError(
            "Gaussian-process fit failed."
        )

    return best


def expected_improvement(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_candidates: np.ndarray,
):
    model = fit_gp(
        x_train,
        y_train,
    )

    cross = matern52_kernel(
        x_candidates,
        x_train,
        model[
            "length_scale"
        ],
    )

    mean_scaled = (
        cross
        @ model[
            "alpha"
        ]
    )

    solved = cho_solve(
        model[
            "factor"
        ],
        cross.T,
        check_finite=True,
    )

    variance = (
        1.0
        - np.sum(
            cross
            * solved.T,
            axis=1,
        )
    )

    variance = np.maximum(
        variance,
        1e-12,
    )

    sigma = np.sqrt(
        variance
    )

    best_scaled = float(
        np.min(
            model[
                "ys"
            ]
        )
    )

    improvement = (
        best_scaled
        - mean_scaled
        - BAYES_XI
    )

    z = (
        improvement
        / sigma
    )

    density = (
        np.exp(
            -0.5
            * z
            * z
        )
        / math.sqrt(
            2.0
            * math.pi
        )
    )

    ei = (
        improvement
        * ndtr(
            z
        )
        + sigma
        * density
    )

    return (
        ei,
        model,
    )


def unique_gp_rows(
    records,
):
    seen = set()
    positions = []
    fitness = []

    for record in records:
        key = record[
            "candidate_key"
        ]

        if key in seen:
            continue

        seen.add(
            key
        )

        positions.append(
            record[
                "position"
            ]
        )

        fitness.append(
            record[
                "fitness"
            ]
        )

    return (
        np.asarray(
            positions,
            dtype=np.float64,
        ),
        np.asarray(
            fitness,
            dtype=np.float64,
        ),
        seen,
    )


def bayes_candidate_pool(
    *,
    optimizer_seed,
    evaluation_index,
    dimension,
):
    sequence = np.random.SeedSequence(
        [
            int(
                optimizer_seed
            ),
            int(
                evaluation_index
            ),
            8675309,
        ]
    )

    rng = np.random.default_rng(
        sequence
    )

    return rng.random(
        (
            BAYES_POOL_SIZE,
            int(
                dimension
            ),
        ),
        dtype=np.float64,
    )


def choose_bayesian_position(
    *,
    records,
    optimizer_seed,
    evaluation_index,
    spec,
):
    (
        x_train,
        y_train,
        seen,
    ) = unique_gp_rows(
        records
    )

    dimension = int(
        spec[
            "position_dimension"
        ]
    )

    pool = bayes_candidate_pool(
        optimizer_seed=optimizer_seed,
        evaluation_index=(
            evaluation_index
        ),
        dimension=dimension,
    )

    keep_positions = []
    keep_keys = set()

    for row in pool:
        position = [
            float(value)
            for value in row
        ]

        candidate = base.decode_position(
            position,
            spec,
        )

        key = base.candidate_key(
            candidate
        )

        if (
            key in seen
            or key in keep_keys
        ):
            continue

        keep_keys.add(
            key
        )

        keep_positions.append(
            position
        )

    if not keep_positions:
        raise RuntimeError(
            "Bayesian candidate pool is empty."
        )

    candidates = np.asarray(
        keep_positions,
        dtype=np.float64,
    )

    if len(x_train) < 2:
        return (
            keep_positions[0],
            {
                "strategy": (
                    "fallback_first_unseen"
                ),
                "pool_size": int(
                    len(candidates)
                ),
            },
        )

    ei, model = expected_improvement(
        x_train=x_train,
        y_train=y_train,
        x_candidates=candidates,
    )

    best_index = int(
        np.argmax(
            ei
        )
    )

    return (
        keep_positions[
            best_index
        ],
        {
            "strategy": (
                "gp_matern52_expected_improvement"
            ),
            "pool_size": int(
                len(candidates)
            ),
            "length_scale": float(
                model[
                    "length_scale"
                ]
            ),
            "log_marginal_likelihood": float(
                model[
                    "lml"
                ]
            ),
            "expected_improvement": float(
                ei[
                    best_index
                ]
            ),
            "n_unique_observations": int(
                len(x_train)
            ),
        },
    )


def run_bayesian_search(
    *,
    optimizer_seed,
    spec,
    evaluator,
    budget,
):
    rng = random.Random(
        int(
            optimizer_seed
        )
    )

    dimension = int(
        spec[
            "position_dimension"
        ]
    )

    records = []
    telemetry = []

    initial = min(
        BAYES_INITIAL_RANDOM,
        int(
            budget
        ),
    )

    for index in range(
        1,
        initial + 1,
    ):
        position = random_position(
            rng,
            dimension,
        )

        records.append(
            evaluate_position(
                position=position,
                evaluator=evaluator,
                spec=spec,
                evaluation_index=index,
                phase=(
                    "bayes_initial_random"
                ),
                generation=0,
            )
        )

    for index in range(
        initial + 1,
        int(
            budget
        )
        + 1,
    ):
        (
            position,
            acquisition,
        ) = choose_bayesian_position(
            records=records,
            optimizer_seed=(
                optimizer_seed
            ),
            evaluation_index=index,
            spec=spec,
        )

        record = evaluate_position(
            position=position,
            evaluator=evaluator,
            spec=spec,
            evaluation_index=index,
            phase=(
                "bayes_expected_improvement"
            ),
            generation=(
                index
                - initial
            ),
        )

        records.append(
            record
        )

        acquisition[
            "evaluation_index"
        ] = int(
            index
        )

        acquisition[
            "selected_candidate_key"
        ] = record[
            "candidate_key"
        ]

        telemetry.append(
            acquisition
        )

    return (
        records,
        telemetry,
    )


def run_method(
    *,
    method,
    optimizer_seed,
    spec,
    evaluator,
    budget,
):
    if method == "random_search":
        return run_random_search(
            optimizer_seed=(
                optimizer_seed
            ),
            spec=spec,
            evaluator=evaluator,
            budget=budget,
        )

    if method == (
        "bayesian_optimization"
    ):
        return run_bayesian_search(
            optimizer_seed=(
                optimizer_seed
            ),
            spec=spec,
            evaluator=evaluator,
            budget=budget,
        )

    raise ValueError(
        "Unknown method: "
        + str(
            method
        )
    )


def summary_from_records(
    *,
    method,
    optimizer_seed,
    records,
    budget,
):
    if len(records) != int(
        budget
    ):
        raise RuntimeError(
            "Evaluation budget mismatch."
        )

    best = min(
        records,
        key=base.rank_key,
    )

    return {
        "method": method,
        "optimizer_seed": int(
            optimizer_seed
        ),
        "evaluation_count": int(
            len(records)
        ),
        "evaluation_budget": int(
            budget
        ),
        "unique_candidate_count": int(
            len(
                {
                    record[
                        "candidate_key"
                    ]
                    for record
                    in records
                }
            )
        ),
        "cache_hit_count": int(
            sum(
                bool(
                    record[
                        "cache_hit"
                    ]
                )
                for record
                in records
            )
        ),
        "best": {
            "evaluation_index": int(
                best[
                    "evaluation_index"
                ]
            ),
            "fitness": float(
                best[
                    "fitness"
                ]
            ),
            "parameter_count": int(
                best[
                    "parameter_count"
                ]
            ),
            "candidate_key": str(
                best[
                    "candidate_key"
                ]
            ),
            "candidate": best[
                "candidate"
            ],
            "fold_losses": [
                float(
                    value
                )
                for value
                in best[
                    "fold_losses"
                ]
            ],
        },
    }


def self_test(
    spec,
):
    for method in METHODS:
        evaluator = (
            base.SyntheticEvaluator()
        )

        original_evaluate = (
            evaluator.evaluate
        )

        def evaluate_with_candidate(
            candidate,
            _original=original_evaluate,
        ):
            result = _original(
                candidate
            )

            result[
                "candidate"
            ] = candidate

            result[
                "candidate_key"
            ] = base.candidate_key(
                candidate
            )

            return result

        evaluator.evaluate = (
            evaluate_with_candidate
        )

        records, _ = run_method(
            method=method,
            optimizer_seed=20260811,
            spec=spec,
            evaluator=evaluator,
            budget=10,
        )

        summary = (
            summary_from_records(
                method=method,
                optimizer_seed=(
                    20260811
                ),
                records=records,
                budget=10,
            )
        )

        if (
            summary[
                "evaluation_count"
            ]
            != 10
        ):
            raise RuntimeError(
                method
                + " self-test budget mismatch"
            )

        if len(
            summary[
                "best"
            ][
                "fold_losses"
            ]
        ) != 5:
            raise RuntimeError(
                method
                + " self-test fold mismatch"
            )

        print(
            method
            + ": SELF_TEST=PASS"
        )

    print(
        "HPO_ENGINE_SELF_TEST=PASS"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--spec",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--outer-fold",
        type=int,
    )

    parser.add_argument(
        "--method",
        choices=METHODS,
    )

    parser.add_argument(
        "--optimizer-seed",
        type=int,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
    )

    parser.add_argument(
        "--trainer",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--evaluation-budget",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    spec_path = (
        args.spec
        .expanduser()
        .resolve()
    )

    spec = base.load_spec(
        spec_path
    )

    if args.self_test:
        self_test(
            spec
        )
        return

    required = {
        "outer_fold":
            args.outer_fold,
        "method":
            args.method,
        "optimizer_seed":
            args.optimizer_seed,
        "data_root":
            args.data_root,
        "trainer":
            args.trainer,
        "output_dir":
            args.output_dir,
    }

    missing = [
        key
        for key, value
        in required.items()
        if value is None
    ]

    if missing:
        raise RuntimeError(
            "Missing arguments: "
            + ", ".join(
                missing
            )
        )

    if not (
        1 <= int(
            args.evaluation_budget
        ) <= 30
    ):
        raise ValueError(
            "evaluation-budget must be 1..30"
        )

    output = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    work = Path(
        str(output)
        + ".work"
    )

    if output.exists():
        raise RuntimeError(
            "Output exists: "
            + str(
                output
            )
        )

    if work.exists():
        raise RuntimeError(
            "Work directory exists: "
            + str(
                work
            )
        )

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    evaluator = (
        Atlas20RobustEvaluator(
            outer_fold=(
                args.outer_fold
            ),
            data_root=(
                args.data_root
            ),
            trainer=(
                args.trainer
            ),
            output_dir=work,
            spec=spec,
            workers=(
                args.workers
            ),
        )
    )

    try:
        records, telemetry = (
            run_method(
                method=args.method,
                optimizer_seed=(
                    args.optimizer_seed
                ),
                spec=spec,
                evaluator=evaluator,
                budget=(
                    args.evaluation_budget
                ),
            )
        )

        summary = (
            summary_from_records(
                method=args.method,
                optimizer_seed=(
                    args.optimizer_seed
                ),
                records=records,
                budget=(
                    args.evaluation_budget
                ),
            )
        )

        evaluation_path = (
            work
            / "evaluations.jsonl"
        )

        with evaluation_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                    )
                    + "\n"
                )

        telemetry_path = (
            work
            / "optimizer_telemetry.json"
        )

        write_json(
            telemetry_path,
            telemetry,
        )

        summary_path = (
            work
            / "search_summary.json"
        )

        write_json(
            summary_path,
            summary,
        )

        run_config = {
            "run_type":
                "paper_revision_hpo_baseline",
            "method":
                args.method,
            "outer_fold":
                int(
                    args.outer_fold
                ),
            "optimizer_seed":
                int(
                    args.optimizer_seed
                ),
            "evaluation_budget":
                int(
                    args.evaluation_budget
                ),
            "workers":
                int(
                    args.workers
                ),
            "data_root":
                str(
                    Path(
                        args.data_root
                    ).resolve()
                ),
            "trainer":
                str(
                    Path(
                        args.trainer
                    ).resolve()
                ),
            "trainer_sha256":
                sha256_file(
                    Path(
                        args.trainer
                    ).resolve()
                ),
            "spec":
                str(
                    spec_path
                ),
            "spec_sha256":
                sha256_file(
                    spec_path
                ),
            "fixed_trainer_seed":
                int(
                    spec[
                        "fixed_training"
                    ][
                        "trainer_seed"
                    ]
                ),
            "inner_folds_per_candidate":
                int(
                    spec[
                        "fitness"
                    ][
                        "inner_folds_per_candidate"
                    ]
                ),
            "objective":
                spec[
                    "fixed_training"
                ][
                    "objective"
                ],
            "outer_test_evaluated":
                False,
            "numerical_failure_policy": {
                "markers": list(
                    robust.NUMERICAL_MARKERS
                ),
                "failed_fold_penalty":
                    float(
                        robust.NUMERICAL_FAILURE_FOLD_LOSS
                    ),
                "non_numerical_failures":
                    "fatal",
            },
            "bayesian_optimization": {
                "surrogate":
                    "Gaussian process",
                "kernel":
                    "Matern-5/2 isotropic",
                "length_scale_selection":
                    "maximum log marginal likelihood over fixed grid",
                "length_scale_grid": list(
                    BAYES_LENGTH_SCALES
                ),
                "acquisition":
                    "expected improvement",
                "xi":
                    BAYES_XI,
                "initial_random_evaluations":
                    BAYES_INITIAL_RANDOM,
                "acquisition_candidate_pool":
                    BAYES_POOL_SIZE,
            },
        }

        run_config_path = (
            work
            / "run_config.json"
        )

        write_json(
            run_config_path,
            run_config,
        )

        files = [
            evaluation_path,
            telemetry_path,
            summary_path,
            run_config_path,
        ]

        checksum_path = (
            work
            / "SHA256SUMS.txt"
        )

        with checksum_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for path in files:
                handle.write(
                    sha256_file(
                        path
                    )
                    + "  "
                    + path.name
                    + "\n"
                )

        work.replace(
            output
        )

    except Exception:
        raise

    print(
        "=" * 80
    )
    print(
        "PAPER REVISION HPO SEARCH"
    )
    print(
        "=" * 80
    )
    print(
        "method="
        + args.method
    )
    print(
        "outer_fold="
        + str(
            args.outer_fold
        )
    )
    print(
        "optimizer_seed="
        + str(
            args.optimizer_seed
        )
    )
    print(
        "evaluation_budget="
        + str(
            args.evaluation_budget
        )
    )
    print(
        "best_fitness="
        + repr(
            summary[
                "best"
            ][
                "fitness"
            ]
        )
    )
    print(
        "outer_test_evaluated=NO"
    )
    print(
        "HPO_SEARCH=PASS"
    )


if __name__ == "__main__":
    main()
