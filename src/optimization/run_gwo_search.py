#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from gwo_space import (
    candidate_key,
    decode_position,
    gru_parameter_count,
    load_spec,
    validate_candidate,
    trainer_arguments,
)


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(path: Path, obj):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    tmp.replace(path)


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


def clip01(value: float) -> float:
    return min(
        1.0,
        max(
            0.0,
            float(value),
        ),
    )


def features_for_method(
    spec,
    method: str,
) -> set[str]:
    methods = spec[
        "search_budget"
    ]["methods"]

    if method not in methods:
        raise ValueError(
            f"Unknown method: {method}"
        )

    return set(
        methods[method]
    )


def rank_key(record):
    return (
        float(record["fitness"]),
        int(record["parameter_count"]),
        str(record["candidate_key"]),
    )


def population_diversity(
    positions,
) -> float:
    n = len(positions)

    if n <= 1:
        return 0.0

    d = len(
        positions[0]
    )

    stds = []

    for j in range(d):
        values = [
            float(row[j])
            for row in positions
        ]

        mean = sum(values) / n

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / n

        stds.append(
            math.sqrt(variance)
        )

    raw = sum(stds) / d

    return clip01(
        raw / 0.5
    )


def progress_value(
    generation_index: int,
    n_generations: int,
) -> float:
    if n_generations <= 1:
        return 1.0

    return (
        generation_index
        / (n_generations - 1)
    )


def control_parameter_a(
    *,
    generation_index,
    n_generations,
    positions,
    use_m2,
    algorithm_spec,
):
    progress = progress_value(
        generation_index,
        n_generations,
    )

    a_linear = (
        2.0
        * (1.0 - progress)
    )

    diversity = population_diversity(
        positions
    )

    if use_m2:
        boost = float(
            algorithm_spec[
                "M2"
            ][
                "boost_coefficient"
            ]
        )

        a = min(
            2.0,
            max(
                0.0,
                a_linear
                * (
                    1.0
                    + boost
                    * (
                        1.0
                        - diversity
                    )
                ),
            ),
        )
    else:
        a = a_linear

    return {
        "progress": progress,
        "a_linear": a_linear,
        "a": a,
        "diversity": diversity,
    }


def uniform_initial_positions(
    rng: random.Random,
    population_size: int,
    dimension: int,
):
    return [
        [
            rng.random()
            for _ in range(
                dimension
            )
        ]
        for _ in range(
            population_size
        )
    ]


def chaotic_opposition_positions(
    rng: random.Random,
    *,
    population_size,
    dimension,
    algorithm_spec,
):
    m1 = algorithm_spec["M1"]

    mu = float(
        m1["mu"]
    )

    burn_in = int(
        m1["burn_in"]
    )

    state = rng.uniform(
        0.11,
        0.89,
    )

    def step(value):
        value = (
            mu
            * value
            * (1.0 - value)
        )

        return min(
            1.0 - 1e-12,
            max(
                1e-12,
                value,
            ),
        )

    for _ in range(
        burn_in
    ):
        state = step(state)

    base = []

    for _ in range(
        population_size
    ):
        row = []

        for _ in range(
            dimension
        ):
            state = step(state)
            row.append(state)

        base.append(row)

    opposition = [
        [
            1.0 - value
            for value in row
        ]
        for row in base
    ]

    return base + opposition


def standard_gwo_update(
    current,
    leaders,
    a,
    rng,
):
    output = []

    for dimension_index in range(
        len(current)
    ):
        estimates = []

        x = float(
            current[
                dimension_index
            ]
        )

        for leader in leaders:
            leader_value = float(
                leader[
                    dimension_index
                ]
            )

            r1 = rng.random()
            r2 = rng.random()

            A = (
                2.0 * a * r1
                - a
            )

            C = 2.0 * r2

            D = abs(
                C * leader_value
                - x
            )

            estimates.append(
                leader_value
                - A * D
            )

        output.append(
            clip01(
                sum(estimates)
                / len(estimates)
            )
        )

    return output


def mantegna_sigma(
    beta: float,
) -> float:
    numerator = (
        math.gamma(
            1.0 + beta
        )
        * math.sin(
            math.pi
            * beta
            / 2.0
        )
    )

    denominator = (
        math.gamma(
            (1.0 + beta)
            / 2.0
        )
        * beta
        * (
            2.0
            ** (
                (beta - 1.0)
                / 2.0
            )
        )
    )

    return (
        numerator
        / denominator
    ) ** (
        1.0 / beta
    )


def apply_levy_mutation(
    position,
    *,
    progress,
    rng,
    algorithm_spec,
):
    m3 = algorithm_spec["M3"]

    probability = float(
        m3[
            "mutation_probability_per_wolf"
        ]
    )

    if rng.random() >= probability:
        return (
            list(position),
            False,
        )

    beta = float(
        m3["beta"]
    )

    sigma = mantegna_sigma(
        beta
    )

    start = float(
        m3["scale_start"]
    )

    end = float(
        m3["scale_end"]
    )

    scale = (
        start
        + progress
        * (
            end - start
        )
    )

    mutated = []

    for value in position:
        u = rng.gauss(
            0.0,
            sigma,
        )

        v = rng.gauss(
            0.0,
            1.0,
        )

        denominator = max(
            abs(v),
            1e-12,
        ) ** (
            1.0 / beta
        )

        step = (
            u
            / denominator
        )

        mutated.append(
            clip01(
                float(value)
                + scale * step
            )
        )

    return (
        mutated,
        True,
    )


class SyntheticEvaluator:
    def __init__(self):
        self.cache = {}

    def evaluate(
        self,
        candidate,
    ):
        key = candidate_key(
            candidate
        )

        if key in self.cache:
            record = dict(
                self.cache[key]
            )

            record[
                "cache_hit"
            ] = True

            return record

        hidden = (
            (
                candidate[
                    "hidden_size"
                ]
                - 32.0
            )
            / 80.0
        )

        layers = (
            candidate[
                "num_layers"
            ]
            - 2.0
        ) / 2.0

        dropout = (
            candidate["dropout"]
            - 0.2
        ) / 0.4

        lr = (
            math.log10(
                candidate[
                    "learning_rate"
                ]
            )
            - math.log10(
                0.001
            )
        ) / math.log10(
            50.0
        )

        wd = (
            math.log10(
                candidate[
                    "weight_decay"
                ]
            )
            - math.log10(
                0.0001
            )
        ) / 3.0

        fitness = float(
            hidden * hidden
            + layers * layers
            + dropout * dropout
            + lr * lr
            + wd * wd
        )

        record = {
            "fitness": fitness,
            "fold_losses": [
                fitness
            ] * 5,
            "parameter_count": (
                gru_parameter_count(
                    candidate[
                        "hidden_size"
                    ],
                    candidate[
                        "num_layers"
                    ],
                )
            ),
            "cache_hit": False,
        }

        self.cache[key] = dict(
            record
        )

        return record


class RealEvaluator:
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
        self.outer_fold = int(
            outer_fold
        )

        self.data_root = Path(
            data_root
        ).resolve()

        self.trainer = Path(
            trainer
        ).resolve()

        self.output_dir = Path(
            output_dir
        ).resolve()

        self.spec = spec

        self.workers = int(
            workers
        )

        self.cache_dir = (
            self.output_dir
            / "candidate_cache"
        )

        self.tmp_dir = (
            self.output_dir
            / "tmp"
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.tmp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not (
            1 <= self.outer_fold <= 11
        ):
            raise ValueError(
                "outer_fold must be 1..11"
            )

        if self.workers <= 0:
            raise ValueError(
                "workers must be positive"
            )

        if not self.trainer.is_file():
            raise FileNotFoundError(
                self.trainer
            )

        self.trainer_sha256 = (
            sha256_file(
                self.trainer
            )
        )

    def fold_ids(self):
        return [
            (
                f"fold_"
                f"{self.outer_fold:02d}"
                f"_inner_{inner:02d}"
            )
            for inner
            in range(1, 6)
        ]

    def _run_fold(
        self,
        fold_id,
        candidate,
        temp_root,
    ):
        fold_npz = (
            self.data_root
            / "folds"
            / f"{fold_id}.npz"
        )

        if not fold_npz.is_file():
            raise FileNotFoundError(
                fold_npz
            )

        fold_output = (
            temp_root
            / fold_id
        )

        command = [
            sys.executable,
            str(self.trainer),
            "--fold-npz",
            str(fold_npz),
            "--output-dir",
            str(fold_output),
        ] + trainer_arguments(
            candidate,
            self.spec,
        )

        env = os.environ.copy()

        for name in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]:
            env[name] = "1"

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
            diagnostic = (
                temp_root
                / (
                    fold_id
                    + "_FAILED.txt"
                )
            )

            diagnostic.write_text(
                "COMMAND:\n"
                + " ".join(command)
                + "\n\nSTDOUT:\n"
                + process.stdout
                + "\n\nSTDERR:\n"
                + process.stderr,
                encoding="utf-8",
            )

            raise RuntimeError(
                f"{fold_id} failed "
                f"with rc={process.returncode}; "
                f"see {diagnostic}"
            )

        metadata_path = (
            fold_output
            / "run_metadata.json"
        )

        metadata = read_json(
            metadata_path
        )

        if metadata[
            "leakage_policy"
        ][
            "outer_test_used"
        ] is not False:
            raise RuntimeError(
                f"{fold_id}: outer test used"
            )

        expected_parameters = (
            gru_parameter_count(
                candidate[
                    "hidden_size"
                ],
                candidate[
                    "num_layers"
                ],
            )
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
                f"{fold_id}: parameter "
                "count mismatch"
            )

        result = metadata[
            "result"
        ]

        return {
            "fold_id": fold_id,
            "fold_npz": str(
                fold_npz
            ),
            "fold_npz_sha256": (
                sha256_file(
                    fold_npz
                )
            ),
            "best_valid_loss": float(
                result[
                    "best_valid_loss"
                ]
            ),
            "best_epoch": int(
                result[
                    "best_epoch"
                ]
            ),
            "epochs_executed": int(
                result[
                    "epochs_executed"
                ]
            ),
            "runtime_seconds": runtime,
            "n_parameters": (
                actual_parameters
            ),
            "outer_test_used": False,
        }

    def evaluate(
        self,
        candidate,
    ):
        validate_candidate(
            candidate,
            self.spec,
        )

        key = candidate_key(
            candidate
        )

        digest = hashlib.sha256(
            key.encode("utf-8")
        ).hexdigest()

        cache_path = (
            self.cache_dir
            / (
                digest
                + ".json"
            )
        )

        if cache_path.is_file():
            record = read_json(
                cache_path
            )

            if (
                record[
                    "candidate_key"
                ]
                != key
            ):
                raise RuntimeError(
                    "Candidate cache key mismatch."
                )

            record[
                "cache_hit"
            ] = True

            return record

        temp_root = Path(
            tempfile.mkdtemp(
                prefix=(
                    digest[:12]
                    + "_"
                ),
                dir=self.tmp_dir,
            )
        )

        fold_ids = self.fold_ids()

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(
                    self.workers,
                    5,
                )
            ) as pool:

                futures = {
                    pool.submit(
                        self._run_fold,
                        fold_id,
                        candidate,
                        temp_root,
                    ): fold_id
                    for fold_id
                    in fold_ids
                }

                fold_records = []

                for future in (
                    concurrent.futures.as_completed(
                        futures
                    )
                ):
                    fold_records.append(
                        future.result()
                    )

            fold_records.sort(
                key=lambda item:
                item["fold_id"]
            )

            losses = [
                float(
                    item[
                        "best_valid_loss"
                    ]
                )
                for item
                in fold_records
            ]

            fitness = float(
                sum(losses)
                / len(losses)
            )

            record = {
                "candidate": candidate,
                "candidate_key": key,
                "candidate_hash": digest,
                "fitness": fitness,
                "fold_losses": losses,
                "fold_records": fold_records,
                "parameter_count": (
                    gru_parameter_count(
                        candidate[
                            "hidden_size"
                        ],
                        candidate[
                            "num_layers"
                        ],
                    )
                ),
                "trainer": str(
                    self.trainer
                ),
                "trainer_sha256": (
                    self.trainer_sha256
                ),
                "outer_fold": (
                    self.outer_fold
                ),
                "outer_test_used": False,
                "cache_hit": False,
            }

            write_json(
                cache_path,
                record,
            )

        except Exception:
            raise

        else:
            shutil.rmtree(
                temp_root
            )

        return record


def evaluate_positions(
    *,
    positions,
    evaluator,
    spec,
    phase,
    generation,
    starting_index,
):
    records = []

    evaluation_index = int(
        starting_index
    )

    for position in positions:
        candidate = decode_position(
            position,
            spec,
        )

        result = evaluator.evaluate(
            candidate
        )

        record = {
            "evaluation_index": (
                evaluation_index
            ),
            "phase": phase,
            "generation": generation,
            "position": [
                float(value)
                for value
                in position
            ],
            "candidate": candidate,
            "candidate_key": (
                candidate_key(
                    candidate
                )
            ),
            "fitness": float(
                result["fitness"]
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

        records.append(record)

        evaluation_index += 1

    return records


def run_search_core(
    *,
    method,
    optimizer_seed,
    spec,
    algorithm_spec,
    evaluator,
):
    features = features_for_method(
        spec,
        method,
    )

    rng = random.Random(
        int(optimizer_seed)
    )

    population_size = int(
        spec[
            "search_budget"
        ][
            "population_size"
        ]
    )

    dimension = int(
        spec[
            "position_dimension"
        ]
    )

    if "M1" in features:
        positions = (
            chaotic_opposition_positions(
                rng,
                population_size=(
                    population_size
                ),
                dimension=dimension,
                algorithm_spec=(
                    algorithm_spec
                ),
            )
        )

        expected_initial = int(
            spec[
                "search_budget"
            ][
                "m1_accounting"
            ][
                "initial_candidate_evaluations"
            ]
        )

        update_generations = int(
            spec[
                "search_budget"
            ][
                "m1_accounting"
            ][
                "update_generations"
            ]
        )
    else:
        positions = (
            uniform_initial_positions(
                rng,
                population_size,
                dimension,
            )
        )

        expected_initial = int(
            spec[
                "search_budget"
            ][
                "standard_or_no_m1_accounting"
            ][
                "initial_population_evaluations"
            ]
        )

        update_generations = int(
            spec[
                "search_budget"
            ][
                "standard_or_no_m1_accounting"
            ][
                "update_generations"
            ]
        )

    if len(positions) != expected_initial:
        raise RuntimeError(
            "Initial evaluation count mismatch."
        )

    all_records = evaluate_positions(
        positions=positions,
        evaluator=evaluator,
        spec=spec,
        phase="initialization",
        generation=0,
        starting_index=1,
    )

    ranked_initial = sorted(
        all_records,
        key=rank_key,
    )

    population = [
        {
            "position": list(
                record[
                    "position"
                ]
            ),
            "record": record,
        }
        for record in ranked_initial[
            :population_size
        ]
    ]

    telemetry = []

    mutation_count_total = 0

    for generation_index in range(
        update_generations
    ):
        ranked = sorted(
            population,
            key=lambda item:
            rank_key(
                item["record"]
            ),
        )

        leaders = [
            list(
                ranked[index][
                    "position"
                ]
            )
            for index
            in range(3)
        ]

        current_positions = [
            list(
                item["position"]
            )
            for item
            in population
        ]

        control = (
            control_parameter_a(
                generation_index=(
                    generation_index
                ),
                n_generations=(
                    update_generations
                ),
                positions=(
                    current_positions
                ),
                use_m2=(
                    "M2" in features
                ),
                algorithm_spec=(
                    algorithm_spec
                ),
            )
        )

        next_positions = []

        mutation_count = 0

        for item in population:
            updated = standard_gwo_update(
                item["position"],
                leaders,
                control["a"],
                rng,
            )

            if "M3" in features:
                (
                    updated,
                    mutated,
                ) = apply_levy_mutation(
                    updated,
                    progress=(
                        control[
                            "progress"
                        ]
                    ),
                    rng=rng,
                    algorithm_spec=(
                        algorithm_spec
                    ),
                )

                mutation_count += int(
                    mutated
                )

            next_positions.append(
                updated
            )

        mutation_count_total += (
            mutation_count
        )

        starting_index = (
            len(all_records)
            + 1
        )

        generation_records = (
            evaluate_positions(
                positions=next_positions,
                evaluator=evaluator,
                spec=spec,
                phase="update",
                generation=(
                    generation_index + 1
                ),
                starting_index=(
                    starting_index
                ),
            )
        )

        all_records.extend(
            generation_records
        )

        population = [
            {
                "position": list(
                    record[
                        "position"
                    ]
                ),
                "record": record,
            }
            for record
            in generation_records
        ]

        best_so_far = min(
            all_records,
            key=rank_key,
        )

        telemetry.append(
            {
                "generation": (
                    generation_index + 1
                ),
                "progress": (
                    control["progress"]
                ),
                "a_linear": (
                    control["a_linear"]
                ),
                "a": control["a"],
                "diversity": (
                    control["diversity"]
                ),
                "mutation_count": (
                    mutation_count
                ),
                "best_fitness_so_far": (
                    best_so_far[
                        "fitness"
                    ]
                ),
            }
        )

    expected_budget = int(
        spec[
            "search_budget"
        ][
            "fitness_evaluation_budget_per_search"
        ]
    )

    if (
        len(all_records)
        != expected_budget
    ):
        raise RuntimeError(
            f"Expected {expected_budget} "
            f"evaluations, got "
            f"{len(all_records)}"
        )

    best = min(
        all_records,
        key=rank_key,
    )

    return {
        "method": method,
        "features": sorted(
            features
        ),
        "optimizer_seed": int(
            optimizer_seed
        ),
        "evaluation_budget": (
            expected_budget
        ),
        "evaluation_count": (
            len(all_records)
        ),
        "cache_hit_count": sum(
            int(
                record[
                    "cache_hit"
                ]
            )
            for record
            in all_records
        ),
        "initial_evaluation_count": (
            expected_initial
        ),
        "update_generations": (
            update_generations
        ),
        "mutation_count_total": (
            mutation_count_total
        ),
        "best": best,
        "telemetry": telemetry,
        "evaluations": all_records,
    }


def self_test(
    spec,
    algorithm_spec,
):
    methods = list(
        spec[
            "search_budget"
        ][
            "methods"
        ].keys()
    )

    seed = int(
        spec[
            "search_budget"
        ][
            "optimizer_seeds"
        ][0]
    )

    failures = []

    def check(
        condition,
        label,
    ):
        print(
            label + ":",
            "PASS"
            if condition
            else "FAIL",
        )

        if not condition:
            failures.append(
                label
            )

    for method in methods:
        first = run_search_core(
            method=method,
            optimizer_seed=seed,
            spec=spec,
            algorithm_spec=(
                algorithm_spec
            ),
            evaluator=(
                SyntheticEvaluator()
            ),
        )

        second = run_search_core(
            method=method,
            optimizer_seed=seed,
            spec=spec,
            algorithm_spec=(
                algorithm_spec
            ),
            evaluator=(
                SyntheticEvaluator()
            ),
        )

        print()
        print(
            "METHOD:",
            method,
        )

        print(
            "features:",
            first["features"],
        )

        print(
            "best_fitness:",
            first["best"][
                "fitness"
            ],
        )

        print(
            "initial_evaluations:",
            first[
                "initial_evaluation_count"
            ],
        )

        print(
            "update_generations:",
            first[
                "update_generations"
            ],
        )

        print(
            "mutations:",
            first[
                "mutation_count_total"
            ],
        )

        check(
            first[
                "evaluation_count"
            ] == 30,
            method
            + "_BUDGET_30",
        )

        check(
            json.dumps(
                first,
                sort_keys=True,
            )
            == json.dumps(
                second,
                sort_keys=True,
            ),
            method
            + "_DETERMINISTIC",
        )

        has_m1 = (
            "M1"
            in first[
                "features"
            ]
        )

        check(
            first[
                "initial_evaluation_count"
            ]
            == (
                10
                if has_m1
                else 5
            ),
            method
            + "_M1_ACCOUNTING",
        )

        has_m2 = (
            "M2"
            in first[
                "features"
            ]
        )

        if has_m2:
            changed = any(
                abs(
                    row["a"]
                    - row[
                        "a_linear"
                    ]
                )
                > 1e-12
                for row
                in first[
                    "telemetry"
                ]
            )

            check(
                changed,
                method
                + "_M2_ACTIVE",
            )
        else:
            unchanged = all(
                abs(
                    row["a"]
                    - row[
                        "a_linear"
                    ]
                )
                <= 1e-12
                for row
                in first[
                    "telemetry"
                ]
            )

            check(
                unchanged,
                method
                + "_M2_DISABLED",
            )

        has_m3 = (
            "M3"
            in first[
                "features"
            ]
        )

        if has_m3:
            check(
                first[
                    "mutation_count_total"
                ] > 0,
                method
                + "_M3_ACTIVE",
            )
        else:
            check(
                first[
                    "mutation_count_total"
                ] == 0,
                method
                + "_M3_DISABLED",
            )

    check(
        algorithm_spec[
            "M1"
        ]["mu"] == 4.0,
        "M1_LOGISTIC_MU_4",
    )

    check(
        algorithm_spec[
            "M2"
        ][
            "boost_coefficient"
        ] == 0.5,
        "M2_BOOST_0_5",
    )

    check(
        algorithm_spec[
            "M3"
        ]["beta"] == 1.5,
        "M3_BETA_1_5",
    )

    check(
        algorithm_spec[
            "M3"
        ][
            "mutation_probability_per_wolf"
        ] == 0.2,
        "M3_PROBABILITY_0_2",
    )

    if failures:
        print()
        print(
            "GWO_ENGINE_SELF_TEST: FAIL"
        )

        raise SystemExit(1)

    print()
    print(
        "GWO_ENGINE_SELF_TEST: PASS"
    )


def write_search_outputs(
    output_dir,
    result,
    run_config,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_json(
        output_dir
        / "run_config.json",
        run_config,
    )

    summary = {
        key: value
        for key, value
        in result.items()
        if key
        not in {
            "evaluations",
            "telemetry",
        }
    }

    write_json(
        output_dir
        / "search_summary.json",
        summary,
    )

    write_json(
        output_dir
        / "telemetry.json",
        result[
            "telemetry"
        ],
    )

    trace_path = (
        output_dir
        / "evaluation_trace.jsonl"
    )

    with trace_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in result[
            "evaluations"
        ]:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
                + "\n"
            )

    files = [
        "run_config.json",
        "search_summary.json",
        "telemetry.json",
        "evaluation_trace.jsonl",
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--spec",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--algorithm-config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--evaluate-baseline",
        action="store_true",
    )

    parser.add_argument(
        "--outer-fold",
        type=int,
    )

    parser.add_argument(
        "--method",
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

    args = parser.parse_args()

    spec = load_spec(
        args.spec
    )

    algorithm_spec = read_json(
        args.algorithm_config
    )

    if args.self_test:
        self_test(
            spec,
            algorithm_spec,
        )

        if not (
            args.evaluate_baseline
            or args.method
        ):
            return

    if args.evaluate_baseline:
        required = [
            args.outer_fold,
            args.data_root,
            args.trainer,
            args.output_dir,
        ]

        if any(
            value is None
            for value in required
        ):
            raise RuntimeError(
                "Real baseline evaluation "
                "requires outer-fold, data-root, "
                "trainer and output-dir."
            )

        output_dir = (
            args.output_dir
            .expanduser()
            .resolve()
        )

        if output_dir.exists():
            raise RuntimeError(
                f"Output exists: {output_dir}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        evaluator = RealEvaluator(
            outer_fold=(
                args.outer_fold
            ),
            data_root=(
                args.data_root
            ),
            trainer=args.trainer,
            output_dir=(
                output_dir
            ),
            spec=spec,
            workers=args.workers,
        )

        candidate = dict(
            spec[
                "baseline_reference"
            ]
        )

        candidate.pop(
            "parameter_count",
            None,
        )

        validate_candidate(
            candidate,
            spec,
        )

        result = evaluator.evaluate(
            candidate
        )

        write_json(
            output_dir
            / "single_evaluation.json",
            result,
        )

        print(
            "outer_fold:",
            args.outer_fold,
        )

        print(
            "candidate:",
            candidate,
        )

        print(
            "parameter_count:",
            result[
                "parameter_count"
            ],
        )

        print(
            "fold_losses:",
            result[
                "fold_losses"
            ],
        )

        print(
            "fitness:",
            result[
                "fitness"
            ],
        )

        print(
            "outer_test_used:",
            result[
                "outer_test_used"
            ],
        )

        print(
            "REAL_FIVE_FOLD_FITNESS: PASS"
        )

        return

    if args.method is None:
        return

    required = [
        args.outer_fold,
        args.optimizer_seed,
        args.data_root,
        args.trainer,
        args.output_dir,
    ]

    if any(
        value is None
        for value in required
    ):
        raise RuntimeError(
            "Full search requires outer-fold, "
            "method, optimizer-seed, data-root, "
            "trainer and output-dir."
        )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if output_dir.exists():
        raise RuntimeError(
            f"Output exists: {output_dir}"
        )

    evaluator_root = (
        output_dir.parent
        / (
            output_dir.name
            + ".work"
        )
    )

    if evaluator_root.exists():
        raise RuntimeError(
            f"Work output exists: "
            f"{evaluator_root}"
        )

    evaluator_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    evaluator = RealEvaluator(
        outer_fold=args.outer_fold,
        data_root=args.data_root,
        trainer=args.trainer,
        output_dir=evaluator_root,
        spec=spec,
        workers=args.workers,
    )

    result = run_search_core(
        method=args.method,
        optimizer_seed=(
            args.optimizer_seed
        ),
        spec=spec,
        algorithm_spec=(
            algorithm_spec
        ),
        evaluator=evaluator,
    )

    run_config = {
        "outer_fold": (
            args.outer_fold
        ),
        "method": args.method,
        "optimizer_seed": (
            args.optimizer_seed
        ),
        "spec": str(
            args.spec.resolve()
        ),
        "spec_sha256": (
            sha256_file(
                args.spec
            )
        ),
        "algorithm_config": str(
            args.algorithm_config.resolve()
        ),
        "algorithm_config_sha256": (
            sha256_file(
                args.algorithm_config
            )
        ),
        "trainer": str(
            args.trainer.resolve()
        ),
        "trainer_sha256": (
            sha256_file(
                args.trainer
            )
        ),
        "data_root": str(
            args.data_root.resolve()
        ),
        "workers": args.workers,
        "outer_test_evaluated": False,
    }

    write_search_outputs(
        output_dir,
        result,
        run_config,
    )

    shutil.rmtree(
        evaluator_root
    )

    print(
        "method:",
        args.method,
    )

    print(
        "outer_fold:",
        args.outer_fold,
    )

    print(
        "optimizer_seed:",
        args.optimizer_seed,
    )

    print(
        "evaluations:",
        result[
            "evaluation_count"
        ],
    )

    print(
        "cache_hits:",
        result[
            "cache_hit_count"
        ],
    )

    print(
        "best_fitness:",
        result[
            "best"
        ][
            "fitness"
        ],
    )

    print(
        "best_candidate:",
        result[
            "best"
        ][
            "candidate"
        ],
    )

    print(
        "outer_test_evaluated: NO"
    )

    print(
        "GWO_SEARCH: PASS"
    )


if __name__ == "__main__":
    main()
