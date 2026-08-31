#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = [
    "standard_gwo",
    "improved_gwo",
    "improved_no_m1",
    "improved_no_m2",
    "improved_no_m3",
]

SEEDS = [
    20260811,
    20260812,
    20260813,
]

EXPECTED_SPEC_SHA = (
    "18338c44d297a3f116998c06877e5f5ec"
    "fbb3aa2a04a44c5745e796f0f0473df"
)

EXPECTED_ALGO_SHA = (
    "3bdae410576c2dfbdeffdd4cb683790c8"
    "3e7ec8d5747678a33aeaf5460346076"
)

EXPECTED_TRAINER_SHA = (
    "1201be34061d2374d6490700ad7f36967"
    "e516e98df049fc25179e71847a137a4"
)


def sha256_file(path: Path) -> str:
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


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    value,
):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows,
    fieldnames,
):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def exact_sign_test(
    wins: int,
    losses: int,
) -> float:
    n = wins + losses

    if n == 0:
        return 1.0

    k = min(
        wins,
        losses,
    )

    numerator = sum(
        math.comb(
            n,
            i,
        )
        for i in range(
            k + 1
        )
    )

    tail = numerator / (
        2 ** n
    )

    return min(
        1.0,
        2.0 * tail,
    )


def holm_adjust(
    labels,
    pvalues,
):
    m = len(pvalues)

    order = sorted(
        range(m),
        key=lambda index:
        pvalues[index],
    )

    adjusted = [
        None
    ] * m

    running = 0.0

    for rank, index in enumerate(
        order
    ):
        multiplier = (
            m - rank
        )

        value = min(
            1.0,
            pvalues[index]
            * multiplier,
        )

        running = max(
            running,
            value,
        )

        adjusted[index] = running

    return {
        labels[index]:
        adjusted[index]
        for index
        in range(m)
    }


def sample_std(values):
    if len(values) < 2:
        return 0.0

    return statistics.stdev(
        values
    )


def verify_checksums(
    root: Path,
):
    checksum_file = (
        root
        / "SHA256SUMS.txt"
    )

    if not checksum_file.is_file():
        raise RuntimeError(
            f"Missing checksum file: {root}"
        )

    for line in checksum_file.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        expected, name = line.split(
            None,
            1,
        )

        name = name.strip()

        path = root / name

        if not path.is_file():
            raise RuntimeError(
                f"Missing checksum target: "
                f"{path}"
            )

        actual = sha256_file(
            path
        )

        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch: {path}"
            )


def candidate_rank_key(row):
    return (
        float(
            row["best_fitness"]
        ),
        int(
            row["parameter_count"]
        ),
        str(
            row["candidate_key"]
        ),
        int(
            row["optimizer_seed"]
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not run_root.is_dir():
        raise FileNotFoundError(
            run_root
        )

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: "
            f"{output_dir}"
        )

    manifest_path = (
        run_root
        / "provenance"
        / "task_manifest.tsv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )

    with manifest_path.open(
        encoding="utf-8",
        newline="",
    ) as handle:

        manifest = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    if len(manifest) != 165:
        raise RuntimeError(
            "Expected 165 manifest rows."
        )

    expected = set()

    for row in manifest:

        outer = int(
            row["outer_fold"]
        )

        method = row[
            "method"
        ]

        seed = int(
            row[
                "optimizer_seed"
            ]
        )

        expected.add(
            (
                outer,
                method,
                seed,
            )
        )

    if len(expected) != 165:
        raise RuntimeError(
            "Manifest combinations "
            "are not unique."
        )

    search_rows = []

    convergence_values = defaultdict(
        list
    )

    n_checksum_verified = 0
    n_fitness_verified = 0

    for outer, method, seed in sorted(
        expected
    ):

        if not (
            1 <= outer <= 11
        ):
            raise RuntimeError(
                "Invalid outer fold."
            )

        if method not in METHODS:
            raise RuntimeError(
                f"Unexpected method: {method}"
            )

        if seed not in SEEDS:
            raise RuntimeError(
                f"Unexpected seed: {seed}"
            )

        search_dir = (
            run_root
            / "searches"
            / f"fold_{outer:02d}"
            / method
            / f"seed_{seed}"
        )

        required = [
            "run_config.json",
            "search_summary.json",
            "telemetry.json",
            "evaluation_trace.jsonl",
            "SHA256SUMS.txt",
        ]

        for name in required:

            if not (
                search_dir
                / name
            ).is_file():

                raise RuntimeError(
                    f"Missing {name}: "
                    f"{search_dir}"
                )

        verify_checksums(
            search_dir
        )

        n_checksum_verified += 1

        config = read_json(
            search_dir
            / "run_config.json"
        )

        summary = read_json(
            search_dir
            / "search_summary.json"
        )

        trace = [
            json.loads(line)
            for line in (
                search_dir
                / "evaluation_trace.jsonl"
            ).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if config[
            "outer_fold"
        ] != outer:
            raise RuntimeError(
                "Outer-fold mismatch."
            )

        if config[
            "method"
        ] != method:
            raise RuntimeError(
                "Method mismatch."
            )

        if config[
            "optimizer_seed"
        ] != seed:
            raise RuntimeError(
                "Seed mismatch."
            )

        if config[
            "outer_test_evaluated"
        ] is not False:
            raise RuntimeError(
                "Outer test was evaluated."
            )

        if (
            config[
                "spec_sha256"
            ]
            != EXPECTED_SPEC_SHA
        ):
            raise RuntimeError(
                "Search-space SHA mismatch."
            )

        if (
            config[
                "algorithm_config_sha256"
            ]
            != EXPECTED_ALGO_SHA
        ):
            raise RuntimeError(
                "Algorithm SHA mismatch."
            )

        if (
            config[
                "trainer_sha256"
            ]
            != EXPECTED_TRAINER_SHA
        ):
            raise RuntimeError(
                "Trainer SHA mismatch."
            )

        if summary[
            "evaluation_count"
        ] != 30:
            raise RuntimeError(
                "Evaluation count != 30."
            )

        if summary[
            "evaluation_budget"
        ] != 30:
            raise RuntimeError(
                "Evaluation budget != 30."
            )

        if len(trace) != 30:
            raise RuntimeError(
                "Trace length != 30."
            )

        indices = [
            int(
                record[
                    "evaluation_index"
                ]
            )
            for record in trace
        ]

        if indices != list(
            range(1, 31)
        ):
            raise RuntimeError(
                "Evaluation indices "
                "must be exactly 1..30."
            )

        cumulative_best = float(
            "inf"
        )

        for index, record in enumerate(
            trace,
            start=1,
        ):

            losses = [
                float(value)
                for value
                in record[
                    "fold_losses"
                ]
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
                    "Non-finite fold loss."
                )

            recomputed = (
                sum(losses)
                / 5.0
            )

            fitness = float(
                record[
                    "fitness"
                ]
            )

            if not math.isclose(
                fitness,
                recomputed,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RuntimeError(
                    "Fitness is not exact "
                    "five-fold mean."
                )

            n_fitness_verified += len(losses)

            cumulative_best = min(
                cumulative_best,
                fitness,
            )

            convergence_values[
                (
                    method,
                    index,
                )
            ].append(
                cumulative_best
            )

        best = summary[
            "best"
        ]

        trace_best = min(
            (
                float(
                    record[
                        "fitness"
                    ]
                ),
                int(
                    record[
                        "parameter_count"
                    ]
                ),
                str(
                    record[
                        "candidate_key"
                    ]
                ),
            )
            for record in trace
        )

        reported_best = (
            float(
                best[
                    "fitness"
                ]
            ),
            int(
                best[
                    "parameter_count"
                ]
            ),
            str(
                best[
                    "candidate_key"
                ]
            ),
        )

        if reported_best != trace_best:
            raise RuntimeError(
                "Reported best does not "
                "match trace ranking."
            )

        candidate = best[
            "candidate"
        ]

        if len(
            best[
                "fold_losses"
            ]
        ) != 5:
            raise RuntimeError(
                "Best candidate does not "
                "have five fold losses."
            )

        search_rows.append(
            {
                "outer_fold": outer,
                "method": method,
                "optimizer_seed": seed,
                "best_fitness": float(
                    best[
                        "fitness"
                    ]
                ),
                "parameter_count": int(
                    best[
                        "parameter_count"
                    ]
                ),
                "hidden_size": int(
                    candidate[
                        "hidden_size"
                    ]
                ),
                "num_layers": int(
                    candidate[
                        "num_layers"
                    ]
                ),
                "dropout": float(
                    candidate[
                        "dropout"
                    ]
                ),
                "learning_rate": float(
                    candidate[
                        "learning_rate"
                    ]
                ),
                "weight_decay": float(
                    candidate[
                        "weight_decay"
                    ]
                ),
                "candidate_key": str(
                    best[
                        "candidate_key"
                    ]
                ),
                "cache_hit_count": int(
                    summary[
                        "cache_hit_count"
                    ]
                ),
                "mutation_count_total": int(
                    summary[
                        "mutation_count_total"
                    ]
                ),
                "initial_evaluation_count": int(
                    summary[
                        "initial_evaluation_count"
                    ]
                ),
                "update_generations": int(
                    summary[
                        "update_generations"
                    ]
                ),
                "fold_loss_1": float(
                    best[
                        "fold_losses"
                    ][0]
                ),
                "fold_loss_2": float(
                    best[
                        "fold_losses"
                    ][1]
                ),
                "fold_loss_3": float(
                    best[
                        "fold_losses"
                    ][2]
                ),
                "fold_loss_4": float(
                    best[
                        "fold_losses"
                    ][3]
                ),
                "fold_loss_5": float(
                    best[
                        "fold_losses"
                    ][4]
                ),
                "outer_test_evaluated": False,
            }
        )

    if len(search_rows) != 165:
        raise RuntimeError(
            "Expected 165 aggregated searches."
        )

    # ---------------------------------------------
    # Descriptive method summaries
    # ---------------------------------------------

    method_summary = []

    for method in METHODS:

        rows = [
            row
            for row in search_rows
            if row["method"] == method
        ]

        if len(rows) != 33:
            raise RuntimeError(
                f"{method}: expected 33 rows."
            )

        fitnesses = [
            row[
                "best_fitness"
            ]
            for row in rows
        ]

        parameters = [
            row[
                "parameter_count"
            ]
            for row in rows
        ]

        cache_hits = [
            row[
                "cache_hit_count"
            ]
            for row in rows
        ]

        method_summary.append(
            {
                "method": method,
                "n_searches": len(rows),
                "mean_best_fitness": statistics.mean(
                    fitnesses
                ),
                "median_best_fitness": statistics.median(
                    fitnesses
                ),
                "std_best_fitness": sample_std(
                    fitnesses
                ),
                "min_best_fitness": min(
                    fitnesses
                ),
                "max_best_fitness": max(
                    fitnesses
                ),
                "mean_parameter_count": statistics.mean(
                    parameters
                ),
                "median_parameter_count": statistics.median(
                    parameters
                ),
                "total_cache_hits": sum(
                    cache_hits
                ),
                "mean_cache_hits": statistics.mean(
                    cache_hits
                ),
            }
        )

    # ---------------------------------------------
    # Outer-context summaries, 3 optimizer seeds
    # ---------------------------------------------

    outer_method_summary = []

    for outer in range(
        1,
        12,
    ):

        for method in METHODS:

            rows = [
                row
                for row in search_rows
                if (
                    row[
                        "outer_fold"
                    ] == outer
                    and row[
                        "method"
                    ] == method
                )
            ]

            if len(rows) != 3:
                raise RuntimeError(
                    "Expected three seeds per "
                    "outer/method."
                )

            values = [
                row[
                    "best_fitness"
                ]
                for row in rows
            ]

            outer_method_summary.append(
                {
                    "outer_fold": outer,
                    "method": method,
                    "n_optimizer_seeds": 3,
                    "mean_best_fitness": statistics.mean(
                        values
                    ),
                    "median_best_fitness": statistics.median(
                        values
                    ),
                    "min_best_fitness": min(
                        values
                    ),
                    "max_best_fitness": max(
                        values
                    ),
                    "std_best_fitness": sample_std(
                        values
                    ),
                }
            )

    # ---------------------------------------------
    # Primary paired comparisons.
    #
    # Independent statistical unit:
    #     outer context / held-out protein.
    #
    # The three stochastic optimizer seeds are
    # averaged within each outer/method first.
    #
    # This avoids treating optimizer restarts
    # within the same outer context as independent
    # biological/generalization units.
    #
    # Positive delta means Improved GWO is better.
    # ---------------------------------------------

    outer_lookup = {
        (
            int(
                row[
                    "outer_fold"
                ]
            ),
            row[
                "method"
            ],
        ): row
        for row in outer_method_summary
    }

    comparators = [
        "standard_gwo",
        "improved_no_m1",
        "improved_no_m2",
        "improved_no_m3",
    ]

    raw_comparisons = []

    tie_tolerance = 1e-12

    for comparator in comparators:

        deltas = []
        relative = []

        wins = 0
        losses = 0
        ties = 0

        for outer in range(
            1,
            12,
        ):

            improved_fitness = float(
                outer_lookup[
                    (
                        outer,
                        "improved_gwo",
                    )
                ][
                    "mean_best_fitness"
                ]
            )

            other_fitness = float(
                outer_lookup[
                    (
                        outer,
                        comparator,
                    )
                ][
                    "mean_best_fitness"
                ]
            )

            delta = (
                other_fitness
                - improved_fitness
            )

            deltas.append(
                delta
            )

            if other_fitness > 0:
                relative.append(
                    100.0
                    * delta
                    / other_fitness
                )

            if delta > tie_tolerance:
                wins += 1

            elif delta < -tie_tolerance:
                losses += 1

            else:
                ties += 1

        if len(deltas) != 11:
            raise RuntimeError(
                "Expected 11 outer-level "
                "paired comparisons."
            )

        p_value = exact_sign_test(
            wins,
            losses,
        )

        label = (
            "improved_gwo_vs_"
            + comparator
        )

        raw_comparisons.append(
            {
                "comparison": label,
                "reference_method": (
                    "improved_gwo"
                ),
                "comparator_method": (
                    comparator
                ),
                "statistical_unit": (
                    "outer_context"
                ),
                "seed_aggregation": (
                    "mean_of_three_optimizer_seeds"
                ),
                "n_pairs": 11,
                "wins_improved": wins,
                "losses_improved": losses,
                "ties": ties,
                "win_rate_non_ties": (
                    wins
                    / (
                        wins + losses
                    )
                    if (
                        wins + losses
                    )
                    else 0.0
                ),
                "mean_delta_comparator_minus_improved": (
                    statistics.mean(
                        deltas
                    )
                ),
                "median_delta_comparator_minus_improved": (
                    statistics.median(
                        deltas
                    )
                ),
                "mean_relative_improvement_percent": (
                    statistics.mean(
                        relative
                    )
                    if relative
                    else 0.0
                ),
                "median_relative_improvement_percent": (
                    statistics.median(
                        relative
                    )
                    if relative
                    else 0.0
                ),
                "exact_sign_test_p_two_sided": (
                    p_value
                ),
            }
        )

    labels = [
        row["comparison"]
        for row in raw_comparisons
    ]

    pvalues = [
        row[
            "exact_sign_test_p_two_sided"
        ]
        for row in raw_comparisons
    ]

    adjusted = holm_adjust(
        labels,
        pvalues,
    )

    for row in raw_comparisons:

        row[
            "holm_adjusted_p"
        ] = adjusted[
            row[
                "comparison"
            ]
        ]

        row[
            "holm_significant_0_05"
        ] = (
            row[
                "holm_adjusted_p"
            ] < 0.05
        )

    # ---------------------------------------------
    # Convergence:
    # cumulative best at each evaluation,
    # aggregated across 33 searches/method.
    # ---------------------------------------------

    convergence_rows = []

    for method in METHODS:

        for evaluation_index in range(
            1,
            31,
        ):

            values = convergence_values[
                (
                    method,
                    evaluation_index,
                )
            ]

            if len(values) != 33:
                raise RuntimeError(
                    "Convergence aggregation "
                    "does not contain 33 runs."
                )

            convergence_rows.append(
                {
                    "method": method,
                    "evaluation_index": (
                        evaluation_index
                    ),
                    "n_searches": 33,
                    "mean_cumulative_best_fitness": (
                        statistics.mean(
                            values
                        )
                    ),
                    "median_cumulative_best_fitness": (
                        statistics.median(
                            values
                        )
                    ),
                    "std_cumulative_best_fitness": (
                        sample_std(
                            values
                        )
                    ),
                    "min_cumulative_best_fitness": (
                        min(values)
                    ),
                    "max_cumulative_best_fitness": (
                        max(values)
                    ),
                }
            )

    # ---------------------------------------------
    # Primary Improved-GWO hyperparameter selection.
    # Exactly one inner-selected configuration/outer.
    # No outer-test information involved.
    # ---------------------------------------------

    selected_improved = []

    for outer in range(
        1,
        12,
    ):

        rows = [
            row
            for row in search_rows
            if (
                row[
                    "outer_fold"
                ] == outer
                and row[
                    "method"
                ] == "improved_gwo"
            )
        ]

        if len(rows) != 3:
            raise RuntimeError(
                "Expected 3 Improved-GWO seeds."
            )

        selected = min(
            rows,
            key=candidate_rank_key,
        )

        selected_improved.append(
            {
                "outer_fold": outer,
                "selection_method": (
                    "best_improved_gwo_"
                    "across_three_optimizer_seeds"
                ),
                "selected_optimizer_seed": (
                    selected[
                        "optimizer_seed"
                    ]
                ),
                "inner_fitness": (
                    selected[
                        "best_fitness"
                    ]
                ),
                "parameter_count": (
                    selected[
                        "parameter_count"
                    ]
                ),
                "hidden_size": (
                    selected[
                        "hidden_size"
                    ]
                ),
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
                "candidate_key": (
                    selected[
                        "candidate_key"
                    ]
                ),
                "outer_test_evaluated": False,
            }
        )

    # ---------------------------------------------
    # Output
    # ---------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        output_dir
        / "search_results.csv",
        search_rows,
        list(
            search_rows[0].keys()
        ),
    )

    write_csv(
        output_dir
        / "method_summary.csv",
        method_summary,
        list(
            method_summary[0].keys()
        ),
    )

    write_csv(
        output_dir
        / "outer_method_summary.csv",
        outer_method_summary,
        list(
            outer_method_summary[0].keys()
        ),
    )

    write_csv(
        output_dir
        / "paired_comparisons.csv",
        raw_comparisons,
        list(
            raw_comparisons[0].keys()
        ),
    )

    write_csv(
        output_dir
        / "convergence_by_method.csv",
        convergence_rows,
        list(
            convergence_rows[0].keys()
        ),
    )

    write_csv(
        output_dir
        / "improved_selected_by_outer.csv",
        selected_improved,
        list(
            selected_improved[0].keys()
        ),
    )

    summary = {
        "analysis_scope": (
            "inner_validation_only"
        ),
        "outer_test_evaluated": False,
        "n_searches": 165,
        "n_outer_contexts": 11,
        "n_methods": 5,
        "n_optimizer_seeds": 3,
        "fitness_evaluations_per_search": 30,
        "inner_folds_per_fitness": 5,
        "total_algorithmic_fitness_evaluations": (
            165 * 30
        ),
        "total_fold_loss_values_verified": (
            n_fitness_verified
        ),
        "search_checksums_verified": (
            n_checksum_verified
        ),
        "pairing_for_method_comparison": (
            "outer_context_after_mean_across_"
            "three_optimizer_seeds"
        ),
        "n_pairs_per_comparison": 11,
        "inferential_test": (
            "exact_two_sided_paired_sign_test"
        ),
        "multiple_testing_correction": (
            "Holm"
        ),
        "tie_tolerance": (
            tie_tolerance
        ),
        "primary_method": (
            "improved_gwo"
        ),
        "primary_selection_rule": (
            "for_each_outer_context_choose_"
            "lowest_inner_fitness_among_three_"
            "improved_gwo_optimizer_seeds;"
            "tie_break_parameter_count_"
            "candidate_key_seed"
        ),
        "method_summary": (
            method_summary
        ),
        "paired_comparisons": (
            raw_comparisons
        ),
        "selected_improved_by_outer": (
            selected_improved
        ),
    }

    write_json(
        output_dir
        / "summary.json",
        summary,
    )

    output_files = [
        "search_results.csv",
        "method_summary.csv",
        "outer_method_summary.csv",
        "paired_comparisons.csv",
        "convergence_by_method.csv",
        "improved_selected_by_outer.csv",
        "summary.json",
    ]

    with (
        output_dir
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for name in output_files:

            handle.write(
                sha256_file(
                    output_dir
                    / name
                )
                + "  "
                + name
                + "\n"
            )

    # Console scientific summary.
    print(
        "========================================"
    )

    print(
        "GWO INNER BENCHMARK AGGREGATION"
    )

    print(
        "========================================"
    )

    print(
        "searches:",
        len(search_rows),
    )

    print(
        "search_checksums_verified:",
        n_checksum_verified,
    )

    print(
        "fitness_records_verified:",
        n_fitness_verified,
    )

    print()
    print(
        "===== METHOD SUMMARY ====="
    )

    ordered = sorted(
        method_summary,
        key=lambda row:
        row[
            "median_best_fitness"
        ],
    )

    for rank, row in enumerate(
        ordered,
        start=1,
    ):

        print(
            rank,
            row["method"],
            "mean=",
            row[
                "mean_best_fitness"
            ],
            "median=",
            row[
                "median_best_fitness"
            ],
            "std=",
            row[
                "std_best_fitness"
            ],
            "cache_hits=",
            row[
                "total_cache_hits"
            ],
        )

    print()
    print(
        "===== PAIRED IMPROVED COMPARISONS ====="
    )

    for row in raw_comparisons:

        print(
            row[
                "comparison"
            ],
            "wins/losses/ties=",
            (
                row[
                    "wins_improved"
                ],
                row[
                    "losses_improved"
                ],
                row[
                    "ties"
                ],
            ),
            "median_delta=",
            row[
                "median_delta_comparator_minus_improved"
            ],
            "median_relative_%=",
            row[
                "median_relative_improvement_percent"
            ],
            "p=",
            row[
                "exact_sign_test_p_two_sided"
            ],
            "holm_p=",
            row[
                "holm_adjusted_p"
            ],
            "holm_sig=",
            row[
                "holm_significant_0_05"
            ],
        )

    print()
    print(
        "===== SELECTED IMPROVED GWO BY OUTER ====="
    )

    for row in selected_improved:

        print(
            "outer=",
            row["outer_fold"],
            "seed=",
            row[
                "selected_optimizer_seed"
            ],
            "fitness=",
            row[
                "inner_fitness"
            ],
            "hidden=",
            row[
                "hidden_size"
            ],
            "layers=",
            row[
                "num_layers"
            ],
            "dropout=",
            row[
                "dropout"
            ],
            "lr=",
            row[
                "learning_rate"
            ],
            "wd=",
            row[
                "weight_decay"
            ],
        )

    print()
    print(
        "SEARCHES_165: PASS"
    )

    print(
        "SEARCH_CHECKSUMS_165: PASS"
    )

    print(
        "ALGORITHMIC_EVALUATIONS_4950: PASS"
    )

    print(
        "FIVE_FOLD_FITNESS_RECORDS_24750: PASS"
    )

    print(
        "PAIRED_COMPARISONS_11_OUTERS_EACH: PASS"
    )

    print(
        "IMPROVED_SELECTIONS_11: PASS"
    )

    print(
        "OUTER_TEST_EVALUATED: NO"
    )

    print(
        "GWO_INNER_AGGREGATION: PASS"
    )


if __name__ == "__main__":
    main()
