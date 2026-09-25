#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


SEED = 20260817
B = 10000

PAIRS = [
    ("improved_gwo", "random_search"),
    ("improved_gwo", "bayesian_optimization"),
    ("random_search", "bayesian_optimization"),
]


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)

    running = 0.0
    m = len(values)

    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)

    return adjusted


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    with args.runs.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if len(rows) != 180:
        raise RuntimeError(
            f"Expected 180 optimizer runs, got {len(rows)}"
        )

    data = {}

    for row in rows:
        method = row["method"]
        context = (
            int(row["outer_fold"]),
            int(row["optimizer_seed"]),
        )

        data.setdefault(
            method,
            {},
        )[context] = float(
            row["best_fitness"]
        )

    expected_methods = {
        "improved_gwo",
        "random_search",
        "bayesian_optimization",
    }

    if set(data) != expected_methods:
        raise RuntimeError(
            f"Unexpected methods: {sorted(data)}"
        )

    contexts = set(
        data["improved_gwo"]
    )

    if len(contexts) != 60:
        raise RuntimeError(
            "Expected 60 matched contexts"
        )

    for method in expected_methods:
        if set(data[method]) != contexts:
            raise RuntimeError(
                f"Context mismatch for {method}"
            )

    print("METHOD SUMMARIES")
    print("================")

    for method in [
        "improved_gwo",
        "random_search",
        "bayesian_optimization",
    ]:
        values = np.asarray(
            [
                data[method][context]
                for context in sorted(contexts)
            ],
            dtype=float,
        )

        print(
            f"{method}: "
            f"mean={np.mean(values):.12f}, "
            f"median={np.median(values):.12f}"
        )

    rng = np.random.default_rng(
        SEED
    )

    results = []

    for first, second in PAIRS:
        a = np.asarray(
            [
                data[first][context]
                for context in sorted(contexts)
            ],
            dtype=float,
        )

        b = np.asarray(
            [
                data[second][context]
                for context in sorted(contexts)
            ],
            dtype=float,
        )

        delta = a - b

        indices = rng.integers(
            0,
            len(delta),
            size=(B, len(delta)),
        )

        boot = np.mean(
            delta[indices],
            axis=1,
        )

        ci_low, ci_high = np.quantile(
            boot,
            [0.025, 0.975],
        )

        test = wilcoxon(
            delta,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )

        results.append(
            {
                "first": first,
                "second": second,
                "mean_delta": float(
                    np.mean(delta)
                ),
                "median_delta": float(
                    np.median(delta)
                ),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_raw": float(
                    test.pvalue
                ),
                "first_better": int(
                    np.sum(delta < 0)
                ),
                "ties": int(
                    np.sum(delta == 0)
                ),
                "second_better": int(
                    np.sum(delta > 0)
                ),
            }
        )

    adjusted = holm_adjust(
        [
            result["p_raw"]
            for result in results
        ]
    )

    print()
    print("PAIRED COMPARISONS")
    print("==================")
    print("delta = first - second; negative favors first")

    for result, p_holm in zip(
        results,
        adjusted,
    ):
        print()
        print(
            f"{result['first']} - "
            f"{result['second']}"
        )
        print(
            f"mean_delta={result['mean_delta']:.12f}"
        )
        print(
            f"median_delta={result['median_delta']:.12f}"
        )
        print(
            "ci95=["
            f"{result['ci_low']:.12f},"
            f"{result['ci_high']:.12f}"
            "]"
        )
        print(
            f"p_raw={result['p_raw']:.12g}"
        )
        print(
            f"p_holm={p_holm:.12g}"
        )
        print(
            f"contexts="
            f"{result['first_better']}/"
            f"{result['ties']}/"
            f"{result['second_better']}"
        )


if __name__ == "__main__":
    main()
