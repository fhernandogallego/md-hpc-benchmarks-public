#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "standard_gwo",
    "improved_no_m1",
    "improved_no_m2",
    "improved_no_m3",
    "improved_gwo",
]

METHOD_LABELS = {
    "standard_gwo": "Standard GWO",
    "improved_no_m1": "No M1",
    "improved_no_m2": "No M2",
    "improved_no_m3": "No M3",
    "improved_gwo": "Improved GWO",
}

MODULES = {
    "standard_gwo": ("no", "no", "no"),
    "improved_no_m1": ("no", "yes", "yes"),
    "improved_no_m2": ("yes", "no", "yes"),
    "improved_no_m3": ("yes", "yes", "no"),
    "improved_gwo": ("yes", "yes", "yes"),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def read_csv(path):
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
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


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def main():
    args = parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    work = Path(str(output) + ".work")

    if output.exists() or work.exists():
        raise RuntimeError(
            "Output already exists"
        )

    source_manifest = source / "SHA256SUMS.txt"

    method_path = (
        source
        / "optimizer_method_summary.csv"
    )

    paired_path = (
        source
        / "optimizer_paired_vs_improved.csv"
    )

    matrix_path = (
        source
        / "optimizer_outer_fold_matrix.csv"
    )

    metadata_path = (
        source
        / "optimizer_summary_metadata.json"
    )

    for path in [
        source_manifest,
        method_path,
        paired_path,
        matrix_path,
        metadata_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    method_rows = read_csv(
        method_path
    )

    paired_rows = read_csv(
        paired_path
    )

    assert len(method_rows) == 5
    assert len(paired_rows) == 4

    method_lookup = {
        row["method"]: row
        for row in method_rows
    }

    paired_lookup = {
        row["comparator"]: row
        for row in paired_rows
    }

    assert set(method_lookup) == set(
        METHOD_ORDER
    )

    assert set(paired_lookup) == {
        "standard_gwo",
        "improved_no_m1",
        "improved_no_m2",
        "improved_no_m3",
    }

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------
    # Table 1: global optimizer ranking
    # --------------------------------------------------

    benchmark_rows = []

    ranked = sorted(
        method_rows,
        key=lambda row:
            float(row["mean_best_fitness"]),
    )

    rank_lookup = {
        row["method"]: rank
        for rank, row in enumerate(
            ranked,
            start=1,
        )
    }

    for method in METHOD_ORDER:

        row = method_lookup[method]

        benchmark_rows.append(
            {
                "method": method,
                "label":
                    METHOD_LABELS[method],
                "rank":
                    rank_lookup[method],
                "n_searches":
                    int(row["n_searches"]),
                "mean_best_fitness":
                    float(
                        row[
                            "mean_best_fitness"
                        ]
                    ),
                "std_best_fitness":
                    float(
                        row[
                            "std_best_fitness"
                        ]
                    ),
                "median_best_fitness":
                    float(
                        row[
                            "median_best_fitness"
                        ]
                    ),
                "min_best_fitness":
                    float(
                        row[
                            "min_best_fitness"
                        ]
                    ),
                "max_best_fitness":
                    float(
                        row[
                            "max_best_fitness"
                        ]
                    ),
            }
        )

    write_csv(
        work
        / "table_optimizer_benchmark.csv",
        benchmark_rows,
        [
            "method",
            "label",
            "rank",
            "n_searches",
            "mean_best_fitness",
            "std_best_fitness",
            "median_best_fitness",
            "min_best_fitness",
            "max_best_fitness",
        ],
    )

    # --------------------------------------------------
    # Table 2: module-wise ablation
    # --------------------------------------------------

    ablation_rows = []

    for method in METHOD_ORDER:

        row = method_lookup[method]

        m1, m2, m3 = MODULES[method]

        result = {
            "method": method,
            "label":
                METHOD_LABELS[method],
            "M1": m1,
            "M2": m2,
            "M3": m3,
            "mean_best_fitness":
                float(
                    row[
                        "mean_best_fitness"
                    ]
                ),
        }

        if method == "improved_gwo":

            result.update(
                {
                    "paired_delta_full_minus_method":
                        0.0,
                    "ci95_low":
                        0.0,
                    "ci95_high":
                        0.0,
                    "full_better_fraction":
                        0.0,
                    "paired_ci_excludes_zero":
                        False,
                    "comparison":
                        "reference_full_method",
                }
            )

        else:

            paired = paired_lookup[method]

            delta = float(
                paired[
                    "mean_delta_full_minus_comparator"
                ]
            )

            low = float(
                paired["ci95_low"]
            )

            high = float(
                paired["ci95_high"]
            )

            full_better = float(
                paired[
                    "full_better_fraction"
                ]
            )

            excludes = (
                high < 0.0
                or low > 0.0
            )

            if high < 0.0:
                comparison = (
                    "supports_full_improved_gwo"
                )
            elif low > 0.0:
                comparison = (
                    "supports_comparator"
                )
            else:
                comparison = (
                    "inconclusive"
                )

            result.update(
                {
                    "paired_delta_full_minus_method":
                        delta,
                    "ci95_low":
                        low,
                    "ci95_high":
                        high,
                    "full_better_fraction":
                        full_better,
                    "paired_ci_excludes_zero":
                        excludes,
                    "comparison":
                        comparison,
                }
            )

        ablation_rows.append(result)

    write_csv(
        work
        / "table_optimizer_ablation.csv",
        ablation_rows,
        [
            "method",
            "label",
            "M1",
            "M2",
            "M3",
            "mean_best_fitness",
            "paired_delta_full_minus_method",
            "ci95_low",
            "ci95_high",
            "full_better_fraction",
            "paired_ci_excludes_zero",
            "comparison",
        ],
    )

    # --------------------------------------------------
    # Publication figure
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.5),
        constrained_layout=True,
    )

    # Panel A
    ax = axes[0]

    x = np.arange(
        len(METHOD_ORDER)
    )

    means = np.asarray(
        [
            float(
                method_lookup[m][
                    "mean_best_fitness"
                ]
            )
            for m in METHOD_ORDER
        ]
    )

    stds = np.asarray(
        [
            float(
                method_lookup[m][
                    "std_best_fitness"
                ]
            )
            for m in METHOD_ORDER
        ]
    )

    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        capsize=4,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            METHOD_LABELS[m]
            for m in METHOD_ORDER
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Best inner-validation fitness"
    )

    ax.set_title(
        "Optimizer benchmark"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.text(
        0.02,
        0.98,
        "A",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # Panel B
    ax = axes[1]

    comparators = [
        "standard_gwo",
        "improved_no_m1",
        "improved_no_m2",
        "improved_no_m3",
    ]

    y = np.arange(
        len(comparators)
    )

    delta = np.asarray(
        [
            float(
                paired_lookup[m][
                    "mean_delta_full_minus_comparator"
                ]
            )
            for m in comparators
        ]
    )

    low = np.asarray(
        [
            float(
                paired_lookup[m][
                    "ci95_low"
                ]
            )
            for m in comparators
        ]
    )

    high = np.asarray(
        [
            float(
                paired_lookup[m][
                    "ci95_high"
                ]
            )
            for m in comparators
        ]
    )

    xerr = np.vstack(
        [
            delta - low,
            high - delta,
        ]
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.errorbar(
        delta,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        [
            METHOD_LABELS[m]
            for m in comparators
        ]
    )

    ax.set_xlabel(
        (
            "Paired fitness difference\n"
            "Improved GWO - comparator"
        )
    )

    ax.set_title(
        "Paired outer-fold comparison"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.text(
        0.02,
        0.98,
        "B",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # Sign convention ("negative favors the full Improved GWO") is stated in the
    # figure caption; the in-plot annotation was removed because it overlapped the
    # Standard GWO error bar.

    fig_path_pdf = (
        work
        / "figure_optimizer_ablation.pdf"
    )

    fig_path_png = (
        work
        / "figure_optimizer_ablation.png"
    )

    fig.savefig(
        fig_path_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        fig_path_png,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------
    # Claims
    # --------------------------------------------------

    claims = {
        "scope":
            "nested inner-validation optimizer benchmark",
        "outer_test_used":
            False,
        "lower_fitness_is_better":
            True,

        "benchmark_design": {
            "methods": 5,
            "outer_folds": 11,
            "optimizer_seeds_per_fold": 3,
            "searches_per_method": 33,
        },

        "ranking_by_mean_fitness": [
            row["method"]
            for row in ranked
        ],

        "best_mean_fitness_method":
            ranked[0]["method"],

        "best_mean_fitness":
            float(
                ranked[0][
                    "mean_best_fitness"
                ]
            ),

        "full_vs_standard": {
            "mean_delta":
                float(
                    paired_lookup[
                        "standard_gwo"
                    ][
                        "mean_delta_full_minus_comparator"
                    ]
                ),
            "ci95": [
                float(
                    paired_lookup[
                        "standard_gwo"
                    ]["ci95_low"]
                ),
                float(
                    paired_lookup[
                        "standard_gwo"
                    ]["ci95_high"]
                ),
            ],
            "full_better_fraction":
                float(
                    paired_lookup[
                        "standard_gwo"
                    ][
                        "full_better_fraction"
                    ]
                ),
        },

        "module_interpretation": {
            "M1":
                (
                    "clearest supported beneficial "
                    "module under this benchmark"
                ),
            "M2":
                (
                    "small uncertain incremental "
                    "benefit; CI includes zero"
                ),
            "M3":
                (
                    "not beneficial under this "
                    "benchmark; no-M3 achieved "
                    "lower fitness"
                ),
        },

        "scientific_warning":
            (
                "No-M3 was not evaluated on the "
                "frozen outer test and no "
                "generalization claim may be "
                "made for it."
            ),
    }

    (
        work
        / "optimizer_claims.json"
    ).write_text(
        json.dumps(
            claims,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------
    # README
    # --------------------------------------------------

    readme = """ATLAS11 optimizer / ablation publication package

Source:
Frozen Step 166 inner-validation optimizer summary.

Contents:
- table_optimizer_benchmark.csv
- table_optimizer_ablation.csv
- figure_optimizer_ablation.pdf
- figure_optimizer_ablation.png
- optimizer_claims.json
- README.txt
- SHA256SUMS.txt

Scientific scope:
This package summarizes optimizer behavior using nested
inner-validation fitness. It is not an outer-test model comparison.

Interpretation:
- Full Improved GWO clearly outperforms Standard GWO.
- M1 has the clearest beneficial ablation evidence.
- M2 has an uncertain incremental effect.
- No-M3 achieves better inner-validation fitness than the full method.
- No-M3 was not outer-tested and no unseen-protein generalization
  claim is permitted for that variant.
"""

    (
        work
        / "README.txt"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Manifest
    # --------------------------------------------------

    artifacts = [
        work / "table_optimizer_benchmark.csv",
        work / "table_optimizer_ablation.csv",
        work / "figure_optimizer_ablation.pdf",
        work / "figure_optimizer_ablation.png",
        work / "optimizer_claims.json",
        work / "README.txt",
    ]

    with (
        work
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in artifacts:

            handle.write(
                sha256_file(path)
                + "  "
                + path.name
                + "\n"
            )

    work.rename(output)

    # --------------------------------------------------
    # Final console audit
    # --------------------------------------------------

    print(
        "publication_table_rows:",
        len(benchmark_rows),
    )

    print(
        "ablation_table_rows:",
        len(ablation_rows),
    )

    print(
        "best_method:",
        ranked[0]["method"],
    )

    print(
        "best_mean_fitness:",
        f"{float(ranked[0]['mean_best_fitness']):.9f}",
    )

    print(
        "figure_pdf:",
        "PASS",
    )

    print(
        "figure_png:",
        "PASS",
    )

    print(
        "OUTER_TEST_USED: NO"
    )

    print(
        "SCIENTIFIC_RESULTS_RECOMPUTED: NO"
    )

    print(
        "PUBLICATION_PACKAGE_CREATED: PASS"
    )


if __name__ == "__main__":
    main()
