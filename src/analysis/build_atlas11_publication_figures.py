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


TARGETS = [
    "rmsd",
    "rg",
    "sasa",
    "helix",
    "strand",
]

TARGET_LABELS = {
    "rmsd": "RMSD",
    "rg": r"$R_g$",
    "sasa": "SASA",
    "helix": "Helix fraction",
    "strand": "Strand fraction",
}

MODEL_ORDER = [
    "ridge_summary",
    "gru_fixed_h32",
    "gru_improved_gwo",
]

MODEL_LABELS = {
    "ridge_summary": "Ridge",
    "gru_fixed_h32": "Fixed GRU",
    "gru_improved_gwo": "Improved-GWO GRU",
}

MARKERS = {
    "ridge_summary": "o",
    "gru_fixed_h32": "s",
    "gru_improved_gwo": "^",
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--package-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


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


def read_csv(path):

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def unit_label(unit):

    if unit == "A":
        return r"$\AA$"

    if unit == "A2":
        return r"$\AA^2$"

    if unit == "fraction":
        return "fraction"

    return unit


def panel_axes():

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.3),
        constrained_layout=True,
    )

    axes = axes.ravel()

    axes[-1].axis("off")

    return fig, axes


def panel_letter(
    ax,
    letter,
):

    ax.text(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def save_figure(
    fig,
    root,
    stem,
):

    pdf = root / f"{stem}.pdf"
    png = root / f"{stem}.png"

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    return [
        pdf,
        png,
    ]


def figure_mae(
    package,
    output,
):

    rows = read_csv(
        package
        / "figure_mae_comparison.csv"
    )

    assert len(rows) == 15

    fig, axes = panel_axes()

    letters = "ABCDE"

    for index, target in enumerate(
        TARGETS
    ):

        ax = axes[index]

        subset = [
            row
            for row in rows
            if row["target"] == target
        ]

        assert len(subset) == 3

        by_model = {
            row["model"]: row
            for row in subset
        }

        unit = subset[0]["unit"]

        for x, model in enumerate(
            MODEL_ORDER
        ):

            row = by_model[model]

            mae = float(
                row["mae"]
            )

            low = float(
                row["mae_ci_low"]
            )

            high = float(
                row["mae_ci_high"]
            )

            ax.errorbar(
                [x],
                [mae],
                yerr=[
                    [mae - low],
                    [high - mae],
                ],
                marker=MARKERS[model],
                markersize=6,
                capsize=4,
                linestyle="none",
                label=MODEL_LABELS[model],
            )

        ax.set_xticks(
            range(3)
        )

        ax.set_xticklabels(
            [
                "Ridge",
                "Fixed\nGRU",
                "GWO\nGRU",
            ]
        )

        ax.set_ylabel(
            f"MAE ({unit_label(unit)})"
        )

        ax.set_title(
            TARGET_LABELS[target]
        )

        panel_letter(
            ax,
            letters[index],
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower right",
        frameon=False,
    )

    return save_figure(
        fig,
        output,
        "figure1_primary_mae_comparison",
    )


def figure_scatter(
    package,
    output,
):

    rows = read_csv(
        package
        / "figure_gwo_pred_vs_reference.csv"
    )

    assert len(rows) == 51

    fig, axes = panel_axes()

    letters = "ABCDE"

    for index, target in enumerate(
        TARGETS
    ):

        ax = axes[index]

        subset = [
            row
            for row in rows
            if row["target"] == target
        ]

        reference = np.asarray(
            [
                float(row["reference"])
                for row in subset
            ],
            dtype=np.float64,
        )

        prediction = np.asarray(
            [
                float(
                    row["gwo_prediction"]
                )
                for row in subset
            ],
            dtype=np.float64,
        )

        assert np.isfinite(
            reference
        ).all()

        assert np.isfinite(
            prediction
        ).all()

        unit = subset[0]["unit"]

        minimum = float(
            min(
                reference.min(),
                prediction.min(),
            )
        )

        maximum = float(
            max(
                reference.max(),
                prediction.max(),
            )
        )

        span = maximum - minimum

        padding = (
            0.06 * span
            if span > 0.0
            else 1.0
        )

        low = minimum - padding
        high = maximum + padding

        ax.scatter(
            reference,
            prediction,
            s=34,
        )

        ax.plot(
            [low, high],
            [low, high],
            linestyle="--",
            linewidth=1.0,
        )

        ax.set_xlim(
            low,
            high,
        )

        ax.set_ylim(
            low,
            high,
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.set_xlabel(
            f"Reference ({unit_label(unit)})"
        )

        ax.set_ylabel(
            f"Prediction ({unit_label(unit)})"
        )

        ax.set_title(
            TARGET_LABELS[target]
        )

        panel_letter(
            ax,
            letters[index],
        )

        ax.grid(
            alpha=0.20
        )

    return save_figure(
        fig,
        output,
        "figure2_gwo_predicted_vs_reference",
    )


def figure_paired(
    package,
    output,
):

    rows = read_csv(
        package
        / "figure_paired_error_differences.csv"
    )

    subset_all = [
        row
        for row in rows
        if (
            row["comparator"]
            == "gru_fixed_h32"
        )
    ]

    assert len(subset_all) == 51

    fig, axes = panel_axes()

    letters = "ABCDE"

    for index, target in enumerate(
        TARGETS
    ):

        ax = axes[index]

        subset = [
            row
            for row in subset_all
            if row["target"] == target
        ]

        subset = sorted(
            subset,
            key=lambda row:
                int(
                    row["outer_fold"]
                ),
        )

        delta = np.asarray(
            [
                float(
                    row[
                        "delta_gwo_minus_comparator"
                    ]
                )
                for row in subset
            ],
            dtype=np.float64,
        )

        systems = [
            row["system_id"]
            for row in subset
        ]

        unit = subset[0]["unit"]

        x = np.arange(
            len(subset)
        )

        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
        )

        ax.scatter(
            x,
            delta,
            s=34,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            systems,
            rotation=55,
            ha="right",
            fontsize=7,
        )

        ax.set_ylabel(
            (
                "Absolute-error difference\n"
                f"GWO - fixed GRU "
                f"({unit_label(unit)})"
            )
        )

        ax.set_title(
            TARGET_LABELS[target]
        )

        panel_letter(
            ax,
            letters[index],
        )

        ax.grid(
            axis="y",
            alpha=0.20,
        )

        mean_delta = float(
            delta.mean()
        )

        better_fraction = float(
            np.mean(
                delta < 0.0
            )
        )


    return save_figure(
        fig,
        output,
        "figure3_paired_error_gwo_vs_fixed_gru",
    )


def figure_savings(
    package,
    output,
):

    rows = read_csv(
        package
        / "table_computational_savings.csv"
    )

    values = {
        row["quantity"]:
            float(row["value"])
        for row in rows
    }

    full = values[
        "full_trajectory"
    ]

    prefix = values[
        "surrogate_prefix"
    ]

    reduction = values[
        "trajectory_reduction"
    ]

    full_total = values[
        "benchmark_full"
    ]

    prefix_total = values[
        "benchmark_prefix"
    ]

    avoided_total = values[
        "benchmark_avoided"
    ]

    fig, ax = plt.subplots(
        figsize=(6.4, 4.2),
        constrained_layout=True,
    )

    names = [
        "100-ns reference",
        "20-ns surrogate prefix",
    ]

    lengths = [
        full,
        prefix,
    ]

    x = np.arange(2)

    bars = ax.bar(
        x,
        lengths,
        width=0.58,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        names
    )

    ax.set_ylabel(
        "Simulated trajectory length per replica (ns)"
    )

    ax.set_ylim(
        0,
        max(lengths) * 1.24,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    for bar, value in zip(
        bars,
        lengths,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            bar.get_height()
            + 2.0,
            f"{value:.0f} ns",
            ha="center",
            va="bottom",
        )


    return save_figure(
        fig,
        output,
        "figure4_trajectory_length_savings",
    )


def main():

    args = parse_args()

    package = (
        args.package_root
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

    if not package.is_dir():
        raise FileNotFoundError(
            package
        )

    if output.exists() or work.exists():
        raise RuntimeError(
            "Output already exists"
        )

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    generated = []

    generated.extend(
        figure_mae(
            package,
            work,
        )
    )

    generated.extend(
        figure_scatter(
            package,
            work,
        )
    )

    generated.extend(
        figure_paired(
            package,
            work,
        )
    )

    generated.extend(
        figure_savings(
            package,
            work,
        )
    )

    metadata = {
        "analysis":
            "publication_figures",

        "source":
            "frozen_publication_package",

        "figures": [
            {
                "stem":
                    "figure1_primary_mae_comparison",

                "purpose":
                    (
                        "protein-level MAE and "
                        "95% bootstrap confidence intervals"
                    ),
            },
            {
                "stem":
                    "figure2_gwo_predicted_vs_reference",

                "purpose":
                    (
                        "held-out-protein predictions "
                        "versus references"
                    ),
            },
            {
                "stem":
                    "figure3_paired_error_gwo_vs_fixed_gru",

                "purpose":
                    (
                        "paired protein-level absolute-error "
                        "difference"
                    ),
            },
            {
                "stem":
                    "figure4_trajectory_length_savings",

                "purpose":
                    (
                        "trajectory-length computational "
                        "saving"
                    ),
            },
        ],

        "outer_test_rerun":
            False,

        "metrics_recomputed":
            False,

        "models_changed":
            False,

        "scientific_results_changed":
            False,
    }

    metadata_path = (
        work
        / "figure_metadata.json"
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

    generated.append(
        metadata_path
    )

    sums = (
        work
        / "SHA256SUMS.txt"
    )

    with sums.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in sorted(
            generated,
            key=lambda p:
                p.name,
        ):

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
        "FIGURE_1_MAE: PASS"
    )

    print(
        "FIGURE_2_PRED_VS_REFERENCE: PASS"
    )

    print(
        "FIGURE_3_PAIRED_ERROR: PASS"
    )

    print(
        "FIGURE_4_TRAJECTORY_SAVINGS: PASS"
    )

    print(
        "PDF_FILES: 4"
    )

    print(
        "PNG_FILES: 4"
    )

    print(
        "OUTER_TEST_RERUN: NO"
    )

    print(
        "PUBLICATION_FIGURES_CREATED: PASS"
    )


if __name__ == "__main__":
    main()
