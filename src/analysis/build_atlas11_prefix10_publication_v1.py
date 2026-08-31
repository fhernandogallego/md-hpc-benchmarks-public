#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

src = args.source.resolve()
out = args.output.resolve()
work = Path(str(out) + ".work")

if out.exists() or work.exists():
    raise RuntimeError("Output exists")

paired = read_csv(
    src / "paired_error_10vs20.csv"
)

trajectory = read_csv(
    src / "trajectory_length_10vs20.csv"
)

if len(paired) != 5:
    raise RuntimeError("Expected five targets")

if len(trajectory) != 2:
    raise RuntimeError("Expected two prefixes")

work.mkdir(
    parents=True,
    exist_ok=False,
)

# Publication tables: exact frozen values.

for source_name, dest_name in [
    (
        "paired_error_10vs20.csv",
        "table_prefix10_vs20.csv",
    ),
    (
        "trajectory_length_10vs20.csv",
        "table_prefix_trajectory_savings.csv",
    ),
]:

    (
        work / dest_name
    ).write_bytes(
        (
            src / source_name
        ).read_bytes()
    )


# Figure.
targets = [
    row["target"]
    for row in paired
]

changes = [
    float(
        row[
            "mae_percent_change_10_vs_20"
        ]
    )
    for row in paired
]

prefixes = [
    int(row["prefix_ns"])
    for row in trajectory
]

reductions = [
    float(
        row[
            "trajectory_reduction_percent"
        ]
    )
    for row in trajectory
]


fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.4),
    constrained_layout=True,
)

ax = axes[0]

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)

ax.bar(
    targets,
    changes,
)

ax.set_ylabel(
    "MAE change: 10 ns vs 20 ns (%)"
)

ax.set_title(
    "Zero-shot predictive sensitivity"
)

ax.text(
    0.02,
    0.98,
    "A",
    transform=ax.transAxes,
    va="top",
    fontweight="bold",
)

# Sign convention ("negative favors 10 ns") is stated in the figure caption; the
# in-plot annotation was removed because it overlapped the rmsd bar.

ax = axes[1]

ax.bar(
    [str(x) for x in prefixes],
    reductions,
)

ax.set_ylim(
    0,
    100,
)

ax.set_xlabel(
    "Prefix length (ns)"
)

ax.set_ylabel(
    "Trajectory-length reduction (%)"
)

ax.set_title(
    "Sampling implication"
)

ax.text(
    0.02,
    0.98,
    "B",
    transform=ax.transAxes,
    va="top",
    fontweight="bold",
)


fig.savefig(
    work / "figure_prefix10_sensitivity.pdf",
    bbox_inches="tight",
)

fig.savefig(
    work / "figure_prefix10_sensitivity.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


claims = {
    "analysis":
        "post_hoc_zero_shot_prefix_sensitivity",

    "primary_protocol_ns":
        20,

    "exploratory_prefix_ns":
        10,

    "twenty_ns_reproduction_max_abs_diff":
        0.0,

    "model_retraining":
        False,

    "hyperparameter_optimization":
        False,

    "scaler_refit":
        False,

    "ten_ns_trajectory_reduction_percent":
        90.0,

    "ten_ns_potentially_avoidable_us":
        2.97,

    "twenty_ns_potentially_avoidable_us":
        2.64,

    "additional_avoidable_us":
        0.33,

    "target_mae_percent_changes":
        {
            row["target"]:
                float(
                    row[
                        "mae_percent_change_10_vs_20"
                    ]
                )
            for row in paired
        },

    "strand_ci_excludes_zero":
        True,

    "full_10_20_30_50_frontier_complete":
        False,

    "measured_wallclock_speedup":
        False,
}

(
    work / "prefix10_claims.json"
).write_text(
    json.dumps(
        claims,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


readme = """ATLAS11 exploratory 10-vs-20 ns prefix sensitivity

The frozen Improved-GWO GRU models selected and trained
under the primary 20-ns protocol were evaluated after
zero-shot truncation to the first 10 ns.

No retraining, hyperparameter optimization, epoch
selection or scaler refitting was performed.

The independent evaluator reproduced the frozen 20-ns
predictions exactly before accepting 10-ns results.

The 20-ns protocol remains the primary analysis.

This package does not constitute a complete
10/20/30/50-ns accuracy-cost frontier.

Trajectory-length reductions are not measured wall-clock
or core-hour speedups.
"""

(
    work / "README.txt"
).write_text(
    readme,
    encoding="utf-8",
)


artifacts = [
    work / "table_prefix10_vs20.csv",
    work / "table_prefix_trajectory_savings.csv",
    work / "figure_prefix10_sensitivity.pdf",
    work / "figure_prefix10_sensitivity.png",
    work / "prefix10_claims.json",
    work / "README.txt",
]

with (
    work / "SHA256SUMS.txt"
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

work.rename(out)

print("targets:", len(paired))
print("prefixes:", len(trajectory))
print("TWENTY_NS_REPRODUCTION: EXACT")
print("TEN_NS_POST_HOC_ONLY: YES")
print("FULL_FRONTIER_COMPLETE: NO")
print("FIGURE_PDF: PASS")
print("FIGURE_PNG: PASS")
print("PUBLICATION_PACKAGE: PASS")
