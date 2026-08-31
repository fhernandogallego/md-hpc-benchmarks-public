#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--work",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    project = args.project.resolve()
    work_root = args.work.resolve()
    output = args.output.resolve()
    tmp = Path(str(output) + ".work")

    if output.exists() or tmp.exists():
        raise RuntimeError(
            "Output already exists"
        )

    items = [
        (
            "raw_dataset",
            work_root
            / "results/poc_releases/multisystem/"
              "atlas11_v1_2026-08-07/raw/"
              "multisystem_dataset_raw.npz",
        ),
        (
            "outer_preprocessing_manifest",
            work_root
            / "results/preprocessing/"
              "atlas11_outer_training_v1_2026-08-17/"
              "outer_training_manifest.csv",
        ),
        (
            "gwo_checkpoint_manifest",
            work_root
            / "results/modeling/"
              "atlas11_gru_final_outer_freeze_v1_2026-08-17/"
              "final_checkpoint_manifest.csv",
        ),
        (
            "gwo_freeze_metadata",
            work_root
            / "results/modeling/"
              "atlas11_gru_final_outer_freeze_v1_2026-08-17/"
              "freeze_metadata.json",
        ),
        (
            "fixed_gru_checkpoint_manifest",
            work_root
            / "results/modeling/"
              "atlas11_gru_baseline_final_outer_freeze_v1_2026-08-17/"
              "final_checkpoint_manifest.csv",
        ),
        (
            "fixed_gru_freeze_metadata",
            work_root
            / "results/modeling/"
              "atlas11_gru_baseline_final_outer_freeze_v1_2026-08-17/"
              "freeze_metadata.json",
        ),
        (
            "ridge_checkpoint_manifest",
            work_root
            / "results/modeling/"
              "atlas11_ridge_summary_outer_v1_2026-08-17/"
              "ridge_checkpoint_manifest.csv",
        ),
        (
            "outer_test_protocol",
            project
            / "config/atlas11_outer_test_protocol_v1.json",
        ),
        (
            "outer_test_primary_metrics",
            work_root
            / "results/evaluation/"
              "atlas11_outer_test_v1_2026-08-17/"
              "primary_protein_metrics.csv",
        ),
        (
            "outer_test_replica_predictions",
            work_root
            / "results/evaluation/"
              "atlas11_outer_test_v1_2026-08-17/"
              "replica_predictions.csv",
        ),
        (
            "outer_test_protein_predictions",
            work_root
            / "results/evaluation/"
              "atlas11_outer_test_v1_2026-08-17/"
              "protein_predictions.csv",
        ),
        (
            "outer_test_paired_comparisons",
            work_root
            / "results/evaluation/"
              "atlas11_outer_test_v1_2026-08-17/"
              "paired_model_comparisons.csv",
        ),
        (
            "outer_test_freeze_metadata",
            work_root
            / "results/evaluation/"
              "atlas11_outer_test_final_freeze_v1_2026-08-17/"
              "freeze_metadata.json",
        ),
        (
            "optimizer_method_summary",
            work_root
            / "results/analysis/"
              "atlas11_optimizer_ablation_publication_v1_2026-08-18/"
              "optimizer_method_summary.csv",
        ),
        (
            "optimizer_paired_summary",
            work_root
            / "results/analysis/"
              "atlas11_optimizer_ablation_publication_v1_2026-08-18/"
              "optimizer_paired_vs_improved.csv",
        ),
        (
            "computational_savings",
            work_root
            / "results/analysis/"
              "atlas11_computational_savings_v1_2026-08-17/"
              "computational_savings.json",
        ),
        (
            "prefix10_paired_results",
            work_root
            / "results/analysis/"
              "atlas11_prefix10_zero_shot_v1_2026-08-18/"
              "paired_error_10vs20.csv",
        ),
        (
            "prefix10_metadata",
            work_root
            / "results/analysis/"
              "atlas11_prefix10_zero_shot_v1_2026-08-18/"
              "metadata.json",
        ),
        (
            "publication_figure_manifest",
            project
            / "paper/figures/SHA256SUMS.txt",
        ),
    ]

    missing = [
        (role, path)
        for role, path in items
        if not path.is_file()
    ]

    if missing:
        for role, path in missing:
            print(
                "MISSING:",
                role,
                path,
            )

        raise RuntimeError(
            f"{len(missing)} canonical files missing"
        )

    rows = []

    for role, path in items:

        rows.append(
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    tmp.mkdir(
        parents=True,
        exist_ok=False,
    )

    inventory = (
        tmp
        / "scientific_artifact_inventory.csv"
    )

    with inventory.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "role",
                "path",
                "bytes",
                "sha256",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "status":
            "final_scientific_evidence_freeze",

        "dataset":
            "ATLAS11",

        "independent_proteins":
            11,

        "trajectories":
            33,

        "primary_prefix_ns":
            20,

        "primary_reference_ns":
            100,

        "primary_model":
            "Improved-GWO GRU",

        "primary_outer_protocol":
            "LOPO held-out protein",

        "primary_outer_test_opened":
            True,

        "primary_outer_test_complete":
            True,

        "optimizer_benchmark_complete":
            True,

        "optimizer_ablation_complete":
            True,

        "prefix10_sensitivity": {
            "status":
                "post_hoc_zero_shot",

            "primary_protocol_changed":
                False,

            "retraining":
                False,
        },

        "trajectory_length_savings": {
            "20ns_percent":
                80.0,

            "10ns_exploratory_percent":
                90.0,

            "measured_wallclock_speedup":
                False,
        },

        "not_completed": [
            "independently optimized 10 ns model",
            "30 ns prefix sensitivity",
            "50 ns prefix sensitivity",
            "complete 10/20/30/50 accuracy-cost frontier",
            "measured MD wall-clock/core-hour benchmark",
            "mdCATH external validation",
            "p53 cross-engine validation",
        ],

        "scientific_results_recomputed":
            False,

        "artifact_count":
            len(rows),
    }

    metadata_path = (
        tmp
        / "science_freeze_metadata.json"
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

    readme = """ATLAS11 final scientific evidence freeze

This directory is a provenance index only.

It does not contain newly computed scientific results.
It records SHA-256 identities for the canonical frozen
dataset, models, outer-test results, optimizer benchmark,
ablation analysis, computational-savings analysis,
10-vs-20 ns post-hoc sensitivity analysis, and publication
figure manifest.

The primary protocol remains the frozen 20-ns
held-out-protein LOPO evaluation.

The 10-ns analysis is exploratory zero-shot truncation.

No measured MD wall-clock speedup is claimed.

No training, optimization or outer-test evaluation was
rerun while producing this freeze.
"""

    (
        tmp / "README.txt"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    generated = [
        inventory,
        metadata_path,
        tmp / "README.txt",
    ]

    with (
        tmp / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in generated:
            handle.write(
                sha256_file(path)
                + "  "
                + path.name
                + "\n"
            )

    tmp.rename(output)

    print(
        "canonical_artifacts:",
        len(rows),
    )

    for row in rows:
        print(
            f"{row['role']}: PASS "
            f"{row['sha256']}"
        )

    print()
    print(
        "PRIMARY_20NS_PROTOCOL: FROZEN"
    )

    print(
        "FINAL_OUTER_TEST: COMPLETE"
    )

    print(
        "OPTIMIZER_ABLATION: COMPLETE"
    )

    print(
        "PREFIX10_ANALYSIS: POST_HOC_ONLY"
    )

    print(
        "MEASURED_MD_WALLCLOCK: NO"
    )

    print(
        "SCIENTIFIC_RESULTS_RECOMPUTED: NO"
    )

    print(
        "FINAL_SCIENCE_FREEZE: PASS"
    )


if __name__ == "__main__":
    main()
