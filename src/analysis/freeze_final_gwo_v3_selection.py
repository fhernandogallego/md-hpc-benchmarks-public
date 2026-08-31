#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SEEDS = [
    20260811,
    20260812,
    20260813,
]


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


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def rank(row):
    return (
        float(row["inner_fitness"]),
        int(row["parameter_count"]),
        str(row["candidate_key"]),
        int(row["selected_optimizer_seed"]),
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

    parser.add_argument(
        "--expected-spec-sha",
        required=True,
    )

    args = parser.parse_args()

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    output = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if output.exists():
        raise RuntimeError(
            f"Output exists: {output}"
        )

    records = []

    for outer in range(1, 12):

        for seed in SEEDS:

            search = (
                run_root
                / "searches"
                / f"fold_{outer:02d}"
                / "improved_gwo"
                / f"seed_{seed}"
            )

            summary_path = (
                search
                / "search_summary.json"
            )

            config_path = (
                search
                / "run_config.json"
            )

            if not (
                summary_path.is_file()
                and config_path.is_file()
                and (
                    search
                    / "SHA256SUMS.txt"
                ).is_file()
            ):
                raise RuntimeError(
                    f"Incomplete search: "
                    f"outer={outer} seed={seed}"
                )

            summary = read_json(
                summary_path
            )

            config = read_json(
                config_path
            )

            if (
                config["spec_sha256"]
                != args.expected_spec_sha
            ):
                raise RuntimeError(
                    "Search-space SHA mismatch"
                )

            if (
                config[
                    "outer_test_evaluated"
                ]
                is not False
            ):
                raise RuntimeError(
                    "Outer test was evaluated"
                )

            if (
                summary[
                    "evaluation_count"
                ]
                != 30
            ):
                raise RuntimeError(
                    "Unexpected evaluation count"
                )

            best = summary["best"]
            candidate = best["candidate"]

            records.append(
                {
                    "outer_fold": outer,
                    "selected_optimizer_seed":
                        seed,
                    "inner_fitness":
                        float(
                            best["fitness"]
                        ),
                    "hidden_size":
                        int(
                            candidate[
                                "hidden_size"
                            ]
                        ),
                    "num_layers":
                        int(
                            candidate[
                                "num_layers"
                            ]
                        ),
                    "dropout":
                        float(
                            candidate[
                                "dropout"
                            ]
                        ),
                    "learning_rate":
                        float(
                            candidate[
                                "learning_rate"
                            ]
                        ),
                    "weight_decay":
                        float(
                            candidate[
                                "weight_decay"
                            ]
                        ),
                    "parameter_count":
                        int(
                            best[
                                "parameter_count"
                            ]
                        ),
                    "candidate_key":
                        str(
                            best[
                                "candidate_key"
                            ]
                        ),
                    "source_search":
                        str(search),
                    "source_summary_sha256":
                        sha256_file(
                            summary_path
                        ),
                    "outer_test_used":
                        False,
                }
            )

    if len(records) != 33:
        raise RuntimeError(
            "Expected 33 searches"
        )

    selected = []

    for outer in range(1, 12):

        rows = [
            row
            for row in records
            if row[
                "outer_fold"
            ] == outer
        ]

        if len(rows) != 3:
            raise RuntimeError(
                f"outer={outer}: "
                f"expected 3 searches"
            )

        selected.append(
            min(
                rows,
                key=rank,
            )
        )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    csv_path = (
        output
        / "final_selected_models.csv"
    )

    fieldnames = [
        "outer_fold",
        "selected_optimizer_seed",
        "inner_fitness",
        "hidden_size",
        "num_layers",
        "dropout",
        "learning_rate",
        "weight_decay",
        "parameter_count",
        "candidate_key",
        "source_search",
        "source_summary_sha256",
        "outer_test_used",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(selected)

    metadata = {
        "analysis_type":
            "final_gwo_v3_model_selection",
        "n_optimizer_searches": 33,
        "n_selected_models": 11,
        "selection_unit":
            "one_model_per_outer_context",
        "selection_rule": [
            "lower_inner_fitness",
            "lower_parameter_count",
            "lexicographically_smaller_candidate_key",
            "lower_optimizer_seed_only_if_all_previous_tie"
        ],
        "final_search_space":
            "gwo_search_space_v3",
        "hard_hidden_cap": 512,
        "further_hidden_expansion":
            False,
        "mix_with_v2_results":
            False,
        "outer_test_evaluated":
            False,
        "run_root":
            str(run_root),
        "expected_spec_sha256":
            args.expected_spec_sha,
    }

    metadata_path = (
        output
        / "selection_metadata.json"
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

    files = [
        csv_path,
        metadata_path,
    ]

    with (
        output
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in files:
            handle.write(
                sha256_file(path)
                + "  "
                + path.name
                + "\n"
            )

    print(
        "========================================"
    )
    print(
        "FINAL GWO V3 SELECTION"
    )
    print(
        "========================================"
    )

    for row in selected:

        print(
            f"outer={row['outer_fold']:02d}",
            f"seed={row['selected_optimizer_seed']}",
            f"fitness={row['inner_fitness']:.9f}",
            f"hidden={row['hidden_size']}",
            f"layers={row['num_layers']}",
            f"dropout={row['dropout']}",
        )

    print()
    print(
        "selected_models:",
        len(selected),
    )
    print(
        "outer_test_evaluated: NO"
    )
    print(
        "FINAL_SELECTION_FREEZE: PASS"
    )


if __name__ == "__main__":
    main()
