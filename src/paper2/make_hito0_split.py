#!/usr/bin/env python3
"""Freeze a protein-level ATLAS split without inspecting target values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strings(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in values.tolist()]


def read_prior(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "system_id" not in (reader.fieldnames or []):
            raise ValueError("Prior-samples CSV needs a system_id column")
        systems = [row["system_id"].strip() for row in reader]
    if not systems or any(not system for system in systems):
        raise ValueError("Prior-samples CSV has no systems or empty IDs")
    return set(systems)


def make_split(
    system_ids: list[str], prior: set[str], seed: int,
    validation_fraction: float, test_fraction: float,
) -> list[dict[str, str]]:
    systems = set(system_ids)
    missing = prior - systems
    if missing:
        raise ValueError(f"Prior ATLAS20 systems missing: {sorted(missing)}")
    unseen = systems - prior
    ordered = sorted(unseen, key=lambda system: (
        hashlib.sha256(f"{seed}:{system}".encode()).digest(), system
    ))
    n_val = max(2, round(len(ordered) * validation_fraction))
    n_test = max(2, round(len(ordered) * test_fraction))
    if len(ordered) - n_val - n_test < 5:
        raise ValueError("Need at least five new training proteins and two in each holdout")
    assignments = {system: "train" for system in systems}
    assignments.update({system: "validation" for system in ordered[:n_val]})
    assignments.update({system: "test" for system in ordered[n_val:n_val + n_test]})
    return [{"system_id": system, "split": assignments[system],
             "is_prior_atlas20": str(system in prior).lower()}
            for system in sorted(systems)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--prior-samples", type=Path, required=True,
                        help="Published ATLAS20 samples CSV (system_id column)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1):
        parser.error("Both holdout fractions must be between zero and one")
    if args.validation_fraction + args.test_fraction >= 1:
        parser.error("Holdout fractions must sum to less than one")

    dataset = args.dataset_npz.expanduser().resolve()
    prior_path = args.prior_samples.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    with np.load(dataset, allow_pickle=False) as archive:
        system_ids = strings(archive["system_ids"])
        sample_ids = strings(archive["sample_ids"])
    if len(system_ids) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Dataset IDs are inconsistent or repeated")
    prior = read_prior(prior_path)
    rows = make_split(system_ids, prior, args.seed,
                      args.validation_fraction, args.test_fraction)
    output.mkdir(parents=True)
    csv_path = output / "protein_split.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    counts = {part: sum(row["split"] == part for row in rows)
              for part in ("train", "validation", "test")}
    (output / "split_metadata.json").write_text(json.dumps({
        "dataset_sha256": sha256(dataset),
        "prior_samples_sha256": sha256(prior_path),
        "split_sha256": sha256(csv_path),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "n_prior_atlas20": len(prior),
        "n_proteins": len(rows),
        "proteins_per_split": counts,
        "unit_of_split": "system_id (all replicas together)",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, sort_keys=True))
    print(csv_path)


if __name__ == "__main__":
    main()
