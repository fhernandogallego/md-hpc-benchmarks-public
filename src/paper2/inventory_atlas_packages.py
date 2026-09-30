#!/usr/bin/env python3
"""Inventory validated ATLAS prefix packages without selecting duplicate runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.dataset.assemble_multisystem_dataset import load_source


def strings(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in values.tolist()]


def inspect(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        ids = strings(archive["sample_ids"])
    parsed = [sample.rsplit("_R", 1) for sample in ids]
    if (len(ids) != 3 or any(len(parts) != 2 or not parts[1].isdigit()
                             for parts in parsed)):
        raise ValueError(f"Cannot identify exactly one three-replica system: {path}")
    systems = {parts[0] for parts in parsed}
    if len(systems) != 1 or sorted(int(parts[1]) for parts in parsed) != [1, 2, 3]:
        raise ValueError(f"Mixed system IDs or replicas: {path}")
    system = systems.pop()
    source = load_source(system, path.parent)
    if sorted(source["replicas"].astype(int).tolist()) != [1, 2, 3]:
        raise ValueError(f"Expected replicas 1, 2, 3: {path}")
    if source["X_raw"].shape != (3, 201, 8):
        raise ValueError(f"Unexpected prefix tensor shape: {path}")
    return {
        "system": system, "release": str(path.parent.resolve()),
        "npz_sha256": source["npz_sha256"],
        "valid_targets": int(source["target_mask"].sum()),
        "total_targets": int(source["target_mask"].size),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Root of completed ATLAS pipeline runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.runs_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if output.exists():
        raise FileExistsError(output)
    packages = sorted(root.rglob("prefix_sequences_raw.npz"))
    if not packages:
        raise ValueError(f"No prefix packages found under {root}")
    rows = [inspect(path) for path in packages]
    counts = Counter(row["system"] for row in rows)
    duplicates = {system: count for system, count in counts.items() if count > 1}
    output.mkdir(parents=True)
    write_csv(output / "package_inventory.csv", rows,
              ["system", "release", "npz_sha256", "valid_targets", "total_targets"])
    report = {
        "runs_root": str(root), "n_packages": len(rows),
        "n_unique_systems": len(counts),
        "duplicate_systems": duplicates,
        "n_valid_target_cells_all_packages": sum(
            row["valid_targets"] for row in rows
        ),
    }
    (output / "inventory_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not duplicates:
        write_csv(output / "atlas_sources.csv", rows, ["system", "release"])
    print(json.dumps(report, sort_keys=True))
    if duplicates:
        print("Duplicate systems found; select one validated run per system "
              "in a separate system,release CSV before assembling.")
    else:
        print(output / "atlas_sources.csv")


if __name__ == "__main__":
    main()
