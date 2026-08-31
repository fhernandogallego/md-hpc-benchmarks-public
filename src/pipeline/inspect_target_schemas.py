#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


SYSTEM = "1k5n_A"

WORK = (
    Path("/scratch")
    / "uva_dma_2"
    / "uva_dma_2_1"
    / "md-hpc-benchmarks"
)

SYSTEM_ROOT = (
    WORK
    / "processed"
    / "atlas"
    / "smoke"
    / SYSTEM
)

FILES = [
    SYSTEM_ROOT
    / "targets"
    / "validated_targets_long.csv",

    SYSTEM_ROOT
    / "structural_targets_compositional"
    / "structural_targets_long.csv",

    SYSTEM_ROOT
    / "structural_targets_compositional"
    / "structural_model_targets_wide.csv",

    SYSTEM_ROOT
    / "structural_targets_compositional"
    / "secondary_structure_composition.csv",

    SYSTEM_ROOT
    / "master_targets"
    / "master_targets_long.csv",

    SYSTEM_ROOT
    / "master_targets"
    / "master_targets_wide.csv",

    SYSTEM_ROOT
    / "master_targets"
    / "master_target_masks.csv",
]


def sha256(filename: Path) -> str:
    digest = hashlib.sha256()

    with filename.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> int:
    records: list[dict[str, object]] = []

    for filename in FILES:
        print()
        print("=" * 120)
        print(filename)
        print("=" * 120)

        if not filename.is_file():
            print("ERROR: archivo ausente")
            continue

        dataframe = pd.read_csv(
            filename,
            dtype={
                "target_mask_signature": "string",
            },
        )

        print(f"SHA-256: {sha256(filename)}")
        print(f"Forma:   {dataframe.shape}")

        print()
        print("Columnas y tipos:")
        for column in dataframe.columns:
            print(
                f"  {column:<40} "
                f"{str(dataframe[column].dtype)}"
            )

        print()
        print("Primeras filas:")
        print(
            dataframe.head(6).to_string(
                index=False,
            )
        )

        records.append(
            {
                "file": str(filename),
                "sha256": sha256(filename),
                "n_rows": int(len(dataframe)),
                "n_columns": int(
                    len(dataframe.columns)
                ),
                "columns": list(
                    dataframe.columns
                ),
                "dtypes": {
                    column: str(
                        dataframe[column].dtype
                    )
                    for column in dataframe.columns
                },
            }
        )

    output = (
        WORK
        / "results"
        / "parameterization_regression"
        / SYSTEM
        / "target_schema_inventory.json"
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            {
                "system": SYSTEM,
                "files": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print(f"Inventario: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
