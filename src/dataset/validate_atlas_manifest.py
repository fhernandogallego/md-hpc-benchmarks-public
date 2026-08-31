#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida las entradas declaradas en el manifiesto ATLAS."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def as_boolean(value: object) -> bool:
    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Booleano no reconocido: {value!r}")


def validate_system(
    raw_root: Path,
    system: str,
    dataset_group: str,
    expected_replicas: int,
) -> dict[str, object]:
    unpacked = (
        raw_root
        / dataset_group
        / system
        / "unpacked"
    )

    analysis_dir = unpacked / f"{system}_analysis"
    protein_dir = unpacked / f"{system}_protein"

    required_files: list[Path] = [
        analysis_dir / f"{system}.pdb",
        protein_dir / f"{system}.pdb",
        analysis_dir / f"{system}_RMSD.tsv",
        analysis_dir / f"{system}_RMSF.tsv",
        analysis_dir / f"{system}_gyrate.tsv",
    ]

    for replica in range(1, expected_replicas + 1):
        required_files.extend(
            [
                analysis_dir / f"{system}_R{replica}.xtc",
                protein_dir
                / f"{system}_prod_R{replica}_fit.xtc",
            ]
        )

    missing = [
        str(filename)
        for filename in required_files
        if not filename.is_file()
    ]

    existing_sizes = [
        filename.stat().st_size
        for filename in required_files
        if filename.is_file()
    ]

    zero_sized = [
        str(filename)
        for filename in required_files
        if filename.is_file()
        and filename.stat().st_size == 0
    ]

    valid = not missing and not zero_sized

    return {
        "system": system,
        "dataset_group": dataset_group,
        "expected_replicas": expected_replicas,
        "unpacked_dir": str(unpacked),
        "analysis_dir": str(analysis_dir),
        "protein_dir": str(protein_dir),
        "required_file_count": len(required_files),
        "existing_file_count": (
            len(required_files) - len(missing)
        ),
        "missing_file_count": len(missing),
        "zero_sized_file_count": len(zero_sized),
        "existing_total_bytes": sum(existing_sizes),
        "input_valid": valid,
        "missing_files": "|".join(missing),
        "zero_sized_files": "|".join(zero_sized),
    }


def main() -> int:
    args = parse_args()

    manifest_file = args.manifest.resolve()
    raw_root = args.raw_root.resolve()
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        manifest_file,
        dtype={
            "system": "string",
            "dataset_group": "string",
        },
    )

    required_columns = {
        "system",
        "enabled",
        "expected_replicas",
        "dataset_group",
        "status",
    }

    missing_columns = required_columns.difference(
        manifest.columns
    )

    if missing_columns:
        raise RuntimeError(
            f"Faltan columnas: {sorted(missing_columns)}"
        )

    manifest["enabled_bool"] = manifest[
        "enabled"
    ].map(as_boolean)

    enabled = manifest.loc[
        manifest["enabled_bool"]
    ].copy()

    if enabled.empty:
        raise RuntimeError(
            "No hay sistemas habilitados."
        )

    records: list[dict[str, object]] = []

    for _, row in enabled.iterrows():
        system = str(row["system"])
        dataset_group = str(row["dataset_group"])
        expected_replicas = int(
            row["expected_replicas"]
        )

        record = validate_system(
            raw_root=raw_root,
            system=system,
            dataset_group=dataset_group,
            expected_replicas=expected_replicas,
        )

        record["manifest_status"] = str(
            row["status"]
        )
        records.append(record)

    results = pd.DataFrame(records)
    results.to_csv(output_csv, index=False)

    print("=" * 100)
    print("VALIDACIÓN DEL MANIFIESTO ATLAS")
    print("=" * 100)
    print(
        results[
            [
                "system",
                "dataset_group",
                "expected_replicas",
                "required_file_count",
                "existing_file_count",
                "missing_file_count",
                "zero_sized_file_count",
                "input_valid",
            ]
        ].to_string(index=False)
    )

    invalid = results.loc[
        ~results["input_valid"]
    ]

    print()
    print(f"Sistemas habilitados: {len(results)}")
    print(
        f"Sistemas válidos:     "
        f"{int(results['input_valid'].sum())}"
    )
    print(f"Sistemas inválidos:   {len(invalid)}")
    print(f"Resultado:             {output_csv}")

    if not invalid.empty:
        print()
        print("ENTRADAS INVÁLIDAS:")
        print(
            invalid[
                [
                    "system",
                    "missing_files",
                    "zero_sized_files",
                ]
            ].to_string(index=False)
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
