#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventa sistemas ATLAS descomprimidos y selecciona "
            "un lote piloto estratificado por tamaño."
        )
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--exclude",
        default="1k5n_A",
        help="Sistemas separados por comas.",
    )
    return parser.parse_args()


def stable_hash(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def first_match(
    directory: Path,
    patterns: list[str],
) -> Path | None:
    for pattern in patterns:
        matches = sorted(
            path
            for path in directory.glob(pattern)
            if path.is_file()
        )

        if matches:
            return matches[0]

    return None


def count_pdb_structure(
    filename: Path | None,
) -> tuple[int, int]:
    if filename is None:
        return 0, 0

    residues: set[
        tuple[str, str, str]
    ] = set()

    atom_count = 0

    with filename.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue

            atom_count += 1

            atom_name = line[12:16].strip()

            if atom_name != "CA":
                continue

            chain = line[21:22].strip()
            residue_number = line[22:26].strip()
            insertion_code = line[26:27].strip()

            residues.add(
                (
                    chain,
                    residue_number,
                    insertion_code,
                )
            )

    return len(residues), atom_count


def inspect_system(
    analysis_dir: Path,
) -> dict[str, object]:
    suffix = "_analysis"

    system = analysis_dir.name[
        :-len(suffix)
    ]

    parent = analysis_dir.parent
    protein_dir = (
        parent / f"{system}_protein"
    )

    pdb_file = first_match(
        analysis_dir,
        [
            f"{system}.pdb",
            "*.pdb",
        ],
    )

    n_residues, n_atoms = (
        count_pdb_structure(pdb_file)
    )

    missing: list[str] = []

    if pdb_file is None:
        missing.append("analysis_pdb")

    if not protein_dir.is_dir():
        missing.append("protein_dir")

    analysis_tpr_files: list[
        Path | None
    ] = []

    analysis_xtc_files: list[
        Path | None
    ] = []

    protein_xtc_files: list[
        Path | None
    ] = []

    for replica in (1, 2, 3):
        tpr = first_match(
            analysis_dir,
            [
                f"{system}_R{replica}.tpr",
                f"*R{replica}*.tpr",
            ],
        )

        xtc = first_match(
            analysis_dir,
            [
                f"{system}_R{replica}.xtc",
                f"*R{replica}*.xtc",
            ],
        )

        protein_xtc = (
            first_match(
                protein_dir,
                [
                    f"*R{replica}*fit.xtc",
                    f"*R{replica}*.xtc",
                ],
            )
            if protein_dir.is_dir()
            else None
        )

        analysis_tpr_files.append(tpr)
        analysis_xtc_files.append(xtc)
        protein_xtc_files.append(
            protein_xtc
        )

        if tpr is None:
            missing.append(
                f"analysis_R{replica}_tpr"
            )

        if xtc is None:
            missing.append(
                f"analysis_R{replica}_xtc"
            )

        if protein_xtc is None:
            missing.append(
                f"protein_R{replica}_fit_xtc"
            )

    total_trajectory_bytes = sum(
        path.stat().st_size
        for path in (
            analysis_xtc_files
            + protein_xtc_files
        )
        if path is not None
    )

    complete = (
        len(missing) == 0
        and n_residues > 0
        and n_atoms > 0
    )

    return {
        "system": system,
        "complete": complete,
        "n_residues": n_residues,
        "n_atoms": n_atoms,
        "expected_replicas": 3,
        "analysis_dir": str(
            analysis_dir.resolve()
        ),
        "protein_dir": (
            str(protein_dir.resolve())
            if protein_dir.exists()
            else ""
        ),
        "pdb_file": (
            str(pdb_file.resolve())
            if pdb_file is not None
            else ""
        ),
        "analysis_tpr_count": sum(
            path is not None
            for path in analysis_tpr_files
        ),
        "analysis_xtc_count": sum(
            path is not None
            for path in analysis_xtc_files
        ),
        "protein_fit_xtc_count": sum(
            path is not None
            for path in protein_xtc_files
        ),
        "trajectory_size_gb": (
            total_trajectory_bytes
            / 1024**3
        ),
        "missing_components": (
            ";".join(missing)
        ),
    }


def select_batch(
    inventory: pd.DataFrame,
    n_select: int,
    excluded: set[str],
) -> pd.DataFrame:
    eligible = inventory.loc[
        inventory["complete"]
        & ~inventory["system"].isin(
            excluded
        )
    ].copy()

    eligible = (
        eligible.sort_values(
            [
                "n_residues",
                "system",
            ]
        )
        .reset_index(drop=True)
    )

    if eligible.empty:
        return eligible

    n_bins = min(
        4,
        len(eligible),
    )

    size_bin = np.floor(
        np.arange(len(eligible))
        * n_bins
        / len(eligible)
    ).astype(int)

    eligible["size_bin"] = size_bin

    labels = {
        0: "small",
        1: "medium_small",
        2: "medium_large",
        3: "large",
    }

    eligible["size_class"] = (
        eligible["size_bin"]
        .map(labels)
        .fillna("size_group")
    )

    eligible["selection_hash"] = (
        eligible["system"]
        .map(stable_hash)
    )

    target_count = min(
        n_select,
        len(eligible),
    )

    base_quota = (
        target_count // n_bins
    )
    remainder = (
        target_count % n_bins
    )

    selected_indices: list[int] = []

    for bin_index in range(n_bins):
        quota = (
            base_quota
            + (
                1
                if bin_index < remainder
                else 0
            )
        )

        subset = (
            eligible.loc[
                eligible["size_bin"]
                == bin_index
            ]
            .sort_values(
                [
                    "selection_hash",
                    "system",
                ]
            )
        )

        selected_indices.extend(
            subset.head(quota).index.tolist()
        )

    if len(selected_indices) < target_count:
        remaining = (
            eligible.loc[
                ~eligible.index.isin(
                    selected_indices
                )
            ]
            .sort_values(
                [
                    "selection_hash",
                    "system",
                ]
            )
        )

        selected_indices.extend(
            remaining.head(
                target_count
                - len(selected_indices)
            ).index.tolist()
        )

    selected = (
        eligible.loc[selected_indices]
        .sort_values(
            [
                "size_bin",
                "n_residues",
                "system",
            ]
        )
        .reset_index(drop=True)
    )

    selected.insert(
        0,
        "batch_order",
        np.arange(
            1,
            len(selected) + 1,
        ),
    )

    return selected


def main() -> int:
    args = parse_args()

    raw_root = args.raw_root.resolve()
    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not raw_root.is_dir():
        raise FileNotFoundError(
            raw_root
        )

    analysis_dirs = sorted(
        path
        for path in raw_root.rglob(
            "*_analysis"
        )
        if path.is_dir()
    )

    records = [
        inspect_system(path)
        for path in analysis_dirs
    ]

    inventory = pd.DataFrame(
        records
    )

    if inventory.empty:
        inventory = pd.DataFrame(
            columns=[
                "system",
                "complete",
                "n_residues",
                "n_atoms",
                "expected_replicas",
                "analysis_dir",
                "protein_dir",
                "pdb_file",
                "analysis_tpr_count",
                "analysis_xtc_count",
                "protein_fit_xtc_count",
                "trajectory_size_gb",
                "missing_components",
            ]
        )
    else:
        # Si un sistema aparece varias veces, conserva
        # primero la copia completa y después la ruta
        # lexicográficamente más corta.
        inventory["path_length"] = (
            inventory["analysis_dir"]
            .astype(str)
            .str.len()
        )

        inventory = (
            inventory.sort_values(
                [
                    "system",
                    "complete",
                    "path_length",
                    "analysis_dir",
                ],
                ascending=[
                    True,
                    False,
                    True,
                    True,
                ],
            )
            .drop_duplicates(
                subset=["system"],
                keep="first",
            )
            .drop(
                columns=["path_length"]
            )
            .reset_index(drop=True)
        )

    excluded = {
        item.strip()
        for item in args.exclude.split(",")
        if item.strip()
    }

    selected = select_batch(
        inventory=inventory,
        n_select=args.n_select,
        excluded=excluded,
    )

    inventory_file = (
        output_dir
        / "atlas_inventory.csv"
    )

    selected_file = (
        output_dir
        / "atlas_batch_v1.csv"
    )

    inventory.to_csv(
        inventory_file,
        index=False,
    )

    selected.to_csv(
        selected_file,
        index=False,
    )

    print("=" * 110)
    print("INVENTARIO ATLAS")
    print("=" * 110)
    print(
        f"Sistemas detectados: "
        f"{len(inventory)}"
    )
    print(
        f"Sistemas completos:  "
        f"{int(inventory['complete'].sum())}"
    )
    print(
        f"Sistemas incompletos:"
        f" {int((~inventory['complete']).sum())}"
    )
    print(
        f"Sistemas seleccionados: "
        f"{len(selected)}"
    )

    if not inventory.empty:
        print()
        print("Distribución de tamaños:")
        print(
            inventory.loc[
                inventory["complete"],
                "n_residues",
            ]
            .describe()
            .to_string()
        )

    print()
    print("=" * 110)
    print("LOTE PILOTO")
    print("=" * 110)

    if selected.empty:
        print(
            "No hay sistemas completos elegibles "
            "además de los excluidos."
        )
    else:
        print(
            selected[
                [
                    "batch_order",
                    "system",
                    "size_class",
                    "n_residues",
                    "n_atoms",
                    "trajectory_size_gb",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(f"Inventario: {inventory_file}")
    print(f"Lote:       {selected_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
