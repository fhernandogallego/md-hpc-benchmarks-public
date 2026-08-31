#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import rms


VARIANTS = (
    "replica_prefix",
    "replica_full_oracle",
    "consensus_prefix",
    "consensus_full_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula core-RMSD usando núcleos derivados del prefijo "
            "y núcleos completos usados como referencia diagnóstica."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--basic-dir", required=True, type=Path)
    parser.add_argument(
        "--replica-core-csv",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--consensus-core-csv",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--quantiles",
        default="0.30,0.40,0.50",
    )
    parser.add_argument(
        "--prefix-end-ns",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--tail-start-ns",
        type=float,
        default=80.0,
    )
    return parser.parse_args()


def parse_quantiles(value: str) -> list[float]:
    quantiles = sorted(
        {
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )

    if not quantiles:
        raise ValueError("No se proporcionaron percentiles.")

    return quantiles


def boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    valid_values = {"true", "false"}

    if not set(normalized.unique()).issubset(valid_values):
        raise ValueError(
            f"Valores booleanos no reconocidos: "
            f"{sorted(normalized.unique())}"
        )

    return normalized.eq("true")


def subset_quantile(
    dataframe: pd.DataFrame,
    quantile: float,
) -> pd.DataFrame:
    mask = np.isclose(
        dataframe["quantile"].to_numpy(dtype=float),
        quantile,
        rtol=0.0,
        atol=1e-9,
    )

    result = dataframe.loc[mask].copy()

    if result.empty:
        raise RuntimeError(
            f"No hay registros para el percentil {quantile}."
        )

    return result


def selected_residue_indices(
    replica_flags: pd.DataFrame,
    consensus_flags: pd.DataFrame,
    replica: int,
    quantile: float,
) -> dict[str, np.ndarray]:
    replica_subset = subset_quantile(
        replica_flags.loc[
            replica_flags["replica"] == replica
        ],
        quantile,
    )

    consensus_subset = subset_quantile(
        consensus_flags,
        quantile,
    )

    replica_subset["prefix_core"] = boolean_series(
        replica_subset["prefix_core"]
    )
    replica_subset["full_core"] = boolean_series(
        replica_subset["full_core"]
    )
    consensus_subset["prefix_core"] = boolean_series(
        consensus_subset["prefix_core"]
    )
    consensus_subset["full_core"] = boolean_series(
        consensus_subset["full_core"]
    )

    return {
        "replica_prefix": replica_subset.loc[
            replica_subset["prefix_core"],
            "residue_index",
        ].to_numpy(dtype=int),
        "replica_full_oracle": replica_subset.loc[
            replica_subset["full_core"],
            "residue_index",
        ].to_numpy(dtype=int),
        "consensus_prefix": consensus_subset.loc[
            consensus_subset["prefix_core"],
            "residue_index",
        ].to_numpy(dtype=int),
        "consensus_full_oracle": consensus_subset.loc[
            consensus_subset["full_core"],
            "residue_index",
        ].to_numpy(dtype=int),
    }


def create_core_group(
    universe: mda.Universe,
    residue_indices: np.ndarray,
) -> mda.core.groups.AtomGroup:
    calphas = universe.select_atoms(
        "protein and name CA"
    )
    backbone = universe.select_atoms(
        "protein and backbone"
    )

    if len(residue_indices) == 0:
        raise RuntimeError("El núcleo no contiene residuos.")

    positions = residue_indices - 1

    if np.any(positions < 0):
        raise RuntimeError(
            "Hay índices de residuo menores que uno."
        )

    if np.any(positions >= calphas.n_atoms):
        raise RuntimeError(
            "Hay índices de residuo fuera de la topología."
        )

    topology_resindices = calphas.resindices[positions]

    mask = np.isin(
        backbone.resindices,
        topology_resindices,
    )

    group = backbone[mask]

    if group.n_atoms == 0:
        raise RuntimeError(
            "La selección de backbone del núcleo está vacía."
        )

    represented_residues = np.unique(group.resindices)

    if len(represented_residues) != len(residue_indices):
        raise RuntimeError(
            "No todos los residuos del núcleo tienen backbone."
        )

    return group


def comparison_metrics(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float | int]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    if first.shape != second.shape:
        raise ValueError(
            f"Dimensiones diferentes: "
            f"{first.shape} y {second.shape}"
        )

    errors = first - second

    correlation = float(
        np.corrcoef(first, second)[0, 1]
    )

    return {
        "n_points": int(len(first)),
        "bias_A": float(np.mean(errors)),
        "mae_A": float(np.mean(np.abs(errors))),
        "rmse_A": float(
            np.sqrt(np.mean(errors**2))
        ),
        "max_abs_error_A": float(
            np.max(np.abs(errors))
        ),
        "pearson_r": correlation,
    }


def series_summary(
    values: np.ndarray,
    time_ns: np.ndarray,
    prefix_end_ns: float,
    tail_start_ns: float,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    time_ns = np.asarray(time_ns, dtype=float)

    prefix_mask = time_ns <= prefix_end_ns + 1e-9
    tail_mask = time_ns >= tail_start_ns - 1e-9

    return {
        "n_frames": int(len(values)),
        "mean_A": float(np.mean(values)),
        "std_A": float(np.std(values, ddof=1)),
        "min_A": float(np.min(values)),
        "max_A": float(np.max(values)),
        "final_A": float(values[-1]),
        "prefix_mean_A": float(
            np.mean(values[prefix_mask])
        ),
        "prefix_std_A": float(
            np.std(values[prefix_mask], ddof=1)
        ),
        "tail_mean_A": float(
            np.mean(values[tail_mask])
        ),
        "tail_std_A": float(
            np.std(values[tail_mask], ddof=1)
        ),
    }


def process_replica_quantile(
    universe: mda.Universe,
    basic_dataframe: pd.DataFrame,
    residue_sets: dict[str, np.ndarray],
    replica: int,
    quantile: float,
    output_dir: Path,
    prefix_end_ns: float,
    tail_start_ns: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    n_frames = len(universe.trajectory)

    if len(basic_dataframe) != n_frames:
        raise RuntimeError(
            f"R{replica}: el CSV básico tiene "
            f"{len(basic_dataframe)} filas y la trayectoria "
            f"{n_frames} frames."
        )

    groups: dict[str, mda.core.groups.AtomGroup] = {}
    references: dict[str, np.ndarray] = {}

    universe.trajectory[0]

    for variant, residue_indices in residue_sets.items():
        group = create_core_group(
            universe,
            residue_indices,
        )
        groups[variant] = group
        references[variant] = group.positions.copy()

    values = {
        variant: np.empty(n_frames, dtype=np.float64)
        for variant in VARIANTS
    }

    frames = np.empty(n_frames, dtype=int)
    times_ps = np.empty(n_frames, dtype=np.float64)

    for position, timestep in enumerate(
        universe.trajectory
    ):
        frames[position] = int(timestep.frame)
        times_ps[position] = float(timestep.time)

        for variant in VARIANTS:
            values[variant][position] = rms.rmsd(
                groups[variant].positions,
                references[variant],
                center=True,
                superposition=True,
            )

    time_ns = times_ps / 1000.0

    basic_times = basic_dataframe[
        "time_ns"
    ].to_numpy(dtype=float)

    if not np.allclose(
        time_ns,
        basic_times,
        rtol=0.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            f"R{replica}: los tiempos básicos "
            "no coinciden con la trayectoria."
        )

    backbone_rmsd = basic_dataframe[
        "rmsd_backbone_A"
    ].to_numpy(dtype=float)

    quantile_label = int(round(100 * quantile))

    timeseries_dataframe = pd.DataFrame(
        {
            "system": "1k5n_A",
            "replica": replica,
            "quantile": quantile,
            "frame": frames,
            "time_ps": times_ps,
            "time_ns": time_ns,
            "backbone_rmsd_A": backbone_rmsd,
            **{
                f"{variant}_rmsd_A": variant_values
                for variant, variant_values
                in values.items()
            },
        }
    )

    timeseries_file = (
        output_dir
        / (
            f"1k5n_A_R{replica}_q"
            f"{quantile_label:02d}_core_rmsd.csv"
        )
    )
    timeseries_dataframe.to_csv(
        timeseries_file,
        index=False,
    )

    summary_records: list[dict[str, object]] = []
    comparison_records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []

    backbone_summary = series_summary(
        backbone_rmsd,
        time_ns,
        prefix_end_ns,
        tail_start_ns,
    )

    summary_records.append(
        {
            "system": "1k5n_A",
            "replica": replica,
            "quantile": quantile,
            "variant": "whole_backbone",
            "n_core_residues": 276,
            "n_core_backbone_atoms": int(
                universe.select_atoms(
                    "protein and backbone"
                ).n_atoms
            ),
            **backbone_summary,
            "timeseries_csv": str(timeseries_file),
        }
    )

    for variant in VARIANTS:
        variant_summary = series_summary(
            values[variant],
            time_ns,
            prefix_end_ns,
            tail_start_ns,
        )

        summary_records.append(
            {
                "system": "1k5n_A",
                "replica": replica,
                "quantile": quantile,
                "variant": variant,
                "n_core_residues": int(
                    len(residue_sets[variant])
                ),
                "n_core_backbone_atoms": int(
                    groups[variant].n_atoms
                ),
                **variant_summary,
                "timeseries_csv": str(
                    timeseries_file
                ),
            }
        )

        for residue_index in residue_sets[variant]:
            selection_records.append(
                {
                    "system": "1k5n_A",
                    "replica": replica,
                    "quantile": quantile,
                    "variant": variant,
                    "residue_index": int(
                        residue_index
                    ),
                }
            )

        comparison_records.append(
            {
                "system": "1k5n_A",
                "replica": replica,
                "quantile": quantile,
                "comparison": (
                    f"{variant}_vs_whole_backbone"
                ),
                **comparison_metrics(
                    values[variant],
                    backbone_rmsd,
                ),
            }
        )

    comparison_pairs = (
        (
            "replica_prefix",
            "replica_full_oracle",
        ),
        (
            "consensus_prefix",
            "consensus_full_oracle",
        ),
        (
            "replica_prefix",
            "consensus_prefix",
        ),
    )

    for first_name, second_name in comparison_pairs:
        comparison_records.append(
            {
                "system": "1k5n_A",
                "replica": replica,
                "quantile": quantile,
                "comparison": (
                    f"{first_name}_vs_{second_name}"
                ),
                **comparison_metrics(
                    values[first_name],
                    values[second_name],
                ),
            }
        )

    return (
        summary_records,
        comparison_records,
        selection_records,
    )


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    basic_dir = args.basic_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    quantiles = parse_quantiles(args.quantiles)

    replica_flags = pd.read_csv(
        args.replica_core_csv
    )
    consensus_flags = pd.read_csv(
        args.consensus_core_csv
    )

    summary_records: list[dict[str, object]] = []
    comparison_records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []

    protein_dir = input_dir / "1k5n_A_protein"
    topology = protein_dir / "1k5n_A.pdb"

    for replica in (1, 2, 3):
        trajectory = (
            protein_dir
            / f"1k5n_A_prod_R{replica}_fit.xtc"
        )

        universe = mda.Universe(
            str(topology),
            str(trajectory),
        )

        basic_file = (
            basic_dir
            / f"1k5n_A_R{replica}_basic_observables.csv"
        )
        basic_dataframe = pd.read_csv(basic_file)

        for quantile in quantiles:
            residue_sets = selected_residue_indices(
                replica_flags=replica_flags,
                consensus_flags=consensus_flags,
                replica=replica,
                quantile=quantile,
            )

            (
                replica_summary,
                replica_comparisons,
                replica_selections,
            ) = process_replica_quantile(
                universe=universe,
                basic_dataframe=basic_dataframe,
                residue_sets=residue_sets,
                replica=replica,
                quantile=quantile,
                output_dir=output_dir,
                prefix_end_ns=args.prefix_end_ns,
                tail_start_ns=args.tail_start_ns,
            )

            summary_records.extend(replica_summary)
            comparison_records.extend(
                replica_comparisons
            )
            selection_records.extend(
                replica_selections
            )

    summary = pd.DataFrame(summary_records)
    comparisons = pd.DataFrame(
        comparison_records
    )
    selections = pd.DataFrame(selection_records)

    summary_csv = output_dir / "core_rmsd_summary.csv"
    comparisons_csv = (
        output_dir / "core_rmsd_comparisons.csv"
    )
    selections_csv = (
        output_dir / "core_rmsd_selections.csv"
    )
    metadata_json = output_dir / "core_rmsd.json"

    summary.to_csv(summary_csv, index=False)
    comparisons.to_csv(
        comparisons_csv,
        index=False,
    )
    selections.to_csv(
        selections_csv,
        index=False,
    )

    metadata_json.write_text(
        json.dumps(
            {
                "quantiles": quantiles,
                "selection_atom": "CA residue",
                "rmsd_atoms": (
                    "protein backbone atoms of selected residues"
                ),
                "alignment_atoms": (
                    "same core backbone atoms"
                ),
                "reference": "frame 0",
                "weights": None,
                "prefix_end_ns": args.prefix_end_ns,
                "tail_start_ns": args.tail_start_ns,
                "leakage_policy": {
                    "replica_prefix": (
                        "eligible for single-trajectory input"
                    ),
                    "consensus_prefix": (
                        "eligible only if three prefixes "
                        "are available at inference"
                    ),
                    "replica_full_oracle": (
                        "diagnostic only"
                    ),
                    "consensus_full_oracle": (
                        "diagnostic only"
                    ),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 120)
    print("RESUMEN CORE-RMSD")
    print("=" * 120)
    print(
        summary[
            [
                "replica",
                "quantile",
                "variant",
                "n_core_residues",
                "mean_A",
                "prefix_mean_A",
                "tail_mean_A",
                "max_A",
                "final_A",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("=" * 120)
    print("COMPARACIÓN PREFIJO FRENTE A ORÁCULO")
    print("=" * 120)

    diagnostic = comparisons.loc[
        comparisons["comparison"].isin(
            [
                (
                    "replica_prefix_vs_"
                    "replica_full_oracle"
                ),
                (
                    "consensus_prefix_vs_"
                    "consensus_full_oracle"
                ),
            ]
        )
    ]

    print(
        diagnostic[
            [
                "replica",
                "quantile",
                "comparison",
                "bias_A",
                "mae_A",
                "rmse_A",
                "max_abs_error_A",
                "pearson_r",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print()
    print(f"Resumen:       {summary_csv}")
    print(f"Comparaciones: {comparisons_csv}")
    print(f"Selecciones:   {selections_csv}")
    print(f"Metadatos:     {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
