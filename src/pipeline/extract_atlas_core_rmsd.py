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
            "Calcula core-RMSD para un sistema ATLAS usando "
            "núcleos de réplica y núcleos consenso."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--dataset-group", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--replicas", default="1,2,3")
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
    parser.add_argument(
        "--expected-native-step-ps",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--expected-duration-ns",
        type=float,
        default=100.0,
    )
    return parser.parse_args()


def parse_integer_list(value: str) -> list[int]:
    values = sorted(
        {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )

    if not values:
        raise ValueError(
            "No se proporcionaron réplicas."
        )

    if any(item < 1 for item in values):
        raise ValueError(
            "Los índices de réplica deben ser positivos."
        )

    return values


def parse_float_list(value: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )

    if not values:
        raise ValueError(
            "No se proporcionaron percentiles."
        )

    for item in values:
        if not 0.0 < item < 1.0:
            raise ValueError(
                f"Percentil fuera de rango: {item}"
            )

    return values


def boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {"true", "false"}

    if not set(normalized.unique()).issubset(
        allowed
    ):
        raise ValueError(
            "Valores booleanos no reconocidos: "
            f"{sorted(normalized.unique())}"
        )

    return normalized.eq("true")


def subset_quantile(
    dataframe: pd.DataFrame,
    quantile: float,
) -> pd.DataFrame:
    mask = np.isclose(
        dataframe["quantile"].to_numpy(
            dtype=float
        ),
        quantile,
        rtol=0.0,
        atol=1e-9,
    )

    result = dataframe.loc[mask].copy()

    if result.empty:
        raise RuntimeError(
            f"No hay registros para q={quantile:.2f}."
        )

    return result


def validate_core_tables(
    replica_flags: pd.DataFrame,
    consensus_flags: pd.DataFrame,
    system: str,
    replicas: list[int],
) -> None:
    replica_required = {
        "system",
        "replica",
        "quantile",
        "residue_index",
        "prefix_core",
        "full_core",
    }
    consensus_required = {
        "system",
        "quantile",
        "residue_index",
        "prefix_core",
        "full_core",
    }

    missing_replica = replica_required.difference(
        replica_flags.columns
    )
    missing_consensus = consensus_required.difference(
        consensus_flags.columns
    )

    if missing_replica:
        raise RuntimeError(
            "Faltan columnas en núcleos por réplica: "
            f"{sorted(missing_replica)}"
        )

    if missing_consensus:
        raise RuntimeError(
            "Faltan columnas en núcleos consenso: "
            f"{sorted(missing_consensus)}"
        )

    replica_systems = set(
        replica_flags["system"].astype(str).unique()
    )
    consensus_systems = set(
        consensus_flags["system"].astype(str).unique()
    )

    if replica_systems != {system}:
        raise RuntimeError(
            f"Sistemas en núcleos por réplica: "
            f"{sorted(replica_systems)}"
        )

    if consensus_systems != {system}:
        raise RuntimeError(
            f"Sistemas en núcleos consenso: "
            f"{sorted(consensus_systems)}"
        )

    observed_replicas = sorted(
        replica_flags["replica"]
        .astype(int)
        .unique()
        .tolist()
    )

    if observed_replicas != replicas:
        raise RuntimeError(
            f"Réplicas observadas: {observed_replicas}; "
            f"esperadas: {replicas}"
        )


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

    replica_subset["prefix_core"] = (
        boolean_series(
            replica_subset["prefix_core"]
        )
    )
    replica_subset["full_core"] = (
        boolean_series(
            replica_subset["full_core"]
        )
    )
    consensus_subset["prefix_core"] = (
        boolean_series(
            consensus_subset["prefix_core"]
        )
    )
    consensus_subset["full_core"] = (
        boolean_series(
            consensus_subset["full_core"]
        )
    )

    residue_sets = {
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

    for variant, indices in residue_sets.items():
        if len(indices) == 0:
            raise RuntimeError(
                f"R{replica} q={quantile:.2f}: "
                f"el núcleo {variant} está vacío."
            )

        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(
                f"R{replica} q={quantile:.2f}: "
                f"hay residuos duplicados en {variant}."
            )

    return residue_sets


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

    positions = (
        np.asarray(
            residue_indices,
            dtype=int,
        )
        - 1
    )

    if np.any(positions < 0):
        raise RuntimeError(
            "Hay índices de residuo menores que uno."
        )

    if np.any(positions >= calphas.n_atoms):
        raise RuntimeError(
            "Hay índices de residuo fuera de la topología."
        )

    topology_resindices = calphas.resindices[
        positions
    ]

    mask = np.isin(
        backbone.resindices,
        topology_resindices,
    )

    group = backbone[mask]

    if group.n_atoms == 0:
        raise RuntimeError(
            "La selección de backbone del núcleo está vacía."
        )

    represented_residues = np.unique(
        group.resindices
    )

    if len(represented_residues) != len(
        residue_indices
    ):
        raise RuntimeError(
            "No todos los residuos seleccionados "
            "disponen de átomos backbone."
        )

    return group


def validate_time_grid(
    time_ns: np.ndarray,
    expected_step_ps: float,
    expected_duration_ns: float,
) -> dict[str, float | int]:
    if len(time_ns) < 2:
        raise RuntimeError(
            "La trayectoria tiene menos de dos frames."
        )

    increments_ps = np.diff(time_ns) * 1000.0

    if np.any(increments_ps <= 0):
        raise RuntimeError(
            "Los tiempos no son crecientes."
        )

    median_step_ps = float(
        np.median(increments_ps)
    )
    duration_ns = float(
        time_ns[-1] - time_ns[0]
    )

    if not np.isclose(
        median_step_ps,
        expected_step_ps,
        rtol=0.0,
        atol=1e-3,
    ):
        raise RuntimeError(
            f"Paso observado {median_step_ps:.8f} ps; "
            f"esperado {expected_step_ps:.8f} ps."
        )

    if not np.isclose(
        duration_ns,
        expected_duration_ns,
        rtol=0.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            f"Duración observada {duration_ns:.8f} ns; "
            f"esperada {expected_duration_ns:.8f} ns."
        )

    return {
        "n_frames": int(len(time_ns)),
        "start_ns": float(time_ns[0]),
        "end_ns": float(time_ns[-1]),
        "duration_ns": duration_ns,
        "median_step_ps": median_step_ps,
        "maximum_step_error_ps": float(
            np.max(
                np.abs(
                    increments_ps
                    - expected_step_ps
                )
            )
        ),
    }


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

    if np.ptp(first) < 1e-14:
        correlation = float("nan")
    elif np.ptp(second) < 1e-14:
        correlation = float("nan")
    else:
        correlation = float(
            np.corrcoef(first, second)[0, 1]
        )

    return {
        "n_points": int(len(first)),
        "bias_A": float(np.mean(errors)),
        "mae_A": float(
            np.mean(np.abs(errors))
        ),
        "rmse_A": float(
            np.sqrt(
                np.mean(errors**2)
            )
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

    prefix_mask = (
        time_ns <= prefix_end_ns + 1e-9
    )
    tail_mask = (
        time_ns >= tail_start_ns - 1e-9
    )

    if prefix_mask.sum() < 2:
        raise RuntimeError(
            "La ventana del prefijo contiene menos de dos puntos."
        )

    if tail_mask.sum() < 2:
        raise RuntimeError(
            "La ventana final contiene menos de dos puntos."
        )

    return {
        "n_frames": int(len(values)),
        "mean_A": float(np.mean(values)),
        "std_A": float(
            np.std(values, ddof=1)
        ),
        "min_A": float(np.min(values)),
        "max_A": float(np.max(values)),
        "final_A": float(values[-1]),
        "prefix_mean_A": float(
            np.mean(values[prefix_mask])
        ),
        "prefix_std_A": float(
            np.std(
                values[prefix_mask],
                ddof=1,
            )
        ),
        "tail_mean_A": float(
            np.mean(values[tail_mask])
        ),
        "tail_std_A": float(
            np.std(
                values[tail_mask],
                ddof=1,
            )
        ),
    }


def process_replica_quantile(
    system: str,
    universe: mda.Universe,
    basic_dataframe: pd.DataFrame,
    residue_sets: dict[str, np.ndarray],
    replica: int,
    quantile: float,
    output_dir: Path,
    prefix_end_ns: float,
    tail_start_ns: float,
    expected_step_ps: float,
    expected_duration_ns: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    n_frames = len(universe.trajectory)

    if len(basic_dataframe) != n_frames:
        raise RuntimeError(
            f"R{replica}: el CSV básico tiene "
            f"{len(basic_dataframe)} filas y la trayectoria "
            f"{n_frames} frames."
        )

    groups: dict[
        str,
        mda.core.groups.AtomGroup,
    ] = {}
    references: dict[str, np.ndarray] = {}

    universe.trajectory[0]

    for variant, residue_indices in (
        residue_sets.items()
    ):
        group = create_core_group(
            universe,
            residue_indices,
        )
        groups[variant] = group
        references[variant] = (
            group.positions.copy()
        )

    values = {
        variant: np.empty(
            n_frames,
            dtype=np.float64,
        )
        for variant in VARIANTS
    }

    frames = np.empty(
        n_frames,
        dtype=int,
    )
    times_ps = np.empty(
        n_frames,
        dtype=np.float64,
    )

    for position, timestep in enumerate(
        universe.trajectory
    ):
        frames[position] = int(
            timestep.frame
        )
        times_ps[position] = float(
            timestep.time
        )

        for variant in VARIANTS:
            values[variant][position] = (
                rms.rmsd(
                    groups[variant].positions,
                    references[variant],
                    center=True,
                    superposition=True,
                )
            )

    time_ns = times_ps / 1000.0

    temporal_validation = validate_time_grid(
        time_ns=time_ns,
        expected_step_ps=expected_step_ps,
        expected_duration_ns=(
            expected_duration_ns
        ),
    )

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

    if not np.all(
        np.isfinite(backbone_rmsd)
    ):
        raise RuntimeError(
            f"R{replica}: RMSD básico no finito."
        )

    for variant in VARIANTS:
        if not np.all(
            np.isfinite(values[variant])
        ):
            raise RuntimeError(
                f"R{replica}: {variant} contiene "
                "valores no finitos."
            )

    quantile_label = int(
        round(100 * quantile)
    )

    timeseries_dataframe = pd.DataFrame(
        {
            "system": system,
            "replica": replica,
            "quantile": quantile,
            "frame": frames,
            "time_ps": times_ps,
            "time_ns": time_ns,
            "backbone_rmsd_A": (
                backbone_rmsd
            ),
            **{
                f"{variant}_rmsd_A": (
                    variant_values
                )
                for variant, variant_values
                in values.items()
            },
        }
    )

    timeseries_file = (
        output_dir
        / (
            f"{system}_R{replica}_q"
            f"{quantile_label:02d}_core_rmsd.csv"
        )
    )

    timeseries_dataframe.to_csv(
        timeseries_file,
        index=False,
    )

    summary_records: list[
        dict[str, object]
    ] = []
    comparison_records: list[
        dict[str, object]
    ] = []
    selection_records: list[
        dict[str, object]
    ] = []

    whole_backbone = universe.select_atoms(
        "protein and backbone"
    )

    backbone_summary = series_summary(
        values=backbone_rmsd,
        time_ns=time_ns,
        prefix_end_ns=prefix_end_ns,
        tail_start_ns=tail_start_ns,
    )

    summary_records.append(
        {
            "system": system,
            "replica": replica,
            "quantile": quantile,
            "variant": "whole_backbone",
            "n_core_residues": int(
                universe.select_atoms(
                    "protein and name CA"
                ).n_atoms
            ),
            "n_core_backbone_atoms": int(
                whole_backbone.n_atoms
            ),
            **backbone_summary,
            "timeseries_csv": str(
                timeseries_file
            ),
        }
    )

    for variant in VARIANTS:
        variant_summary = series_summary(
            values=values[variant],
            time_ns=time_ns,
            prefix_end_ns=prefix_end_ns,
            tail_start_ns=tail_start_ns,
        )

        summary_records.append(
            {
                "system": system,
                "replica": replica,
                "quantile": quantile,
                "variant": variant,
                "n_core_residues": int(
                    len(
                        residue_sets[
                            variant
                        ]
                    )
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

        for residue_index in (
            residue_sets[variant]
        ):
            selection_records.append(
                {
                    "system": system,
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
                "system": system,
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

    for first_name, second_name in (
        comparison_pairs
    ):
        comparison_records.append(
            {
                "system": system,
                "replica": replica,
                "quantile": quantile,
                "comparison": (
                    f"{first_name}_vs_"
                    f"{second_name}"
                ),
                **comparison_metrics(
                    values[first_name],
                    values[second_name],
                ),
            }
        )

    metadata = {
        "system": system,
        "replica": replica,
        "quantile": quantile,
        "timeseries_csv": str(
            timeseries_file
        ),
        **temporal_validation,
        "core_sizes": {
            variant: int(
                len(residue_sets[variant])
            )
            for variant in VARIANTS
        },
        "core_backbone_atom_counts": {
            variant: int(
                groups[variant].n_atoms
            )
            for variant in VARIANTS
        },
    }

    return (
        summary_records,
        comparison_records,
        selection_records,
        metadata,
    )


def main() -> int:
    args = parse_args()

    replicas = parse_integer_list(
        args.replicas
    )
    quantiles = parse_float_list(
        args.quantiles
    )

    raw_root = args.raw_root.resolve()
    processed_root = (
        args.processed_root.resolve()
    )

    unpacked_dir = (
        raw_root
        / args.dataset_group
        / args.system
        / "unpacked"
    )
    protein_dir = (
        unpacked_dir
        / f"{args.system}_protein"
    )

    system_root = (
        processed_root
        / args.dataset_group
        / args.system
    )

    basic_dir = system_root / "basic"
    core_overlap_dir = (
        system_root / "core_overlap"
    )
    output_dir = (
        system_root / "core_rmsd"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    topology_file = (
        protein_dir
        / f"{args.system}.pdb"
    )
    replica_core_csv = (
        core_overlap_dir
        / "core_replica_residues.csv"
    )
    consensus_core_csv = (
        core_overlap_dir
        / "core_consensus_residues.csv"
    )

    for required_file in (
        topology_file,
        replica_core_csv,
        consensus_core_csv,
    ):
        if not required_file.is_file():
            raise FileNotFoundError(
                required_file
            )

    replica_flags = pd.read_csv(
        replica_core_csv
    )
    consensus_flags = pd.read_csv(
        consensus_core_csv
    )

    validate_core_tables(
        replica_flags=replica_flags,
        consensus_flags=consensus_flags,
        system=args.system,
        replicas=replicas,
    )

    summary_records: list[
        dict[str, object]
    ] = []
    comparison_records: list[
        dict[str, object]
    ] = []
    selection_records: list[
        dict[str, object]
    ] = []
    trajectory_metadata: list[
        dict[str, object]
    ] = []

    for replica in replicas:
        trajectory_file = (
            protein_dir
            / (
                f"{args.system}_prod_"
                f"R{replica}_fit.xtc"
            )
        )

        basic_file = (
            basic_dir
            / (
                f"{args.system}_R{replica}"
                "_basic_observables.csv"
            )
        )

        if not trajectory_file.is_file():
            raise FileNotFoundError(
                trajectory_file
            )

        if not basic_file.is_file():
            raise FileNotFoundError(
                basic_file
            )

        universe = mda.Universe(
            str(topology_file),
            str(trajectory_file),
        )

        basic_dataframe = pd.read_csv(
            basic_file
        )

        for quantile in quantiles:
            residue_sets = (
                selected_residue_indices(
                    replica_flags=(
                        replica_flags
                    ),
                    consensus_flags=(
                        consensus_flags
                    ),
                    replica=replica,
                    quantile=quantile,
                )
            )

            (
                current_summary,
                current_comparisons,
                current_selections,
                current_metadata,
            ) = process_replica_quantile(
                system=args.system,
                universe=universe,
                basic_dataframe=(
                    basic_dataframe
                ),
                residue_sets=residue_sets,
                replica=replica,
                quantile=quantile,
                output_dir=output_dir,
                prefix_end_ns=(
                    args.prefix_end_ns
                ),
                tail_start_ns=(
                    args.tail_start_ns
                ),
                expected_step_ps=(
                    args.expected_native_step_ps
                ),
                expected_duration_ns=(
                    args.expected_duration_ns
                ),
            )

            summary_records.extend(
                current_summary
            )
            comparison_records.extend(
                current_comparisons
            )
            selection_records.extend(
                current_selections
            )
            trajectory_metadata.append(
                current_metadata
            )

    summary = pd.DataFrame(
        summary_records
    )
    comparisons = pd.DataFrame(
        comparison_records
    )
    selections = pd.DataFrame(
        selection_records
    )

    summary_csv = (
        output_dir
        / "core_rmsd_summary.csv"
    )
    comparisons_csv = (
        output_dir
        / "core_rmsd_comparisons.csv"
    )
    selections_csv = (
        output_dir
        / "core_rmsd_selections.csv"
    )
    metadata_json = (
        output_dir
        / "core_rmsd.json"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )
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
                "system": args.system,
                "dataset_group": (
                    args.dataset_group
                ),
                "replicas": replicas,
                "quantiles": quantiles,
                "selection_atom": (
                    "CA residue"
                ),
                "rmsd_atoms": (
                    "protein backbone atoms "
                    "of selected residues"
                ),
                "alignment_atoms": (
                    "same core backbone atoms"
                ),
                "reference": "frame 0",
                "weights": None,
                "prefix_end_ns": (
                    args.prefix_end_ns
                ),
                "tail_start_ns": (
                    args.tail_start_ns
                ),
                "trajectory_validation": (
                    trajectory_metadata
                ),
                "leakage_policy": {
                    "replica_prefix": (
                        "eligible for "
                        "single-trajectory input"
                    ),
                    "consensus_prefix": (
                        "eligible only when all "
                        "replica prefixes are available"
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
    print("CORE-RMSD PARAMETRIZADO")
    print("=" * 120)
    print(f"Sistema: {args.system}")
    print(f"Grupo:   {args.dataset_group}")
    print()

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
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

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

    print()
    print("=" * 120)
    print("PREFIJO FRENTE A ORÁCULO")
    print("=" * 120)
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
            float_format=lambda value: (
                f"{value:.5f}"
            ),
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
