#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass
class OnlineCoordinates:
    n_samples: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def create(cls, n_atoms: int) -> "OnlineCoordinates":
        return cls(
            n_samples=0,
            mean=np.zeros((n_atoms, 3), dtype=np.float64),
            m2=np.zeros((n_atoms, 3), dtype=np.float64),
        )

    def update(self, coordinates: np.ndarray) -> None:
        coordinates = np.asarray(
            coordinates,
            dtype=np.float64,
        )

        if coordinates.shape != self.mean.shape:
            raise ValueError(
                f"Dimensiones incompatibles: "
                f"{coordinates.shape} frente a {self.mean.shape}"
            )

        self.n_samples += 1

        delta = coordinates - self.mean
        self.mean += delta / self.n_samples
        delta_after_update = coordinates - self.mean
        self.m2 += delta * delta_after_update

    def rmsf(self) -> np.ndarray:
        if self.n_samples < 2:
            raise RuntimeError(
                "Se necesitan al menos dos frames para RMSF."
            )

        mean_squared_displacement = (
            np.sum(self.m2, axis=1)
            / self.n_samples
        )

        return np.sqrt(
            np.maximum(
                mean_squared_displacement,
                0.0,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula RMSF completo y del prefijo para "
            "un sistema ATLAS."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--dataset-group", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--replicas",
        default="1,2,3",
    )
    parser.add_argument(
        "--start-ns",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--prefix-end-ns",
        type=float,
        default=20.0,
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


def parse_replicas(value: str) -> list[int]:
    replicas = sorted(
        {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )

    if not replicas:
        raise ValueError(
            "No se proporcionaron réplicas."
        )

    if any(replica < 1 for replica in replicas):
        raise ValueError(
            "Los índices de réplica deben ser positivos."
        )

    return replicas


def validate_trajectory_times(
    times_ns: np.ndarray,
    expected_step_ps: float,
    expected_duration_ns: float,
) -> dict[str, float | int]:
    if len(times_ns) < 2:
        raise RuntimeError(
            "La trayectoria tiene menos de dos frames."
        )

    increments_ps = np.diff(times_ns) * 1000.0

    if np.any(increments_ps <= 0):
        raise RuntimeError(
            "Los tiempos no son estrictamente crecientes."
        )

    median_step_ps = float(
        np.median(increments_ps)
    )
    maximum_step_error_ps = float(
        np.max(
            np.abs(
                increments_ps
                - expected_step_ps
            )
        )
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

    duration_ns = float(
        times_ns[-1] - times_ns[0]
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
        "trajectory_n_frames": int(len(times_ns)),
        "trajectory_start_ns": float(times_ns[0]),
        "trajectory_end_ns": float(times_ns[-1]),
        "trajectory_duration_ns": duration_ns,
        "trajectory_median_step_ps": median_step_ps,
        "trajectory_maximum_step_error_ps": (
            maximum_step_error_ps
        ),
    }


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
    method: str,
) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    if np.ptp(first) < 1e-14:
        return float("nan")

    if np.ptp(second) < 1e-14:
        return float("nan")

    if method == "pearson":
        return float(
            pearsonr(first, second).statistic
        )

    if method == "spearman":
        return float(
            spearmanr(first, second).statistic
        )

    raise ValueError(
        f"Método de correlación no reconocido: {method}"
    )


def process_replica(
    system: str,
    protein_dir: Path,
    replica: int,
    start_ns: float,
    prefix_end_ns: float,
    expected_native_step_ps: float,
    expected_duration_ns: float,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
    list[dict[str, object]],
]:
    topology_file = (
        protein_dir / f"{system}.pdb"
    )
    trajectory_file = (
        protein_dir
        / f"{system}_prod_R{replica}_fit.xtc"
    )

    if not topology_file.is_file():
        raise FileNotFoundError(topology_file)

    if not trajectory_file.is_file():
        raise FileNotFoundError(trajectory_file)

    universe = mda.Universe(
        str(topology_file),
        str(trajectory_file),
    )

    calphas = universe.select_atoms(
        "protein and name CA"
    )

    if calphas.n_atoms == 0:
        raise RuntimeError(
            f"{system} R{replica}: selección CA vacía."
        )

    trajectory_times_ns = np.array(
        [
            float(timestep.time / 1000.0)
            for timestep in universe.trajectory
        ],
        dtype=np.float64,
    )

    temporal_validation = validate_trajectory_times(
        times_ns=trajectory_times_ns,
        expected_step_ps=expected_native_step_ps,
        expected_duration_ns=expected_duration_ns,
    )

    prefix_accumulator = OnlineCoordinates.create(
        calphas.n_atoms
    )
    full_accumulator = OnlineCoordinates.create(
        calphas.n_atoms
    )

    prefix_first_ns: float | None = None
    prefix_last_ns: float | None = None
    full_first_ns: float | None = None
    full_last_ns: float | None = None

    tolerance = 1e-9

    for timestep in universe.trajectory:
        time_ns = float(
            timestep.time / 1000.0
        )

        if time_ns < start_ns - tolerance:
            continue

        coordinates = calphas.positions.copy()

        full_accumulator.update(coordinates)

        if full_first_ns is None:
            full_first_ns = time_ns

        full_last_ns = time_ns

        if time_ns <= prefix_end_ns + tolerance:
            prefix_accumulator.update(
                coordinates
            )

            if prefix_first_ns is None:
                prefix_first_ns = time_ns

            prefix_last_ns = time_ns

    if prefix_first_ns is None:
        raise RuntimeError(
            f"{system} R{replica}: prefijo vacío."
        )

    if prefix_last_ns is None:
        raise RuntimeError(
            f"{system} R{replica}: final del prefijo ausente."
        )

    if full_first_ns is None:
        raise RuntimeError(
            f"{system} R{replica}: ventana completa vacía."
        )

    if full_last_ns is None:
        raise RuntimeError(
            f"{system} R{replica}: final completo ausente."
        )

    prefix_rmsf = prefix_accumulator.rmsf()
    full_rmsf = full_accumulator.rmsf()

    residue_indices = np.arange(
        1,
        calphas.n_atoms + 1,
        dtype=int,
    )

    records: list[dict[str, object]] = []

    for position in range(calphas.n_atoms):
        common = {
            "system": system,
            "replica": replica,
            "residue_index": int(
                residue_indices[position]
            ),
            "topology_resid": int(
                calphas.resids[position]
            ),
            "resname": str(
                calphas.resnames[position]
            ),
        }

        records.append(
            {
                **common,
                "window": "prefix",
                "start_ns": prefix_first_ns,
                "end_ns": prefix_last_ns,
                "n_frames": (
                    prefix_accumulator.n_samples
                ),
                "rmsf_A": float(
                    prefix_rmsf[position]
                ),
            }
        )

        records.append(
            {
                **common,
                "window": "full",
                "start_ns": full_first_ns,
                "end_ns": full_last_ns,
                "n_frames": (
                    full_accumulator.n_samples
                ),
                "rmsf_A": float(
                    full_rmsf[position]
                ),
            }
        )

    errors = prefix_rmsf - full_rmsf

    summary = {
        "system": system,
        "replica": replica,
        "n_residues": int(
            calphas.n_atoms
        ),
        "prefix_start_ns": prefix_first_ns,
        "prefix_end_ns": prefix_last_ns,
        "prefix_n_frames": (
            prefix_accumulator.n_samples
        ),
        "full_start_ns": full_first_ns,
        "full_end_ns": full_last_ns,
        "full_n_frames": (
            full_accumulator.n_samples
        ),
        "prefix_mean_rmsf_A": float(
            np.mean(prefix_rmsf)
        ),
        "full_mean_rmsf_A": float(
            np.mean(full_rmsf)
        ),
        "prefix_minus_full_bias_A": float(
            np.mean(errors)
        ),
        "prefix_full_mae_A": float(
            np.mean(np.abs(errors))
        ),
        "prefix_full_rmse_A": float(
            np.sqrt(
                np.mean(errors**2)
            )
        ),
        "prefix_full_max_abs_error_A": float(
            np.max(np.abs(errors))
        ),
        "prefix_full_pearson_r": (
            safe_correlation(
                prefix_rmsf,
                full_rmsf,
                method="pearson",
            )
        ),
        "prefix_full_spearman_r": (
            safe_correlation(
                prefix_rmsf,
                full_rmsf,
                method="spearman",
            )
        ),
    }

    metadata_records = [
        {
            "system": system,
            "replica": replica,
            "topology_file": str(
                topology_file
            ),
            "trajectory_file": str(
                trajectory_file
            ),
            "n_ca_atoms": int(
                calphas.n_atoms
            ),
            **temporal_validation,
        }
    ]

    return (
        pd.DataFrame(records),
        summary,
        metadata_records,
    )


def build_consensus(
    long_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        long_dataframe.groupby(
            [
                "system",
                "window",
                "residue_index",
                "topology_resid",
                "resname",
            ],
            as_index=False,
        )
        .agg(
            replica_count=(
                "replica",
                "nunique",
            ),
            rmsf_mean_A=(
                "rmsf_A",
                "mean",
            ),
            rmsf_median_A=(
                "rmsf_A",
                "median",
            ),
            rmsf_std_A=(
                "rmsf_A",
                "std",
            ),
            rmsf_min_A=(
                "rmsf_A",
                "min",
            ),
            rmsf_max_A=(
                "rmsf_A",
                "max",
            ),
        )
    )

    prefix = grouped.loc[
        grouped["window"] == "prefix"
    ].copy()

    full = grouped.loc[
        grouped["window"] == "full"
    ].copy()

    prefix = prefix.drop(
        columns=["window"]
    )
    full = full.drop(
        columns=["window"]
    )

    identifiers = {
        "system",
        "residue_index",
        "topology_resid",
        "resname",
    }

    prefix = prefix.rename(
        columns={
            column: f"prefix_{column}"
            for column in prefix.columns
            if column not in identifiers
        }
    )

    full = full.rename(
        columns={
            column: f"full_{column}"
            for column in full.columns
            if column not in identifiers
        }
    )

    consensus = prefix.merge(
        full,
        on=[
            "system",
            "residue_index",
            "topology_resid",
            "resname",
        ],
        how="inner",
        validate="one_to_one",
    )

    consensus["median_difference_A"] = (
        consensus["prefix_rmsf_median_A"]
        - consensus["full_rmsf_median_A"]
    )
    consensus["median_absolute_error_A"] = (
        consensus[
            "median_difference_A"
        ].abs()
    )

    return consensus


def consensus_summary(
    consensus: pd.DataFrame,
) -> dict[str, float | int]:
    prefix = consensus[
        "prefix_rmsf_median_A"
    ].to_numpy(dtype=float)

    full = consensus[
        "full_rmsf_median_A"
    ].to_numpy(dtype=float)

    errors = prefix - full

    return {
        "n_residues": int(len(consensus)),
        "median_profile_prefix_mean_A": float(
            np.mean(prefix)
        ),
        "median_profile_full_mean_A": float(
            np.mean(full)
        ),
        "median_profile_bias_A": float(
            np.mean(errors)
        ),
        "median_profile_mae_A": float(
            np.mean(np.abs(errors))
        ),
        "median_profile_rmse_A": float(
            np.sqrt(
                np.mean(errors**2)
            )
        ),
        "median_profile_max_abs_error_A": float(
            np.max(np.abs(errors))
        ),
        "median_profile_pearson_r": (
            safe_correlation(
                prefix,
                full,
                method="pearson",
            )
        ),
        "median_profile_spearman_r": (
            safe_correlation(
                prefix,
                full,
                method="spearman",
            )
        ),
    }


def main() -> int:
    args = parse_args()

    replicas = parse_replicas(
        args.replicas
    )

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()

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

    output_dir = (
        output_root
        / args.dataset_group
        / args.system
        / "rmsf"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    replica_dataframes: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    trajectory_metadata: list[dict[str, object]] = []

    for replica in replicas:
        dataframe, summary, metadata = (
            process_replica(
                system=args.system,
                protein_dir=protein_dir,
                replica=replica,
                start_ns=args.start_ns,
                prefix_end_ns=args.prefix_end_ns,
                expected_native_step_ps=(
                    args.expected_native_step_ps
                ),
                expected_duration_ns=(
                    args.expected_duration_ns
                ),
            )
        )

        replica_dataframes.append(dataframe)
        summaries.append(summary)
        trajectory_metadata.extend(metadata)

    residue_counts = {
        int(summary["n_residues"])
        for summary in summaries
    }

    if len(residue_counts) != 1:
        raise RuntimeError(
            "Las réplicas tienen distinto número "
            f"de residuos CA: {sorted(residue_counts)}"
        )

    long_dataframe = pd.concat(
        replica_dataframes,
        ignore_index=True,
    )

    summary_dataframe = pd.DataFrame(
        summaries
    )

    consensus = build_consensus(
        long_dataframe
    )
    aggregate_summary = consensus_summary(
        consensus
    )

    long_csv = (
        output_dir
        / "rmsf_prefix_full_long.csv"
    )
    summary_csv = (
        output_dir
        / "rmsf_prefix_full_replica_summary.csv"
    )
    consensus_csv = (
        output_dir
        / "rmsf_prefix_full_consensus.csv"
    )
    metadata_json = (
        output_dir
        / "rmsf_prefix_full_metadata.json"
    )

    long_dataframe.to_csv(
        long_csv,
        index=False,
    )
    summary_dataframe.to_csv(
        summary_csv,
        index=False,
    )
    consensus.to_csv(
        consensus_csv,
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
                "canonical_protocol": {
                    "selection": (
                        "protein and name CA"
                    ),
                    "alignment": "none",
                    "start_ns": args.start_ns,
                    "prefix_end_ns": (
                        args.prefix_end_ns
                    ),
                    "rmsf_reference": (
                        "mean position in each window"
                    ),
                    "variance_denominator": "N",
                },
                "trajectory_validation": (
                    trajectory_metadata
                ),
                "replica_summaries": summaries,
                "consensus": aggregate_summary,
                "leakage_policy": {
                    "full_profile": (
                        "diagnostic and target analysis only"
                    ),
                    "prefix_profile": (
                        "eligible for model features "
                        "and core detection"
                    ),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("RMSF PARAMETRIZADO")
    print("=" * 100)
    print(f"Sistema:  {args.system}")
    print(f"Grupo:    {args.dataset_group}")
    print(f"Réplicas: {replicas}")
    print()

    print(
        summary_dataframe.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Consenso:")
    for key, value in aggregate_summary.items():
        if isinstance(value, float):
            print(
                f"  {key:<38} {value:.6f}"
            )
        else:
            print(
                f"  {key:<38} {value}"
            )

    print()
    print(f"Perfil largo: {long_csv}")
    print(f"Resumen:      {summary_csv}")
    print(f"Consenso:     {consensus_csv}")
    print(f"Metadatos:    {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
