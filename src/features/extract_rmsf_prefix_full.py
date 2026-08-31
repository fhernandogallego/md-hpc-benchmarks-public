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
        coordinates = np.asarray(coordinates, dtype=np.float64)

        if coordinates.shape != self.mean.shape:
            raise ValueError(
                f"Dimensiones incompatibles: {coordinates.shape} "
                f"frente a {self.mean.shape}"
            )

        self.n_samples += 1

        delta = coordinates - self.mean
        self.mean += delta / self.n_samples
        delta_after_update = coordinates - self.mean
        self.m2 += delta * delta_after_update

    def rmsf(self) -> np.ndarray:
        if self.n_samples < 2:
            raise RuntimeError(
                "Se necesitan al menos dos frames para calcular RMSF."
            )

        mean_squared_displacement = (
            np.sum(self.m2, axis=1) / self.n_samples
        )

        return np.sqrt(
            np.maximum(mean_squared_displacement, 0.0)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula RMSF de carbonos alfa para el prefijo "
            "y la trayectoria completa."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-ns", type=float, default=0.1)
    parser.add_argument("--prefix-end-ns", type=float, default=20.0)
    return parser.parse_args()


def process_replica(
    input_dir: Path,
    replica: int,
    start_ns: float,
    prefix_end_ns: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    protein_dir = input_dir / "1k5n_A_protein"

    topology = protein_dir / "1k5n_A.pdb"
    trajectory = (
        protein_dir
        / f"1k5n_A_prod_R{replica}_fit.xtc"
    )

    if not topology.is_file():
        raise FileNotFoundError(topology)

    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)

    universe = mda.Universe(
        str(topology),
        str(trajectory),
    )

    calphas = universe.select_atoms(
        "protein and name CA"
    )

    if calphas.n_atoms == 0:
        raise RuntimeError(
            f"R{replica}: la selección CA está vacía."
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
        time_ns = float(timestep.time / 1000.0)

        if time_ns < start_ns - tolerance:
            continue

        coordinates = calphas.positions.copy()

        full_accumulator.update(coordinates)

        if full_first_ns is None:
            full_first_ns = time_ns
        full_last_ns = time_ns

        if time_ns <= prefix_end_ns + tolerance:
            prefix_accumulator.update(coordinates)

            if prefix_first_ns is None:
                prefix_first_ns = time_ns
            prefix_last_ns = time_ns

    if prefix_first_ns is None or prefix_last_ns is None:
        raise RuntimeError(
            f"R{replica}: no se encontraron frames del prefijo."
        )

    if full_first_ns is None or full_last_ns is None:
        raise RuntimeError(
            f"R{replica}: no se encontraron frames completos."
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
        base = {
            "system": "1k5n_A",
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
                **base,
                "window": "prefix",
                "start_ns": prefix_first_ns,
                "end_ns": prefix_last_ns,
                "n_frames": prefix_accumulator.n_samples,
                "rmsf_A": float(prefix_rmsf[position]),
            }
        )

        records.append(
            {
                **base,
                "window": "full",
                "start_ns": full_first_ns,
                "end_ns": full_last_ns,
                "n_frames": full_accumulator.n_samples,
                "rmsf_A": float(full_rmsf[position]),
            }
        )

    comparison_errors = prefix_rmsf - full_rmsf

    pearson_value = float(
        pearsonr(prefix_rmsf, full_rmsf).statistic
    )
    spearman_value = float(
        spearmanr(prefix_rmsf, full_rmsf).statistic
    )

    summary = {
        "system": "1k5n_A",
        "replica": replica,
        "n_residues": int(calphas.n_atoms),
        "prefix_start_ns": prefix_first_ns,
        "prefix_end_ns": prefix_last_ns,
        "prefix_n_frames": prefix_accumulator.n_samples,
        "full_start_ns": full_first_ns,
        "full_end_ns": full_last_ns,
        "full_n_frames": full_accumulator.n_samples,
        "prefix_mean_rmsf_A": float(
            np.mean(prefix_rmsf)
        ),
        "full_mean_rmsf_A": float(
            np.mean(full_rmsf)
        ),
        "prefix_minus_full_bias_A": float(
            np.mean(comparison_errors)
        ),
        "prefix_full_mae_A": float(
            np.mean(np.abs(comparison_errors))
        ),
        "prefix_full_rmse_A": float(
            np.sqrt(np.mean(comparison_errors**2))
        ),
        "prefix_full_max_abs_error_A": float(
            np.max(np.abs(comparison_errors))
        ),
        "prefix_full_pearson_r": pearson_value,
        "prefix_full_spearman_r": spearman_value,
    }

    return pd.DataFrame(records), summary


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
            replica_count=("replica", "nunique"),
            rmsf_mean_A=("rmsf_A", "mean"),
            rmsf_median_A=("rmsf_A", "median"),
            rmsf_std_A=("rmsf_A", "std"),
            rmsf_min_A=("rmsf_A", "min"),
            rmsf_max_A=("rmsf_A", "max"),
        )
    )

    prefix = grouped.loc[
        grouped["window"] == "prefix"
    ].copy()

    full = grouped.loc[
        grouped["window"] == "full"
    ].copy()

    prefix = prefix.drop(columns=["window"])
    full = full.drop(columns=["window"])

    prefix = prefix.rename(
        columns={
            column: f"prefix_{column}"
            for column in prefix.columns
            if column
            not in {
                "system",
                "residue_index",
                "topology_resid",
                "resname",
            }
        }
    )

    full = full.rename(
        columns={
            column: f"full_{column}"
            for column in full.columns
            if column
            not in {
                "system",
                "residue_index",
                "topology_resid",
                "resname",
            }
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
        consensus["median_difference_A"].abs()
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
            np.sqrt(np.mean(errors**2))
        ),
        "median_profile_max_abs_error_A": float(
            np.max(np.abs(errors))
        ),
        "median_profile_pearson_r": float(
            pearsonr(prefix, full).statistic
        ),
        "median_profile_spearman_r": float(
            spearmanr(prefix, full).statistic
        ),
    }


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    replica_dataframes: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for replica in (1, 2, 3):
        dataframe, summary = process_replica(
            input_dir=input_dir,
            replica=replica,
            start_ns=args.start_ns,
            prefix_end_ns=args.prefix_end_ns,
        )

        replica_dataframes.append(dataframe)
        summaries.append(summary)

    long_dataframe = pd.concat(
        replica_dataframes,
        ignore_index=True,
    )

    summary_dataframe = pd.DataFrame(summaries)
    consensus = build_consensus(long_dataframe)
    aggregate_summary = consensus_summary(consensus)

    long_csv = output_dir / "rmsf_prefix_full_long.csv"
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

    long_dataframe.to_csv(long_csv, index=False)
    summary_dataframe.to_csv(summary_csv, index=False)
    consensus.to_csv(consensus_csv, index=False)

    metadata_json.write_text(
        json.dumps(
            {
                "canonical_protocol": {
                    "selection": "protein and name CA",
                    "alignment": "none",
                    "start_ns": args.start_ns,
                    "prefix_end_ns": args.prefix_end_ns,
                    "full_end_ns": 100.0,
                    "rmsf_reference": (
                        "mean position in each window"
                    ),
                    "variance_denominator": "N",
                },
                "replicas": summaries,
                "consensus": aggregate_summary,
                "leakage_policy": {
                    "full_profile": (
                        "diagnostic and target analysis only"
                    ),
                    "prefix_profile": (
                        "eligible for input feature and "
                        "core detection"
                    ),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("RMSF: PREFIJO FRENTE A TRAYECTORIA COMPLETA")
    print("=" * 100)
    print(
        summary_dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 100)
    print("CONSENSO ENTRE RÉPLICAS")
    print("=" * 100)

    for key, value in aggregate_summary.items():
        if isinstance(value, float):
            print(f"{key:<38} {value:.6f}")
        else:
            print(f"{key:<38} {value}")

    print()
    print(f"Perfil largo:  {long_csv}")
    print(f"Resumen:       {summary_csv}")
    print(f"Consenso:      {consensus_csv}")
    print(f"Metadatos:     {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
