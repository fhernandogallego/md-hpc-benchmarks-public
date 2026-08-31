#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import align


START_PROTOCOLS = {
    1: "from_0.1ns",
    2: "after_0.1ns",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara varias implementaciones RMSF con ATLAS."
    )
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def calculate_rmsf(coordinates: np.ndarray) -> np.ndarray:
    if coordinates.ndim != 3:
        raise ValueError(
            "Las coordenadas deben tener forma frames x atoms x 3."
        )

    average = coordinates.mean(axis=0)
    displacements = coordinates - average

    return np.sqrt(
        np.mean(
            np.sum(displacements**2, axis=2),
            axis=0,
        )
    )


def official_column(
    dataframe: pd.DataFrame,
    replica: int,
) -> str:
    candidates = [
        column
        for column in dataframe.columns[1:]
        if f"R{replica}" in column.upper()
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"No se pudo identificar la columna oficial de R{replica}. "
            f"Candidatas: {candidates}"
        )

    return candidates[0]


def compare(
    official: np.ndarray,
    calculated: np.ndarray,
) -> dict[str, float | int]:
    official = np.asarray(official, dtype=float)
    calculated = np.asarray(calculated, dtype=float)

    if official.shape != calculated.shape:
        raise ValueError(
            f"Dimensiones diferentes: {official.shape} "
            f"y {calculated.shape}"
        )

    valid = np.isfinite(official) & np.isfinite(calculated)
    official = official[valid]
    calculated = calculated[valid]

    errors = calculated - official

    if len(official) > 1:
        correlation = float(
            np.corrcoef(official, calculated)[0, 1]
        )
    else:
        correlation = float("nan")

    return {
        "n_residues": int(len(official)),
        "official_mean_A": float(np.mean(official)),
        "calculated_mean_A": float(np.mean(calculated)),
        "bias_A": float(np.mean(errors)),
        "mae_A": float(np.mean(np.abs(errors))),
        "rmse_A": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error_A": float(np.max(np.abs(errors))),
        "pearson_r": correlation,
    }


def collect_coordinates(
    universe: mda.Universe,
    reference: mda.Universe,
    start_frame: int,
) -> tuple[dict[str, np.ndarray], float, int]:
    calphas = universe.select_atoms("protein and name CA")
    reference_cas = reference.select_atoms("protein and name CA")

    if calphas.n_atoms != reference_cas.n_atoms:
        raise RuntimeError(
            "La selección CA móvil y de referencia no coincide."
        )

    direct_coordinates: list[np.ndarray] = []
    equal_fit_coordinates: list[np.ndarray] = []
    mass_fit_coordinates: list[np.ndarray] = []

    first_time_ns: float | None = None

    for frame_index in range(
        start_frame,
        len(universe.trajectory),
    ):
        universe.trajectory[frame_index]

        if first_time_ns is None:
            first_time_ns = float(
                universe.trajectory.ts.time / 1000.0
            )

        direct_coordinates.append(
            calphas.positions.copy()
        )

        align.alignto(
            universe,
            reference,
            select="protein and name CA",
            weights=None,
        )
        equal_fit_coordinates.append(
            calphas.positions.copy()
        )

        # Recargar el frame original antes del ajuste ponderado.
        universe.trajectory[frame_index]

        align.alignto(
            universe,
            reference,
            select="protein and name CA",
            weights="mass",
        )
        mass_fit_coordinates.append(
            calphas.positions.copy()
        )

    if first_time_ns is None:
        raise RuntimeError("No se procesaron frames.")

    return (
        {
            "direct": np.asarray(
                direct_coordinates,
                dtype=np.float64,
            ),
            "fit_ca_equal": np.asarray(
                equal_fit_coordinates,
                dtype=np.float64,
            ),
            "fit_ca_mass": np.asarray(
                mass_fit_coordinates,
                dtype=np.float64,
            ),
        },
        first_time_ns,
        len(direct_coordinates),
    )


def main() -> int:
    args = parse_args()

    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_file = analysis_dir / "1k5n_A.pdb"
    official_file = analysis_dir / "1k5n_A_RMSF.tsv"

    official_dataframe = pd.read_csv(
        official_file,
        sep=r"\s+",
    )

    residue_axis_column = official_dataframe.columns[0]

    summary_records: list[dict[str, object]] = []
    per_residue_records: list[dict[str, object]] = []

    for replica in (1, 2, 3):
        trajectory_file = (
            analysis_dir / f"1k5n_A_R{replica}.xtc"
        )

        universe = mda.Universe(
            str(pdb_file),
            str(trajectory_file),
        )
        reference = mda.Universe(str(pdb_file))

        calphas = universe.select_atoms(
            "protein and name CA"
        )

        official_name = official_column(
            official_dataframe,
            replica,
        )
        official_values = official_dataframe[
            official_name
        ].to_numpy(dtype=float)

        if len(official_values) != calphas.n_atoms:
            raise RuntimeError(
                f"R{replica}: ATLAS tiene "
                f"{len(official_values)} posiciones y la "
                f"topología {calphas.n_atoms} carbonos alfa."
            )

        residue_ids = calphas.resids.astype(int)
        residue_names = calphas.resnames.astype(str)
        residue_indices = np.arange(
            1,
            calphas.n_atoms + 1,
        )

        for start_frame, start_name in START_PROTOCOLS.items():
            coordinate_sets, first_time_ns, n_frames = (
                collect_coordinates(
                    universe=universe,
                    reference=reference,
                    start_frame=start_frame,
                )
            )

            for alignment_name, coordinates in (
                coordinate_sets.items()
            ):
                protocol = (
                    f"{alignment_name}_{start_name}"
                )

                calculated_values = calculate_rmsf(
                    coordinates
                )

                metrics = compare(
                    official=official_values,
                    calculated=calculated_values,
                )

                summary_records.append(
                    {
                        "system": "1k5n_A",
                        "replica": replica,
                        "protocol": protocol,
                        "alignment": alignment_name,
                        "start_rule": start_name,
                        "start_frame": start_frame,
                        "first_time_ns": first_time_ns,
                        "n_frames": n_frames,
                        **metrics,
                    }
                )

                for position in range(
                    calphas.n_atoms
                ):
                    per_residue_records.append(
                        {
                            "system": "1k5n_A",
                            "replica": replica,
                            "protocol": protocol,
                            "residue_index": int(
                                residue_indices[position]
                            ),
                            "topology_resid": int(
                                residue_ids[position]
                            ),
                            "resname": str(
                                residue_names[position]
                            ),
                            "official_axis": (
                                official_dataframe[
                                    residue_axis_column
                                ].iloc[position]
                            ),
                            "official_rmsf_A": float(
                                official_values[position]
                            ),
                            "calculated_rmsf_A": float(
                                calculated_values[position]
                            ),
                            "difference_A": float(
                                calculated_values[position]
                                - official_values[position]
                            ),
                            "absolute_error_A": float(
                                abs(
                                    calculated_values[position]
                                    - official_values[position]
                                )
                            ),
                        }
                    )

    summary = pd.DataFrame(summary_records)
    per_residue = pd.DataFrame(per_residue_records)

    aggregate = (
        summary.groupby(
            [
                "protocol",
                "alignment",
                "start_rule",
            ],
            as_index=False,
        )
        .agg(
            replicas=("replica", "nunique"),
            mean_mae_A=("mae_A", "mean"),
            max_mae_A=("mae_A", "max"),
            mean_rmse_A=("rmse_A", "mean"),
            mean_bias_A=("bias_A", "mean"),
            min_pearson_r=("pearson_r", "min"),
            mean_pearson_r=("pearson_r", "mean"),
        )
        .sort_values(
            ["mean_mae_A", "mean_rmse_A"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    best_protocol = str(
        aggregate.iloc[0]["protocol"]
    )

    best_profiles = per_residue.loc[
        per_residue["protocol"] == best_protocol
    ].copy()

    summary_file = output_dir / "rmsf_validation_summary.csv"
    aggregate_file = (
        output_dir / "rmsf_protocol_ranking.csv"
    )
    profiles_file = (
        output_dir / "rmsf_all_protocols_long.csv"
    )
    best_file = (
        output_dir / "rmsf_best_protocol_long.csv"
    )
    metadata_file = output_dir / "rmsf_validation.json"

    summary.to_csv(summary_file, index=False)
    aggregate.to_csv(aggregate_file, index=False)
    per_residue.to_csv(profiles_file, index=False)
    best_profiles.to_csv(best_file, index=False)

    metadata_file.write_text(
        json.dumps(
            {
                "best_protocol": best_protocol,
                "ranking": aggregate.to_dict(
                    orient="records"
                ),
                "rules": {
                    "rmsf_atoms": "protein and name CA",
                    "official_source": str(
                        official_file
                    ),
                    "tested_start_frames": (
                        START_PROTOCOLS
                    ),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("RANKING DE PROTOCOLOS RMSF")
    print("=" * 100)
    print(
        aggregate.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 100)
    print("RESULTADOS POR RÉPLICA")
    print("=" * 100)
    print(
        summary[
            [
                "replica",
                "protocol",
                "n_frames",
                "first_time_ns",
                "mae_A",
                "rmse_A",
                "bias_A",
                "pearson_r",
            ]
        ]
        .sort_values(
            ["replica", "mae_A"]
        )
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"Mejor protocolo: {best_protocol}")
    print(f"Resumen:         {summary_file}")
    print(f"Ranking:         {aggregate_file}")
    print(f"Mejor perfil:    {best_file}")
    print(f"Metadatos:       {metadata_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
