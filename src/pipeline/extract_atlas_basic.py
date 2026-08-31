#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import rms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrae RMSD de backbone y radio de giro "
            "para una réplica de un sistema ATLAS."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--dataset-group", required=True)
    parser.add_argument("--replica", required=True, type=int)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--prefix-end-ns",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--coarse-step-ps",
        type=float,
        default=100.0,
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


def validate_times(
    times_ps: np.ndarray,
    expected_step_ps: float,
    expected_duration_ns: float,
) -> dict[str, float | int]:
    if len(times_ps) < 2:
        raise RuntimeError(
            "La trayectoria contiene menos de dos frames."
        )

    increments = np.diff(times_ps)

    if np.any(increments <= 0):
        raise RuntimeError(
            "Los tiempos de la trayectoria no son crecientes."
        )

    median_step_ps = float(np.median(increments))
    maximum_step_error_ps = float(
        np.max(
            np.abs(
                increments - expected_step_ps
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
            f"Paso temporal observado: {median_step_ps:.8f} ps; "
            f"esperado: {expected_step_ps:.8f} ps."
        )

    duration_ns = float(
        (times_ps[-1] - times_ps[0]) / 1000.0
    )

    if not np.isclose(
        duration_ns,
        expected_duration_ns,
        rtol=0.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            f"Duración observada: {duration_ns:.8f} ns; "
            f"esperada: {expected_duration_ns:.8f} ns."
        )

    return {
        "n_frames": int(len(times_ps)),
        "start_time_ps": float(times_ps[0]),
        "end_time_ps": float(times_ps[-1]),
        "duration_ns": duration_ns,
        "median_step_ps": median_step_ps,
        "maximum_step_error_ps": maximum_step_error_ps,
    }


def coarse_mask(
    times_ps: np.ndarray,
    coarse_step_ps: float,
) -> np.ndarray:
    nearest_index = np.rint(
        times_ps / coarse_step_ps
    )

    expected_times = (
        nearest_index * coarse_step_ps
    )

    mask = np.isclose(
        times_ps,
        expected_times,
        rtol=0.0,
        atol=1e-4,
    )

    if not mask[0] or not mask[-1]:
        raise RuntimeError(
            "La rejilla reducida no contiene los extremos."
        )

    return mask


def main() -> int:
    args = parse_args()

    if args.replica < 1:
        raise ValueError(
            "El índice de réplica debe ser positivo."
        )

    unpacked_dir = (
        args.raw_root.resolve()
        / args.dataset_group
        / args.system
        / "unpacked"
    )
    protein_dir = (
        unpacked_dir
        / f"{args.system}_protein"
    )

    topology_file = (
        protein_dir
        / f"{args.system}.pdb"
    )
    trajectory_file = (
        protein_dir
        / (
            f"{args.system}_prod_"
            f"R{args.replica}_fit.xtc"
        )
    )

    if not topology_file.is_file():
        raise FileNotFoundError(topology_file)

    if not trajectory_file.is_file():
        raise FileNotFoundError(trajectory_file)

    output_dir = (
        args.output_root.resolve()
        / args.dataset_group
        / args.system
        / "basic"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = mda.Universe(
        str(topology_file),
        str(trajectory_file),
    )

    protein = universe.select_atoms("protein")
    backbone = universe.select_atoms(
        "protein and backbone"
    )
    calphas = universe.select_atoms(
        "protein and name CA"
    )

    if protein.n_atoms == 0:
        raise RuntimeError(
            "La selección protein está vacía."
        )

    if backbone.n_atoms == 0:
        raise RuntimeError(
            "La selección backbone está vacía."
        )

    if calphas.n_atoms == 0:
        raise RuntimeError(
            "La selección CA está vacía."
        )

    rmsd_analysis = rms.RMSD(
        universe,
        reference=universe,
        select="protein and backbone",
        ref_frame=0,
        weights=None,
    ).run()

    rmsd_results = np.asarray(
        rmsd_analysis.results.rmsd,
        dtype=np.float64,
    )

    frames = rmsd_results[:, 0].astype(int)
    times_ps = rmsd_results[:, 1]
    rmsd_backbone_A = rmsd_results[:, 2]

    radius_gyration_A = np.empty(
        len(universe.trajectory),
        dtype=np.float64,
    )

    trajectory_times_ps = np.empty(
        len(universe.trajectory),
        dtype=np.float64,
    )

    for position, timestep in enumerate(
        universe.trajectory
    ):
        trajectory_times_ps[position] = float(
            timestep.time
        )
        radius_gyration_A[position] = float(
            protein.radius_of_gyration()
        )

    if len(radius_gyration_A) != len(
        rmsd_backbone_A
    ):
        raise RuntimeError(
            "RMSD y radio de giro tienen longitudes diferentes."
        )

    if not np.allclose(
        trajectory_times_ps,
        times_ps,
        rtol=0.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Los tiempos de RMSD y Rg no coinciden."
        )

    if not np.all(
        np.isfinite(rmsd_backbone_A)
    ):
        raise RuntimeError(
            "El RMSD contiene valores no finitos."
        )

    if not np.all(
        np.isfinite(radius_gyration_A)
    ):
        raise RuntimeError(
            "El radio de giro contiene valores no finitos."
        )

    temporal_validation = validate_times(
        times_ps=times_ps,
        expected_step_ps=(
            args.expected_native_step_ps
        ),
        expected_duration_ns=(
            args.expected_duration_ns
        ),
    )

    time_ns = times_ps / 1000.0

    dataframe = pd.DataFrame(
        {
            "system": args.system,
            "replica": args.replica,
            "sample_id": (
                f"{args.system}_R{args.replica}"
            ),
            "frame": frames,
            "time_ps": times_ps,
            "time_ns": time_ns,
            "rmsd_backbone_A": (
                rmsd_backbone_A
            ),
            "radius_gyration_A": (
                radius_gyration_A
            ),
            "is_prefix": (
                time_ns
                <= args.prefix_end_ns + 1e-9
            ),
        }
    )

    reduced_mask = coarse_mask(
        times_ps=times_ps,
        coarse_step_ps=args.coarse_step_ps,
    )

    reduced = dataframe.loc[
        reduced_mask
    ].reset_index(drop=True)

    expected_reduced_frames = (
        int(
            round(
                args.expected_duration_ns
                * 1000.0
                / args.coarse_step_ps
            )
        )
        + 1
    )

    if len(reduced) != expected_reduced_frames:
        raise RuntimeError(
            "Número incorrecto de frames reducidos: "
            f"{len(reduced)}; "
            f"esperados: {expected_reduced_frames}."
        )

    full_csv = (
        output_dir
        / (
            f"{args.system}_R{args.replica}"
            "_basic_observables.csv"
        )
    )
    reduced_csv = (
        output_dir
        / (
            f"{args.system}_R{args.replica}"
            "_basic_observables_100ps.csv"
        )
    )
    summary_json = (
        output_dir
        / (
            f"{args.system}_R{args.replica}"
            "_basic_observables.json"
        )
    )

    dataframe.to_csv(full_csv, index=False)
    reduced.to_csv(reduced_csv, index=False)

    prefix_mask = dataframe["is_prefix"].to_numpy(
        dtype=bool
    )

    summary = {
        "system": args.system,
        "dataset_group": args.dataset_group,
        "replica": args.replica,
        "topology_file": str(topology_file),
        "trajectory_file": str(trajectory_file),
        "output_full_csv": str(full_csv),
        "output_100ps_csv": str(reduced_csv),
        "n_protein_atoms": int(protein.n_atoms),
        "n_backbone_atoms": int(
            backbone.n_atoms
        ),
        "n_ca_atoms": int(calphas.n_atoms),
        **temporal_validation,
        "coarse_step_ps": args.coarse_step_ps,
        "coarse_n_frames": int(len(reduced)),
        "prefix_end_ns": args.prefix_end_ns,
        "prefix_n_frames": int(
            prefix_mask.sum()
        ),
        "rmsd_mean_A": float(
            np.mean(rmsd_backbone_A)
        ),
        "rmsd_prefix_mean_A": float(
            np.mean(
                rmsd_backbone_A[prefix_mask]
            )
        ),
        "rmsd_final_A": float(
            rmsd_backbone_A[-1]
        ),
        "rg_mean_A": float(
            np.mean(radius_gyration_A)
        ),
        "rg_prefix_mean_A": float(
            np.mean(
                radius_gyration_A[prefix_mask]
            )
        ),
        "rg_final_A": float(
            radius_gyration_A[-1]
        ),
        "canonical_rmsd": {
            "selection": (
                "protein and backbone"
            ),
            "reference_frame": 0,
            "weights": None,
            "superposition": True,
        },
        "canonical_rg": {
            "selection": "protein",
            "weights": "atomic masses",
        },
    }

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 82)
    print("EXTRACCIÓN BÁSICA PARAMETRIZADA")
    print("=" * 82)
    print(f"Sistema:           {args.system}")
    print(f"Grupo:             {args.dataset_group}")
    print(f"Réplica:           R{args.replica}")
    print(f"Átomos proteína:   {protein.n_atoms}")
    print(f"Átomos backbone:   {backbone.n_atoms}")
    print(f"Residuos CA:       {calphas.n_atoms}")
    print(f"Frames nativos:    {len(dataframe)}")
    print(f"Frames 100 ps:     {len(reduced)}")
    print(
        f"Tiempo:            "
        f"{time_ns[0]:.1f}–{time_ns[-1]:.1f} ns"
    )
    print(
        f"RMSD medio:        "
        f"{np.mean(rmsd_backbone_A):.6f} Å"
    )
    print(
        f"Rg medio:          "
        f"{np.mean(radius_gyration_A):.6f} Å"
    )
    print(f"CSV 10 ps:         {full_csv}")
    print(f"CSV 100 ps:        {reduced_csv}")
    print(f"Metadatos:         {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
