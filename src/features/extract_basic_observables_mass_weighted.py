#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import rms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae RMSD y radio de giro de las trayectorias completas de ATLAS."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directorio unpacked de la entrada ATLAS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio de resultados.",
    )
    parser.add_argument(
        "--prefix-ns",
        type=float,
        default=20.0,
        help="Duración del prefijo que utilizará el modelo.",
    )
    return parser.parse_args()


def trajectory_paths(input_dir: Path, replica: int) -> tuple[Path, Path]:
    protein_dir = input_dir / "1k5n_A_protein"

    topology = protein_dir / "1k5n_A.pdb"
    trajectory = protein_dir / f"1k5n_A_prod_R{replica}_fit.xtc"

    return topology, trajectory


def calculate_replica(
    input_dir: Path,
    output_dir: Path,
    replica: int,
    prefix_ns: float,
) -> dict[str, object]:
    topology, trajectory = trajectory_paths(input_dir, replica)

    if not topology.is_file():
        raise FileNotFoundError(f"No existe la topología: {topology}")

    if not trajectory.is_file():
        raise FileNotFoundError(f"No existe la trayectoria: {trajectory}")

    print("=" * 72)
    print(f"RÉPLICA R{replica}")
    print(f"Topología:   {topology}")
    print(f"Trayectoria: {trajectory}")

    universe = mda.Universe(str(topology), str(trajectory))

    backbone = universe.select_atoms("protein and backbone")
    protein = universe.select_atoms("protein")

    if backbone.n_atoms == 0:
        raise RuntimeError("La selección de backbone está vacía.")

    if protein.n_atoms == 0:
        raise RuntimeError("La selección de proteína está vacía.")

    if not hasattr(protein, "masses"):
        raise RuntimeError("La topología no contiene masas atómicas.")

    masses = np.asarray(protein.masses, dtype=float)

    if not np.all(np.isfinite(masses)):
        raise RuntimeError("Hay masas atómicas no finitas.")

    if np.any(masses <= 0):
        bad_count = int(np.sum(masses <= 0))
        raise RuntimeError(
            f"Hay {bad_count} átomos con masa menor o igual que cero."
        )

    print(f"Átomos proteína: {protein.n_atoms}")
    print(f"Átomos backbone: {backbone.n_atoms}")
    print(f"Frames:          {len(universe.trajectory)}")
    print(f"dt:              {universe.trajectory.dt} ps")

    # RMSD con ajuste por mínimos cuadrados del backbone
    # respecto al primer frame.
    rmsd_analysis = rms.RMSD(
        universe,
        reference=universe,
        select="protein and backbone",
        weights="mass",
        ref_frame=0,
    )

    rmsd_analysis.run(verbose=False)

    rmsd_values = rmsd_analysis.results.rmsd

    # Columnas de MDAnalysis:
    # 0 = frame, 1 = tiempo en ps, 2 = RMSD en Å.
    rmsd_frames = rmsd_values[:, 0].astype(int)
    rmsd_times_ps = rmsd_values[:, 1].astype(float)
    rmsd_angstrom = rmsd_values[:, 2].astype(float)

    # Radio de giro para todos los átomos de la proteína.
    rg_angstrom: list[float] = []
    rg_frames: list[int] = []
    rg_times_ps: list[float] = []

    for timestep in universe.trajectory:
        rg_frames.append(int(timestep.frame))
        rg_times_ps.append(float(timestep.time))
        rg_angstrom.append(float(protein.radius_of_gyration()))

    rg_frames_array = np.asarray(rg_frames, dtype=int)
    rg_times_ps_array = np.asarray(rg_times_ps, dtype=float)
    rg_angstrom_array = np.asarray(rg_angstrom, dtype=float)

    if not np.array_equal(rmsd_frames, rg_frames_array):
        raise RuntimeError("Los frames de RMSD y radio de giro no coinciden.")

    if not np.allclose(rmsd_times_ps, rg_times_ps_array):
        raise RuntimeError("Los tiempos de RMSD y radio de giro no coinciden.")

    time_ns = rmsd_times_ps / 1000.0
    prefix_mask = time_ns <= prefix_ns + 1e-9

    dataframe = pd.DataFrame(
        {
            "replica": replica,
            "frame": rmsd_frames,
            "time_ps": rmsd_times_ps,
            "time_ns": time_ns,
            "rmsd_backbone_A": rmsd_angstrom,
            "radius_gyration_A": rg_angstrom_array,
            "is_prefix": prefix_mask,
        }
    )

    output_csv = output_dir / f"1k5n_A_R{replica}_basic_observables.csv"
    dataframe.to_csv(output_csv, index=False)

    # Versión a 100 ps para compararla con el paquete analysis.
    comparison_dataframe = dataframe.loc[
        dataframe["frame"] % 10 == 0
    ].copy()

    comparison_csv = (
        output_dir
        / f"1k5n_A_R{replica}_basic_observables_100ps.csv"
    )
    comparison_dataframe.to_csv(comparison_csv, index=False)

    full_summary = {
        "rmsd_mean_A": float(dataframe["rmsd_backbone_A"].mean()),
        "rmsd_std_A": float(dataframe["rmsd_backbone_A"].std(ddof=1)),
        "rmsd_min_A": float(dataframe["rmsd_backbone_A"].min()),
        "rmsd_max_A": float(dataframe["rmsd_backbone_A"].max()),
        "rg_mean_A": float(dataframe["radius_gyration_A"].mean()),
        "rg_std_A": float(dataframe["radius_gyration_A"].std(ddof=1)),
        "rg_min_A": float(dataframe["radius_gyration_A"].min()),
        "rg_max_A": float(dataframe["radius_gyration_A"].max()),
    }

    prefix = dataframe.loc[prefix_mask]

    prefix_summary = {
        "n_frames": int(len(prefix)),
        "last_time_ns": float(prefix["time_ns"].iloc[-1]),
        "rmsd_mean_A": float(prefix["rmsd_backbone_A"].mean()),
        "rmsd_std_A": float(prefix["rmsd_backbone_A"].std(ddof=1)),
        "rg_mean_A": float(prefix["radius_gyration_A"].mean()),
        "rg_std_A": float(prefix["radius_gyration_A"].std(ddof=1)),
    }

    result = {
        "system": "1k5n_A",
        "replica": replica,
        "topology": str(topology),
        "trajectory": str(trajectory),
        "n_atoms": int(universe.atoms.n_atoms),
        "n_protein_atoms": int(protein.n_atoms),
        "n_backbone_atoms": int(backbone.n_atoms),
        "n_frames": int(len(dataframe)),
        "dt_ps": float(universe.trajectory.dt),
        "last_time_ns": float(dataframe["time_ns"].iloc[-1]),
        "prefix_ns": prefix_ns,
        "prefix": prefix_summary,
        "full_trajectory": full_summary,
        "output_csv": str(output_csv),
        "comparison_100ps_csv": str(comparison_csv),
    }

    print(f"Frames del prefijo: {len(prefix)}")
    print(f"RMSD medio total:   {full_summary['rmsd_mean_A']:.4f} Å")
    print(f"Rg medio total:     {full_summary['rg_mean_A']:.4f} Å")
    print(f"CSV:                {output_csv}")

    return result


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        print(
            f"ERROR: no existe el directorio: {input_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    for replica in (1, 2, 3):
        result = calculate_replica(
            input_dir=input_dir,
            output_dir=output_dir,
            replica=replica,
            prefix_ns=args.prefix_ns,
        )
        results.append(result)

    summary_json = output_dir / "basic_observables_summary.json"
    summary_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_rows: list[dict[str, object]] = []

    for result in results:
        full = result["full_trajectory"]
        prefix = result["prefix"]

        summary_rows.append(
            {
                "system": result["system"],
                "replica": result["replica"],
                "n_frames": result["n_frames"],
                "dt_ps": result["dt_ps"],
                "prefix_ns": result["prefix_ns"],
                "prefix_frames": prefix["n_frames"],
                "prefix_rmsd_mean_A": prefix["rmsd_mean_A"],
                "prefix_rg_mean_A": prefix["rg_mean_A"],
                "full_rmsd_mean_A": full["rmsd_mean_A"],
                "full_rmsd_std_A": full["rmsd_std_A"],
                "full_rg_mean_A": full["rg_mean_A"],
                "full_rg_std_A": full["rg_std_A"],
            }
        )

    summary_csv = output_dir / "basic_observables_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print()
    print("=" * 72)
    print("EXTRACCIÓN COMPLETADA")
    print("=" * 72)
    print(f"Réplicas procesadas: {len(results)}")
    print(f"Resumen JSON:        {summary_json}")
    print(f"Resumen CSV:         {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
