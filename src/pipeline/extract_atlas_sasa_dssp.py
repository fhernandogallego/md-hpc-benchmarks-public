#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrae SASA y estructura secundaria DSSP "
            "para un sistema ATLAS."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--dataset-group", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--replicas", default="1,2,3")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--sphere-points", type=int, default=960)
    parser.add_argument("--probe-radius-nm", type=float, default=0.14)
    parser.add_argument("--prefix-end-ns", type=float, default=20.0)
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


def safe_std(
    values: np.ndarray,
    ddof: int,
) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) <= ddof:
        return 0.0

    return float(np.std(values, ddof=ddof))


def validate_time_grid(
    time_ps: np.ndarray,
    expected_step_ps: float,
    expected_duration_ns: float,
) -> dict[str, float | int]:
    time_ps = np.asarray(time_ps, dtype=float)

    if len(time_ps) < 2:
        raise RuntimeError(
            "La trayectoria reducida tiene menos de dos frames."
        )

    increments = np.diff(time_ps)

    if np.any(increments <= 0):
        raise RuntimeError(
            "Los tiempos no son estrictamente crecientes."
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
            f"Paso observado: {median_step_ps:.8f} ps; "
            f"esperado: {expected_step_ps:.8f} ps."
        )

    duration_ns = float(
        (time_ps[-1] - time_ps[0]) / 1000.0
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
        "n_frames": int(len(time_ps)),
        "start_ps": float(time_ps[0]),
        "end_ps": float(time_ps[-1]),
        "duration_ns": duration_ns,
        "median_step_ps": median_step_ps,
        "maximum_step_error_ps": maximum_step_error_ps,
    }


def summarize_window(
    dataframe: pd.DataFrame,
    system: str,
    replica: int,
    window: str,
) -> dict[str, object]:
    if dataframe.empty:
        raise RuntimeError(
            f"{system} R{replica}: ventana {window} vacía."
        )

    return {
        "system": system,
        "replica": replica,
        "window": window,
        "n_frames": int(len(dataframe)),
        "start_ns": float(dataframe["time_ns"].iloc[0]),
        "end_ns": float(dataframe["time_ns"].iloc[-1]),
        "total_sasa_mean_A2": float(
            dataframe["total_sasa_A2"].mean()
        ),
        "total_sasa_std_A2": float(
            dataframe["total_sasa_A2"].std(ddof=1)
        ),
        "helix_fraction_mean": float(
            dataframe["helix_fraction"].mean()
        ),
        "strand_fraction_mean": float(
            dataframe["strand_fraction"].mean()
        ),
        "coil_fraction_mean": float(
            dataframe["coil_fraction"].mean()
        ),
        "structured_fraction_mean": float(
            dataframe["structured_fraction"].mean()
        ),
    }


def residue_profiles(
    system: str,
    replica: int,
    residue_metadata: list[dict[str, object]],
    sasa_A2: np.ndarray,
    dssp: np.ndarray,
    prefix_mask: np.ndarray,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    full_mask = np.ones(
        len(prefix_mask),
        dtype=bool,
    )

    for residue_position, metadata in enumerate(
        residue_metadata
    ):
        prefix_sasa = sasa_A2[
            prefix_mask,
            residue_position,
        ]
        full_sasa = sasa_A2[
            full_mask,
            residue_position,
        ]

        prefix_dssp = dssp[
            prefix_mask,
            residue_position,
        ]
        full_dssp = dssp[
            full_mask,
            residue_position,
        ]

        records.append(
            {
                "system": system,
                "replica": replica,
                **metadata,
                "prefix_sasa_mean_A2": float(
                    np.mean(prefix_sasa)
                ),
                "prefix_sasa_std_A2": safe_std(
                    prefix_sasa,
                    ddof=0,
                ),
                "full_sasa_mean_A2": float(
                    np.mean(full_sasa)
                ),
                "full_sasa_std_A2": safe_std(
                    full_sasa,
                    ddof=0,
                ),
                "prefix_helix_occupancy": float(
                    np.mean(prefix_dssp == "H")
                ),
                "prefix_strand_occupancy": float(
                    np.mean(prefix_dssp == "E")
                ),
                "prefix_coil_occupancy": float(
                    np.mean(prefix_dssp == "C")
                ),
                "full_helix_occupancy": float(
                    np.mean(full_dssp == "H")
                ),
                "full_strand_occupancy": float(
                    np.mean(full_dssp == "E")
                ),
                "full_coil_occupancy": float(
                    np.mean(full_dssp == "C")
                ),
            }
        )

    return records


def process_replica(
    system: str,
    replica: int,
    topology_file: Path,
    trajectory_file: Path,
    output_dir: Path,
    stride: int,
    chunk_size: int,
    sphere_points: int,
    probe_radius_nm: float,
    prefix_end_ns: float,
    expected_native_step_ps: float,
    expected_duration_ns: float,
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    dict[str, object],
]:
    if not topology_file.is_file():
        raise FileNotFoundError(topology_file)

    if not trajectory_file.is_file():
        raise FileNotFoundError(trajectory_file)

    time_chunks: list[np.ndarray] = []
    sasa_chunks: list[np.ndarray] = []
    dssp_chunks: list[np.ndarray] = []

    residue_metadata: list[dict[str, object]] | None = None
    processed_chunks = 0

    for trajectory_chunk in md.iterload(
        str(trajectory_file),
        top=str(topology_file),
        chunk=chunk_size,
        stride=stride,
    ):
        if trajectory_chunk.n_frames == 0:
            continue

        if residue_metadata is None:
            residue_metadata = []

            for residue_position, residue in enumerate(
                trajectory_chunk.topology.residues,
                start=1,
            ):
                residue_sequence = residue.resSeq

                if residue_sequence is None:
                    residue_sequence = residue_position

                residue_metadata.append(
                    {
                        "residue_index": residue_position,
                        "topology_resid": int(
                            residue_sequence
                        ),
                        "resname": str(residue.name),
                    }
                )

        sasa_nm2 = md.shrake_rupley(
            trajectory_chunk,
            probe_radius=probe_radius_nm,
            n_sphere_points=sphere_points,
            mode="residue",
        )

        dssp = md.compute_dssp(
            trajectory_chunk,
            simplified=True,
        )

        if sasa_nm2.shape != dssp.shape:
            raise RuntimeError(
                f"{system} R{replica}: SASA y DSSP "
                f"tienen formas diferentes: "
                f"{sasa_nm2.shape} y {dssp.shape}."
            )

        if residue_metadata is None:
            raise RuntimeError(
                "No se pudo obtener la topología."
            )

        if sasa_nm2.shape[1] != len(residue_metadata):
            raise RuntimeError(
                f"{system} R{replica}: número de residuos "
                "inconsistente."
            )

        time_chunks.append(
            np.asarray(
                trajectory_chunk.time,
                dtype=np.float64,
            )
        )
        sasa_chunks.append(
            np.asarray(
                sasa_nm2,
                dtype=np.float64,
            )
            * 100.0
        )
        dssp_chunks.append(
            np.asarray(dssp)
        )

        processed_chunks += 1

    if not time_chunks:
        raise RuntimeError(
            f"{system} R{replica}: no se cargaron frames."
        )

    if residue_metadata is None:
        raise RuntimeError(
            f"{system} R{replica}: no se obtuvo la topología."
        )

    time_ps = np.concatenate(
        time_chunks,
        axis=0,
    )
    sasa_A2 = np.concatenate(
        sasa_chunks,
        axis=0,
    )
    dssp = np.concatenate(
        dssp_chunks,
        axis=0,
    )

    expected_step_ps = (
        expected_native_step_ps * stride
    )

    temporal_validation = validate_time_grid(
        time_ps=time_ps,
        expected_step_ps=expected_step_ps,
        expected_duration_ns=expected_duration_ns,
    )

    total_sasa_A2 = np.sum(
        sasa_A2,
        axis=1,
    )
    mean_residue_sasa_A2 = np.mean(
        sasa_A2,
        axis=1,
    )

    helix_fraction = np.mean(
        dssp == "H",
        axis=1,
    )
    strand_fraction = np.mean(
        dssp == "E",
        axis=1,
    )
    coil_fraction = np.mean(
        dssp == "C",
        axis=1,
    )
    structured_fraction = (
        helix_fraction + strand_fraction
    )

    composition_sum = (
        helix_fraction
        + strand_fraction
        + coil_fraction
    )

    maximum_composition_error = float(
        np.max(
            np.abs(
                composition_sum - 1.0
            )
        )
    )

    if maximum_composition_error > 1e-10:
        raise RuntimeError(
            f"{system} R{replica}: H+E+C no suma uno. "
            f"Error máximo={maximum_composition_error:.3e}"
        )

    time_ns = time_ps / 1000.0
    prefix_mask = (
        time_ns <= prefix_end_ns + 1e-9
    )

    if int(prefix_mask.sum()) != (
        int(
            round(
                prefix_end_ns * 1000.0
                / expected_step_ps
            )
        )
        + 1
    ):
        raise RuntimeError(
            f"{system} R{replica}: número inesperado "
            f"de frames del prefijo: {prefix_mask.sum()}."
        )

    dataframe = pd.DataFrame(
        {
            "system": system,
            "replica": replica,
            "frame": np.arange(
                len(time_ps),
                dtype=int,
            ),
            "time_ps": time_ps,
            "time_ns": time_ns,
            "total_sasa_A2": total_sasa_A2,
            "mean_residue_sasa_A2": (
                mean_residue_sasa_A2
            ),
            "helix_fraction": helix_fraction,
            "strand_fraction": strand_fraction,
            "coil_fraction": coil_fraction,
            "structured_fraction": (
                structured_fraction
            ),
            "is_prefix": prefix_mask,
        }
    )

    output_csv = (
        output_dir
        / (
            f"{system}_R{replica}"
            "_sasa_dssp_100ps.csv"
        )
    )

    dataframe.to_csv(
        output_csv,
        index=False,
    )

    profile_records = residue_profiles(
        system=system,
        replica=replica,
        residue_metadata=residue_metadata,
        sasa_A2=sasa_A2,
        dssp=dssp,
        prefix_mask=prefix_mask,
    )

    metadata = {
        "system": system,
        "replica": replica,
        "topology_file": str(topology_file),
        "trajectory_file": str(trajectory_file),
        "output_csv": str(output_csv),
        "processed_chunks": processed_chunks,
        "n_residues": len(residue_metadata),
        "prefix_n_frames": int(
            prefix_mask.sum()
        ),
        "full_n_frames": int(
            len(time_ps)
        ),
        "maximum_composition_error": (
            maximum_composition_error
        ),
        **temporal_validation,
    }

    return dataframe, profile_records, metadata


def main() -> int:
    args = parse_args()

    replicas = parse_replicas(
        args.replicas
    )

    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()

    protein_dir = (
        raw_root
        / args.dataset_group
        / args.system
        / "unpacked"
        / f"{args.system}_protein"
    )

    topology_file = (
        protein_dir
        / f"{args.system}.pdb"
    )

    output_dir = (
        processed_root
        / args.dataset_group
        / args.system
        / "sasa_dssp"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_records: list[dict[str, object]] = []
    profile_records: list[dict[str, object]] = []
    replica_metadata: list[dict[str, object]] = []

    for replica in replicas:
        trajectory_file = (
            protein_dir
            / (
                f"{args.system}_prod_"
                f"R{replica}_fit.xtc"
            )
        )

        dataframe, current_profiles, metadata = (
            process_replica(
                system=args.system,
                replica=replica,
                topology_file=topology_file,
                trajectory_file=trajectory_file,
                output_dir=output_dir,
                stride=args.stride,
                chunk_size=args.chunk_size,
                sphere_points=args.sphere_points,
                probe_radius_nm=args.probe_radius_nm,
                prefix_end_ns=args.prefix_end_ns,
                expected_native_step_ps=(
                    args.expected_native_step_ps
                ),
                expected_duration_ns=(
                    args.expected_duration_ns
                ),
            )
        )

        full = dataframe
        prefix = dataframe.loc[
            dataframe["is_prefix"]
        ]

        summary_records.append(
            summarize_window(
                dataframe=full,
                system=args.system,
                replica=replica,
                window="full",
            )
        )
        summary_records.append(
            summarize_window(
                dataframe=prefix,
                system=args.system,
                replica=replica,
                window="prefix",
            )
        )

        profile_records.extend(
            current_profiles
        )
        replica_metadata.append(metadata)

        print("=" * 72)
        print(f"RÉPLICA R{replica}")
        print(f"Frames completos: {len(full)}")
        print(f"Frames prefijo:   {len(prefix)}")
        print(
            f"SASA media:       "
            f"{full['total_sasa_A2'].mean():.2f} Å²"
        )
        print(
            "H/E/C medios:     "
            f"{full['helix_fraction'].mean():.4f} / "
            f"{full['strand_fraction'].mean():.4f} / "
            f"{full['coil_fraction'].mean():.4f}"
        )

    summary = pd.DataFrame(
        summary_records
    )
    profiles = pd.DataFrame(
        profile_records
    )

    summary_csv = (
        output_dir
        / "sasa_dssp_summary.csv"
    )
    profiles_csv = (
        output_dir
        / "sasa_dssp_residue_profiles.csv"
    )
    metadata_json = (
        output_dir
        / "sasa_dssp_metadata.json"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )
    profiles.to_csv(
        profiles_csv,
        index=False,
    )

    metadata_json.write_text(
        json.dumps(
            {
                "system": args.system,
                "dataset_group": args.dataset_group,
                "replicas": replicas,
                "stride": args.stride,
                "chunk_size": args.chunk_size,
                "sphere_points": args.sphere_points,
                "probe_radius_nm": (
                    args.probe_radius_nm
                ),
                "prefix_end_ns": (
                    args.prefix_end_ns
                ),
                "sasa_mode": "residue",
                "sasa_output_unit": "A2",
                "dssp_simplified": True,
                "replica_metadata": (
                    replica_metadata
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("RESUMEN SASA Y DSSP")
    print("=" * 120)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()
    print(f"Resumen:   {summary_csv}")
    print(f"Residuos:  {profiles_csv}")
    print(f"Metadatos: {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
