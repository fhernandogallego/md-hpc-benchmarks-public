#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd


SS_CODES = ("H", "E", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae SASA y estructura secundaria de ATLAS."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--sphere-points", type=int, default=960)
    parser.add_argument("--prefix-end-ns", type=float, default=20.0)
    return parser.parse_args()


def window_summary(
    dataframe: pd.DataFrame,
    replica: int,
    window: str,
) -> dict[str, object]:
    return {
        "system": "1k5n_A",
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


def safe_std(
    value_sum: np.ndarray,
    value_sum_sq: np.ndarray,
    count: int,
) -> np.ndarray:
    mean = value_sum / count
    variance = np.maximum(
        value_sum_sq / count - mean**2,
        0.0,
    )
    return np.sqrt(variance)


def process_replica(
    protein_dir: Path,
    output_dir: Path,
    replica: int,
    stride: int,
    chunk_size: int,
    sphere_points: int,
    prefix_end_ns: float,
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    pd.DataFrame,
]:
    topology_file = protein_dir / "1k5n_A.pdb"
    trajectory_file = (
        protein_dir
        / f"1k5n_A_prod_R{replica}_fit.xtc"
    )

    if not topology_file.is_file():
        raise FileNotFoundError(topology_file)

    if not trajectory_file.is_file():
        raise FileNotFoundError(trajectory_file)

    frame_records: list[dict[str, object]] = []

    residue_metadata: list[dict[str, object]] | None = None

    full_sasa_sum: np.ndarray | None = None
    full_sasa_sum_sq: np.ndarray | None = None
    prefix_sasa_sum: np.ndarray | None = None
    prefix_sasa_sum_sq: np.ndarray | None = None

    full_ss_counts: dict[str, np.ndarray] = {}
    prefix_ss_counts: dict[str, np.ndarray] = {}

    full_count = 0
    prefix_count = 0

    iterator = md.iterload(
        str(trajectory_file),
        top=str(topology_file),
        chunk=chunk_size,
        stride=stride,
    )

    for trajectory_chunk in iterator:
        if residue_metadata is None:
            residues = list(
                trajectory_chunk.topology.residues
            )
            n_residues = len(residues)

            residue_metadata = [
                {
                    "residue_index": residue.index + 1,
                    "topology_resid": residue.resSeq,
                    "resname": residue.name,
                }
                for residue in residues
            ]

            full_sasa_sum = np.zeros(
                n_residues,
                dtype=np.float64,
            )
            full_sasa_sum_sq = np.zeros(
                n_residues,
                dtype=np.float64,
            )
            prefix_sasa_sum = np.zeros(
                n_residues,
                dtype=np.float64,
            )
            prefix_sasa_sum_sq = np.zeros(
                n_residues,
                dtype=np.float64,
            )

            for code in SS_CODES:
                full_ss_counts[code] = np.zeros(
                    n_residues,
                    dtype=np.int64,
                )
                prefix_ss_counts[code] = np.zeros(
                    n_residues,
                    dtype=np.int64,
                )

        # MDTraj devuelve SASA en nm². 1 nm² = 100 Å².
        residue_sasa_A2 = (
            md.shrake_rupley(
                trajectory_chunk,
                mode="residue",
                n_sphere_points=sphere_points,
            )
            * 100.0
        )

        dssp = md.compute_dssp(
            trajectory_chunk,
            simplified=True,
        )

        if residue_sasa_A2.shape != dssp.shape:
            raise RuntimeError(
                "Las dimensiones de SASA y DSSP no coinciden."
            )

        times_ns = (
            trajectory_chunk.time.astype(float)
            / 1000.0
        )

        valid_dssp = np.isin(dssp, SS_CODES)
        valid_counts = valid_dssp.sum(axis=1)

        if np.any(valid_counts == 0):
            raise RuntimeError(
                "Al menos un frame no tiene residuos DSSP válidos."
            )

        helix_fraction = (
            (dssp == "H").sum(axis=1)
            / valid_counts
        )
        strand_fraction = (
            (dssp == "E").sum(axis=1)
            / valid_counts
        )
        coil_fraction = (
            (dssp == "C").sum(axis=1)
            / valid_counts
        )

        total_sasa_A2 = residue_sasa_A2.sum(axis=1)

        for index in range(trajectory_chunk.n_frames):
            frame_records.append(
                {
                    "system": "1k5n_A",
                    "replica": replica,
                    "time_ps": float(
                        trajectory_chunk.time[index]
                    ),
                    "time_ns": float(times_ns[index]),
                    "total_sasa_A2": float(
                        total_sasa_A2[index]
                    ),
                    "mean_residue_sasa_A2": float(
                        residue_sasa_A2[index].mean()
                    ),
                    "helix_fraction": float(
                        helix_fraction[index]
                    ),
                    "strand_fraction": float(
                        strand_fraction[index]
                    ),
                    "coil_fraction": float(
                        coil_fraction[index]
                    ),
                    "structured_fraction": float(
                        helix_fraction[index]
                        + strand_fraction[index]
                    ),
                }
            )

        chunk_frames = trajectory_chunk.n_frames

        full_sasa_sum += residue_sasa_A2.sum(axis=0)
        full_sasa_sum_sq += (
            residue_sasa_A2**2
        ).sum(axis=0)
        full_count += chunk_frames

        for code in SS_CODES:
            full_ss_counts[code] += (
                dssp == code
            ).sum(axis=0)

        prefix_mask = (
            times_ns <= prefix_end_ns + 1e-9
        )
        prefix_frames = int(prefix_mask.sum())

        if prefix_frames:
            prefix_values = residue_sasa_A2[
                prefix_mask
            ]

            prefix_sasa_sum += prefix_values.sum(
                axis=0
            )
            prefix_sasa_sum_sq += (
                prefix_values**2
            ).sum(axis=0)
            prefix_count += prefix_frames

            for code in SS_CODES:
                prefix_ss_counts[code] += (
                    dssp[prefix_mask] == code
                ).sum(axis=0)

    if residue_metadata is None:
        raise RuntimeError(
            f"R{replica}: no se leyó ningún frame."
        )

    if prefix_count == 0:
        raise RuntimeError(
            f"R{replica}: el prefijo está vacío."
        )

    timeseries = pd.DataFrame(frame_records)
    timeseries = timeseries.sort_values(
        "time_ns"
    ).reset_index(drop=True)

    prefix_dataframe = timeseries.loc[
        timeseries["time_ns"]
        <= prefix_end_ns + 1e-9
    ].copy()

    summaries = [
        window_summary(
            timeseries,
            replica,
            "full",
        ),
        window_summary(
            prefix_dataframe,
            replica,
            "prefix",
        ),
    ]

    full_sasa_mean = full_sasa_sum / full_count
    prefix_sasa_mean = (
        prefix_sasa_sum / prefix_count
    )

    full_sasa_std = safe_std(
        full_sasa_sum,
        full_sasa_sum_sq,
        full_count,
    )
    prefix_sasa_std = safe_std(
        prefix_sasa_sum,
        prefix_sasa_sum_sq,
        prefix_count,
    )

    residue_records: list[dict[str, object]] = []

    for position, metadata in enumerate(
        residue_metadata
    ):
        residue_records.append(
            {
                "system": "1k5n_A",
                "replica": replica,
                **metadata,
                "prefix_n_frames": prefix_count,
                "full_n_frames": full_count,
                "prefix_sasa_mean_A2": float(
                    prefix_sasa_mean[position]
                ),
                "full_sasa_mean_A2": float(
                    full_sasa_mean[position]
                ),
                "prefix_sasa_std_A2": float(
                    prefix_sasa_std[position]
                ),
                "full_sasa_std_A2": float(
                    full_sasa_std[position]
                ),
                "prefix_helix_fraction": float(
                    prefix_ss_counts["H"][position]
                    / prefix_count
                ),
                "full_helix_fraction": float(
                    full_ss_counts["H"][position]
                    / full_count
                ),
                "prefix_strand_fraction": float(
                    prefix_ss_counts["E"][position]
                    / prefix_count
                ),
                "full_strand_fraction": float(
                    full_ss_counts["E"][position]
                    / full_count
                ),
                "prefix_coil_fraction": float(
                    prefix_ss_counts["C"][position]
                    / prefix_count
                ),
                "full_coil_fraction": float(
                    full_ss_counts["C"][position]
                    / full_count
                ),
            }
        )

    residue_profiles = pd.DataFrame(
        residue_records
    )

    timeseries_file = (
        output_dir
        / f"1k5n_A_R{replica}_sasa_dssp_100ps.csv"
    )
    timeseries.to_csv(
        timeseries_file,
        index=False,
    )

    print("=" * 72)
    print(f"RÉPLICA R{replica}")
    print(f"Frames completos: {len(timeseries)}")
    print(f"Frames prefijo:   {len(prefix_dataframe)}")
    print(
        "SASA media:       "
        f"{timeseries['total_sasa_A2'].mean():.2f} Å²"
    )
    print(
        "H/E/C medios:     "
        f"{timeseries['helix_fraction'].mean():.4f} / "
        f"{timeseries['strand_fraction'].mean():.4f} / "
        f"{timeseries['coil_fraction'].mean():.4f}"
    )
    print(f"CSV:              {timeseries_file}")

    return timeseries, summaries, residue_profiles


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_dir = input_dir / "1k5n_A_protein"

    all_summaries: list[dict[str, object]] = []
    all_residue_profiles: list[pd.DataFrame] = []

    for replica in (1, 2, 3):
        _, summaries, profiles = process_replica(
            protein_dir=protein_dir,
            output_dir=output_dir,
            replica=replica,
            stride=args.stride,
            chunk_size=args.chunk_size,
            sphere_points=args.sphere_points,
            prefix_end_ns=args.prefix_end_ns,
        )

        all_summaries.extend(summaries)
        all_residue_profiles.append(profiles)

    summary_dataframe = pd.DataFrame(
        all_summaries
    )
    residue_dataframe = pd.concat(
        all_residue_profiles,
        ignore_index=True,
    )

    summary_csv = (
        output_dir / "sasa_dssp_summary.csv"
    )
    residue_csv = (
        output_dir
        / "sasa_dssp_residue_profiles.csv"
    )
    metadata_json = (
        output_dir / "sasa_dssp_metadata.json"
    )

    summary_dataframe.to_csv(
        summary_csv,
        index=False,
    )
    residue_dataframe.to_csv(
        residue_csv,
        index=False,
    )

    metadata_json.write_text(
        json.dumps(
            {
                "stride": args.stride,
                "expected_resolution_ps": (
                    10.0 * args.stride
                ),
                "chunk_size": args.chunk_size,
                "sphere_points": args.sphere_points,
                "sasa_method": "Shrake-Rupley",
                "sasa_output_unit": "A^2",
                "dssp": "simplified H/E/C",
                "prefix_end_ns": args.prefix_end_ns,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("RESUMEN SASA Y DSSP")
    print("=" * 72)
    print(
        summary_dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"Resumen:   {summary_csv}")
    print(f"Residuos:  {residue_csv}")
    print(f"Metadatos: {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
