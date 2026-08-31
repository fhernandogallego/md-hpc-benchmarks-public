#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight de un sistema ATLAS con selección "
            "automática de una topología compatible con cada XTC."
        )
    )

    parser.add_argument(
        "--system",
        required=True,
    )
    parser.add_argument(
        "--unpacked-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--replicas",
        default="1,2,3",
    )
    parser.add_argument(
        "--analysis-step-ps",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--protein-step-ps",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--expected-end-ns",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--time-tolerance-ps",
        type=float,
        default=1e-3,
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
            "La lista de réplicas está vacía."
        )

    if any(replica < 1 for replica in replicas):
        raise ValueError(
            "Las réplicas deben ser positivas."
        )

    return replicas


def require_file(filename: Path) -> None:
    if not filename.is_file():
        raise FileNotFoundError(filename)

    if filename.stat().st_size == 0:
        raise RuntimeError(
            f"Archivo vacío: {filename}"
        )


def universe_counts(
    universe: mda.Universe,
) -> dict[str, int]:
    return {
        "n_atoms": int(
            universe.atoms.n_atoms
        ),
        "n_residues": int(
            universe.residues.n_residues
        ),
        "n_protein_atoms": int(
            universe.select_atoms(
                "protein"
            ).n_atoms
        ),
        "n_ca": int(
            universe.select_atoms(
                "protein and name CA"
            ).n_atoms
        ),
        "n_backbone": int(
            universe.select_atoms(
                "protein and backbone"
            ).n_atoms
        ),
    }


def inspect_topology(
    topology: Path,
) -> dict[str, int]:
    require_file(topology)

    universe = mda.Universe(
        str(topology)
    )

    return universe_counts(universe)


def unique_candidates(
    candidates: list[
        tuple[str, Path]
    ],
) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    observed: set[str] = set()

    for kind, filename in candidates:
        key = str(filename.resolve())

        if key in observed:
            continue

        observed.add(key)
        result.append(
            (
                kind,
                filename,
            )
        )

    return result


def load_compatible_universe(
    xtc_file: Path,
    candidates: list[
        tuple[str, Path]
    ],
) -> tuple[
    mda.Universe,
    str,
    Path,
    list[dict[str, Any]],
]:
    require_file(xtc_file)

    attempts: list[
        dict[str, Any]
    ] = []

    for topology_kind, topology_file in (
        unique_candidates(candidates)
    ):
        if not topology_file.is_file():
            attempts.append(
                {
                    "topology_kind": (
                        topology_kind
                    ),
                    "topology_file": str(
                        topology_file
                    ),
                    "success": False,
                    "error": (
                        "topology_file_missing"
                    ),
                }
            )
            continue

        try:
            universe = mda.Universe(
                str(topology_file),
                str(xtc_file),
            )

            # Fuerza la lectura del primer frame.
            universe.trajectory[0]

        except Exception as error:
            attempts.append(
                {
                    "topology_kind": (
                        topology_kind
                    ),
                    "topology_file": str(
                        topology_file.resolve()
                    ),
                    "success": False,
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )
            continue

        attempts.append(
            {
                "topology_kind": (
                    topology_kind
                ),
                "topology_file": str(
                    topology_file.resolve()
                ),
                "success": True,
                "error": "",
                "n_atoms": int(
                    universe.atoms.n_atoms
                ),
            }
        )

        return (
            universe,
            topology_kind,
            topology_file.resolve(),
            attempts,
        )

    formatted_attempts = json.dumps(
        attempts,
        indent=2,
        ensure_ascii=False,
    )

    raise RuntimeError(
        "Ninguna topología es compatible "
        f"con {xtc_file}:\n"
        f"{formatted_attempts}"
    )


def inspect_trajectory(
    *,
    system: str,
    mode: str,
    replica: int,
    reference_pdb: Path,
    declared_tpr: Path,
    xtc_file: Path,
    topology_candidates: list[
        tuple[str, Path]
    ],
    expected_step_ps: float,
    expected_end_ps: float,
    time_tolerance_ps: float,
) -> dict[str, Any]:
    require_file(reference_pdb)
    require_file(declared_tpr)
    require_file(xtc_file)

    pdb_counts = inspect_topology(
        reference_pdb
    )

    declared_tpr_counts = inspect_topology(
        declared_tpr
    )

    (
        universe,
        selected_topology_kind,
        selected_topology_file,
        topology_attempts,
    ) = load_compatible_universe(
        xtc_file=xtc_file,
        candidates=topology_candidates,
    )

    trajectory_counts = universe_counts(
        universe
    )

    n_frames = int(
        len(universe.trajectory)
    )

    if n_frames < 2:
        raise RuntimeError(
            f"{mode} R{replica}: "
            f"solo contiene {n_frames} frames."
        )

    times_ps = np.asarray(
        [
            float(timestep.time)
            for timestep in universe.trajectory
        ],
        dtype=np.float64,
    )

    differences_ps = np.diff(
        times_ps
    )

    observed_step_ps = float(
        np.median(differences_ps)
    )

    maximum_step_deviation_ps = float(
        np.max(
            np.abs(
                differences_ps
                - expected_step_ps
            )
        )
    )

    expected_n_frames = int(
        round(
            expected_end_ps
            / expected_step_ps
        )
        + 1
    )

    topology_matches_pdb = bool(
        trajectory_counts["n_atoms"]
        == pdb_counts["n_atoms"]
        and trajectory_counts[
            "n_residues"
        ]
        == pdb_counts["n_residues"]
        and trajectory_counts["n_ca"]
        == pdb_counts["n_ca"]
    )

    declared_tpr_xtc_compatible = any(
        bool(attempt["success"])
        and Path(
            attempt["topology_file"]
        ).resolve()
        == declared_tpr.resolve()
        for attempt in topology_attempts
    )

    times_finite = bool(
        np.all(
            np.isfinite(times_ps)
        )
    )

    times_strictly_increasing = bool(
        np.all(
            differences_ps > 0
        )
    )

    uniform_step = bool(
        np.allclose(
            differences_ps,
            expected_step_ps,
            rtol=0.0,
            atol=time_tolerance_ps,
        )
    )

    start_time_valid = bool(
        abs(float(times_ps[0]))
        <= time_tolerance_ps
    )

    end_tolerance_ps = max(
        time_tolerance_ps,
        expected_step_ps * 0.01,
    )

    end_time_valid = bool(
        abs(
            float(times_ps[-1])
            - expected_end_ps
        )
        <= end_tolerance_ps
    )

    frame_count_valid = bool(
        n_frames == expected_n_frames
    )

    preflight_pass = bool(
        topology_matches_pdb
        and times_finite
        and times_strictly_increasing
        and uniform_step
        and start_time_valid
        and end_time_valid
        and frame_count_valid
        and trajectory_counts[
            "n_atoms"
        ] > 0
        and trajectory_counts[
            "n_residues"
        ] > 0
        and trajectory_counts[
            "n_ca"
        ] > 0
    )

    return {
        "system": system,
        "mode": mode,
        "replica": replica,
        "reference_pdb": str(
            reference_pdb.resolve()
        ),
        "declared_tpr": str(
            declared_tpr.resolve()
        ),
        "xtc_file": str(
            xtc_file.resolve()
        ),
        "selected_topology_kind": (
            selected_topology_kind
        ),
        "selected_topology_file": str(
            selected_topology_file
        ),
        "pdb_n_atoms": (
            pdb_counts["n_atoms"]
        ),
        "pdb_n_residues": (
            pdb_counts["n_residues"]
        ),
        "pdb_n_ca": (
            pdb_counts["n_ca"]
        ),
        "declared_tpr_n_atoms": (
            declared_tpr_counts["n_atoms"]
        ),
        "declared_tpr_n_residues": (
            declared_tpr_counts[
                "n_residues"
            ]
        ),
        "trajectory_n_atoms": (
            trajectory_counts["n_atoms"]
        ),
        "trajectory_n_residues": (
            trajectory_counts[
                "n_residues"
            ]
        ),
        "trajectory_n_protein_atoms": (
            trajectory_counts[
                "n_protein_atoms"
            ]
        ),
        "trajectory_n_ca": (
            trajectory_counts["n_ca"]
        ),
        "trajectory_n_backbone": (
            trajectory_counts[
                "n_backbone"
            ]
        ),
        "n_frames": n_frames,
        "expected_n_frames": (
            expected_n_frames
        ),
        "start_time_ps": float(
            times_ps[0]
        ),
        "end_time_ps": float(
            times_ps[-1]
        ),
        "end_time_ns": float(
            times_ps[-1] / 1000.0
        ),
        "observed_step_ps": (
            observed_step_ps
        ),
        "expected_step_ps": (
            expected_step_ps
        ),
        "maximum_step_deviation_ps": (
            maximum_step_deviation_ps
        ),
        "declared_tpr_xtc_compatible": (
            declared_tpr_xtc_compatible
        ),
        "topology_matches_pdb": (
            topology_matches_pdb
        ),
        "times_finite": times_finite,
        "times_strictly_increasing": (
            times_strictly_increasing
        ),
        "uniform_step": uniform_step,
        "start_time_valid": (
            start_time_valid
        ),
        "end_time_valid": (
            end_time_valid
        ),
        "frame_count_valid": (
            frame_count_valid
        ),
        "preflight_pass": (
            preflight_pass
        ),
        "topology_attempts_json": (
            json.dumps(
                topology_attempts,
                ensure_ascii=False,
            )
        ),
    }


def check_replica_consistency(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    consistency_columns = [
        "trajectory_n_atoms",
        "trajectory_n_residues",
        "trajectory_n_protein_atoms",
        "trajectory_n_ca",
        "trajectory_n_backbone",
        "n_frames",
        "start_time_ps",
        "end_time_ps",
        "observed_step_ps",
    ]

    result: dict[str, Any] = {}
    overall_pass = True

    for mode, subset in dataframe.groupby(
        "mode",
        sort=True,
    ):
        mode_result: dict[str, bool] = {}

        for column in consistency_columns:
            values = pd.to_numeric(
                subset[column],
                errors="raise",
            ).to_numpy(
                dtype=np.float64
            )

            consistent = bool(
                np.allclose(
                    values,
                    values[0],
                    rtol=0.0,
                    atol=1e-8,
                )
            )

            mode_result[column] = (
                consistent
            )
            overall_pass &= consistent

        result[mode] = mode_result

    return {
        "overall_pass": bool(
            overall_pass
        ),
        "checks": result,
    }


def main() -> int:
    args = parse_args()

    replicas = parse_replicas(
        args.replicas
    )

    unpacked_dir = (
        args.unpacked_dir.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    analysis_dir = (
        unpacked_dir
        / f"{args.system}_analysis"
    )
    protein_dir = (
        unpacked_dir
        / f"{args.system}_protein"
    )

    if not analysis_dir.is_dir():
        raise FileNotFoundError(
            analysis_dir
        )

    if not protein_dir.is_dir():
        raise FileNotFoundError(
            protein_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_pdb = (
        analysis_dir
        / f"{args.system}.pdb"
    )
    protein_pdb = (
        protein_dir
        / f"{args.system}.pdb"
    )

    expected_end_ps = (
        args.expected_end_ns
        * 1000.0
    )

    records: list[
        dict[str, Any]
    ] = []

    for replica in replicas:
        analysis_tpr = (
            analysis_dir
            / (
                f"{args.system}_R"
                f"{replica}.tpr"
            )
        )

        analysis_xtc = (
            analysis_dir
            / (
                f"{args.system}_R"
                f"{replica}.xtc"
            )
        )

        protein_tpr = (
            protein_dir
            / (
                f"{args.system}_prod_R"
                f"{replica}.tpr"
            )
        )

        protein_xtc = (
            protein_dir
            / (
                f"{args.system}_prod_R"
                f"{replica}_fit.xtc"
            )
        )

        records.append(
            inspect_trajectory(
                system=args.system,
                mode="analysis",
                replica=replica,
                reference_pdb=analysis_pdb,
                declared_tpr=analysis_tpr,
                xtc_file=analysis_xtc,
                topology_candidates=[
                    (
                        "analysis_tpr",
                        analysis_tpr,
                    ),
                    (
                        "protein_tpr",
                        protein_tpr,
                    ),
                    (
                        "analysis_pdb",
                        analysis_pdb,
                    ),
                    (
                        "protein_pdb",
                        protein_pdb,
                    ),
                ],
                expected_step_ps=(
                    args.analysis_step_ps
                ),
                expected_end_ps=(
                    expected_end_ps
                ),
                time_tolerance_ps=(
                    args.time_tolerance_ps
                ),
            )
        )

        records.append(
            inspect_trajectory(
                system=args.system,
                mode="protein",
                replica=replica,
                reference_pdb=protein_pdb,
                declared_tpr=protein_tpr,
                xtc_file=protein_xtc,
                topology_candidates=[
                    (
                        "protein_tpr",
                        protein_tpr,
                    ),
                    (
                        "protein_pdb",
                        protein_pdb,
                    ),
                    (
                        "analysis_pdb",
                        analysis_pdb,
                    ),
                    (
                        "analysis_tpr",
                        analysis_tpr,
                    ),
                ],
                expected_step_ps=(
                    args.protein_step_ps
                ),
                expected_end_ps=(
                    expected_end_ps
                ),
                time_tolerance_ps=(
                    args.time_tolerance_ps
                ),
            )
        )

    dataframe = pd.DataFrame(
        records
    ).sort_values(
        [
            "mode",
            "replica",
        ]
    ).reset_index(drop=True)

    replica_consistency = (
        check_replica_consistency(
            dataframe
        )
    )

    rows_pass = bool(
        dataframe[
            "preflight_pass"
        ].all()
    )

    overall_pass = bool(
        rows_pass
        and replica_consistency[
            "overall_pass"
        ]
    )

    detail_file = (
        output_dir
        / "atlas_preflight_detail.csv"
    )
    summary_file = (
        output_dir
        / "atlas_preflight_summary.json"
    )

    dataframe.to_csv(
        detail_file,
        index=False,
    )

    selected_topologies = (
        dataframe[
            [
                "mode",
                "replica",
                "selected_topology_kind",
                "selected_topology_file",
                "declared_tpr_xtc_compatible",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    summary = {
        "system": args.system,
        "overall_pass": overall_pass,
        "replicas": replicas,
        "expected_end_ns": (
            args.expected_end_ns
        ),
        "analysis_step_ps": (
            args.analysis_step_ps
        ),
        "protein_step_ps": (
            args.protein_step_ps
        ),
        "n_checks": int(
            len(dataframe)
        ),
        "n_passed": int(
            dataframe[
                "preflight_pass"
            ].sum()
        ),
        "replica_consistency": (
            replica_consistency
        ),
        "selected_topologies": (
            selected_topologies
        ),
        "detail_csv": str(
            detail_file
        ),
    }

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    display_columns = [
        "mode",
        "replica",
        "declared_tpr_n_atoms",
        "trajectory_n_atoms",
        "trajectory_n_residues",
        "trajectory_n_ca",
        "selected_topology_kind",
        "declared_tpr_xtc_compatible",
        "n_frames",
        "end_time_ns",
        "observed_step_ps",
        "topology_matches_pdb",
        "preflight_pass",
    ]

    print("=" * 165)
    print(
        f"PREFLIGHT ATLAS: {args.system}"
    )
    print("=" * 165)

    print(
        dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Consistencia entre réplicas: "
        + (
            "PASS"
            if replica_consistency[
                "overall_pass"
            ]
            else "FAIL"
        )
    )
    print(
        "Estado global: "
        + (
            "PASS"
            if overall_pass
            else "FAIL"
        )
    )
    print(f"Detalle: {detail_file}")
    print(f"Resumen: {summary_file}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
