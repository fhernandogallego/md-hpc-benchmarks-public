#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import MDAnalysis as mda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida las parejas TPR/XTC de una entrada ATLAS."
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
        help="Directorio donde guardar el informe.",
    )
    return parser.parse_args()


def trajectory_definitions(input_dir: Path) -> list[dict[str, Any]]:
    analysis = input_dir / "1k5n_A_analysis"
    protein = input_dir / "1k5n_A_protein"

    definitions: list[dict[str, Any]] = []

    for replica in (1, 2, 3):
        definitions.append(
            {
                "dataset": "analysis",
                "replica": replica,
                "topology": analysis / f"1k5n_A_R{replica}.tpr",
                "trajectory": analysis / f"1k5n_A_R{replica}.xtc",
            }
        )

        definitions.append(
            {
                "dataset": "protein",
                "replica": replica,
                "topology": protein / f"1k5n_A_prod_R{replica}.tpr",
                "trajectory": protein / f"1k5n_A_prod_R{replica}_fit.xtc",
            }
        )

    return definitions


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def inspect_pair(definition: dict[str, Any]) -> dict[str, Any]:
    topology: Path = definition["topology"]
    trajectory: Path = definition["trajectory"]

    result: dict[str, Any] = {
        "dataset": definition["dataset"],
        "replica": definition["replica"],
        "topology": str(topology),
        "trajectory": str(trajectory),
        "topology_exists": topology.is_file(),
        "trajectory_exists": trajectory.is_file(),
        "topology_size_bytes": topology.stat().st_size if topology.is_file() else None,
        "trajectory_size_bytes": trajectory.stat().st_size if trajectory.is_file() else None,
        "status": "PENDING",
        "error": None,
    }

    if not topology.is_file() or not trajectory.is_file():
        result["status"] = "MISSING_FILE"
        result["error"] = "Falta la topología o la trayectoria."
        return result

    try:
        universe = mda.Universe(str(topology), str(trajectory))

        trajectory_reader = universe.trajectory

        n_frames = len(trajectory_reader)
        n_atoms = universe.atoms.n_atoms
        n_residues = universe.residues.n_residues
        n_segments = universe.segments.n_segments

        protein = universe.select_atoms("protein")
        backbone = universe.select_atoms("protein and backbone")
        alpha_carbons = universe.select_atoms("protein and name CA")

        first_ts = trajectory_reader[0]
        first_time_ps = safe_float(first_ts.time)
        first_dimensions = (
            first_ts.dimensions.tolist()
            if first_ts.dimensions is not None
            else None
        )

        last_ts = trajectory_reader[-1]
        last_time_ps = safe_float(last_ts.time)
        last_dimensions = (
            last_ts.dimensions.tolist()
            if last_ts.dimensions is not None
            else None
        )

        dt_ps = safe_float(trajectory_reader.dt)
        total_time_ps = safe_float(trajectory_reader.totaltime)

        result.update(
            {
                "status": "OK",
                "n_atoms": n_atoms,
                "n_residues": n_residues,
                "n_segments": n_segments,
                "n_protein_atoms": protein.n_atoms,
                "n_backbone_atoms": backbone.n_atoms,
                "n_alpha_carbons": alpha_carbons.n_atoms,
                "n_frames": n_frames,
                "dt_ps": dt_ps,
                "first_time_ps": first_time_ps,
                "last_time_ps": last_time_ps,
                "total_time_ps": total_time_ps,
                "total_time_ns": (
                    total_time_ps / 1000.0
                    if total_time_ps is not None
                    else None
                ),
                "first_dimensions": first_dimensions,
                "last_dimensions": last_dimensions,
                "trajectory_class": type(trajectory_reader).__name__,
                "trajectory_units": dict(trajectory_reader.units),
            }
        )

        # Dejar la trayectoria de nuevo en el primer frame.
        trajectory_reader[0]

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    return result


def write_csv(records: list[dict[str, Any]], output_file: Path) -> None:
    columns = [
        "dataset",
        "replica",
        "status",
        "n_atoms",
        "n_residues",
        "n_segments",
        "n_protein_atoms",
        "n_backbone_atoms",
        "n_alpha_carbons",
        "n_frames",
        "dt_ps",
        "first_time_ps",
        "last_time_ps",
        "total_time_ps",
        "total_time_ns",
        "topology_size_bytes",
        "trajectory_size_bytes",
        "topology",
        "trajectory",
        "error",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def print_record(record: dict[str, Any]) -> None:
    print("=" * 72)
    print(
        f"{record['dataset'].upper()} "
        f"R{record['replica']} — {record['status']}"
    )
    print(f"Topología:   {record['topology']}")
    print(f"Trayectoria: {record['trajectory']}")

    if record["status"] != "OK":
        print(f"Error: {record.get('error')}")
        return

    print(f"Átomos totales:       {record['n_atoms']}")
    print(f"Átomos de proteína:   {record['n_protein_atoms']}")
    print(f"Átomos de backbone:   {record['n_backbone_atoms']}")
    print(f"Carbonos alfa:        {record['n_alpha_carbons']}")
    print(f"Residuos:             {record['n_residues']}")
    print(f"Segmentos:            {record['n_segments']}")
    print(f"Frames:               {record['n_frames']}")
    print(f"Intervalo dt:         {record['dt_ps']} ps")
    print(f"Primer tiempo:        {record['first_time_ps']} ps")
    print(f"Último tiempo:        {record['last_time_ps']} ps")
    print(f"Duración MDAnalysis:  {record['total_time_ns']} ns")
    print(f"Reader:               {record['trajectory_class']}")
    print(f"Unidades:             {record['trajectory_units']}")
    print(f"Caja inicial:         {record['first_dimensions']}")
    print(f"Caja final:           {record['last_dimensions']}")


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(
            f"ERROR: no existe el directorio de entrada: {input_dir}",
            file=sys.stderr,
        )
        return 1

    definitions = trajectory_definitions(input_dir)
    records = [inspect_pair(definition) for definition in definitions]

    for record in records:
        print_record(record)

    json_file = output_dir / "trajectory_validation.json"
    csv_file = output_dir / "trajectory_validation.csv"

    json_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(records, csv_file)

    ok_count = sum(record["status"] == "OK" for record in records)
    error_count = len(records) - ok_count

    print()
    print("=" * 72)
    print("RESUMEN")
    print("=" * 72)
    print(f"Trayectorias comprobadas: {len(records)}")
    print(f"Correctas:                {ok_count}")
    print(f"Con error:                {error_count}")
    print(f"JSON:                     {json_file}")
    print(f"CSV:                      {csv_file}")

    if error_count:
        print(
            "ERROR: al menos una trayectoria no pudo validarse.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
