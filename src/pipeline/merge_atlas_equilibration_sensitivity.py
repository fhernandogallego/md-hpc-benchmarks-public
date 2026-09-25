#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fusiona resultados de sensibilidad calculados por réplica "
            "en los CSV canónicos consumidos por el pipeline ATLAS."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--replicas", default="1,2,3")
    parser.add_argument("--input-parent", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
        raise ValueError("La lista de réplicas está vacía.")

    if any(replica < 1 for replica in replicas):
        raise ValueError("Las réplicas deben ser positivas.")

    return replicas


def require_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        raise RuntimeError(f"Archivo vacío: {path}")

    # Preserve the exact IEEE-754 values encoded by the per-replica CSVs.
    # The default C parser may shift a decimal representation by one ULP on
    # an additional read/write round trip, which breaks exact equivalence with
    # the historical serial output even though the scientific value is unchanged.
    return pd.read_csv(path, float_precision="round_trip")


def main() -> int:
    args = parse_args()

    replicas = parse_replicas(args.replicas)
    input_parent = args.input_parent.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []

    detail_columns: list[str] | None = None
    summary_columns: list[str] | None = None
    detail_rows_per_replica: int | None = None
    summary_rows_per_replica: int | None = None

    metadata: dict[str, object] = {
        "system": args.system,
        "replicas": replicas,
        "input_parent": str(input_parent),
        "output_dir": str(output_dir),
        "replica_outputs": [],
    }

    for replica in replicas:
        replica_dir = input_parent / f"R{replica}"

        detail_file = replica_dir / "equilibration_sensitivity.csv"
        summary_file = replica_dir / "equilibration_sensitivity_summary.csv"

        detail = require_csv(detail_file)
        summary = require_csv(summary_file)

        required_detail = {
            "system",
            "replica",
            "metric",
            "resolution",
            "nskip",
        }
        missing_detail = required_detail.difference(detail.columns)

        if missing_detail:
            raise RuntimeError(
                f"{detail_file}: faltan columnas {sorted(missing_detail)}"
            )

        observed_systems = set(detail["system"].astype(str).unique())
        if observed_systems != {args.system}:
            raise RuntimeError(
                f"{detail_file}: sistemas observados "
                f"{sorted(observed_systems)}; esperado {args.system}."
            )

        observed_replicas = set(
            pd.to_numeric(detail["replica"], errors="raise").astype(int).unique()
        )
        if observed_replicas != {replica}:
            raise RuntimeError(
                f"{detail_file}: réplicas observadas "
                f"{sorted(observed_replicas)}; esperada R{replica}."
            )

        if "replica" not in summary.columns:
            raise RuntimeError(f"{summary_file}: falta columna replica")

        summary_replicas = set(
            pd.to_numeric(summary["replica"], errors="raise").astype(int).unique()
        )
        if summary_replicas != {replica}:
            raise RuntimeError(
                f"{summary_file}: réplicas observadas "
                f"{sorted(summary_replicas)}; esperada R{replica}."
            )

        current_detail_columns = list(detail.columns)
        current_summary_columns = list(summary.columns)

        if detail_columns is None:
            detail_columns = current_detail_columns
        elif current_detail_columns != detail_columns:
            raise RuntimeError(
                f"{detail_file}: esquema distinto entre réplicas."
            )

        if summary_columns is None:
            summary_columns = current_summary_columns
        elif current_summary_columns != summary_columns:
            raise RuntimeError(
                f"{summary_file}: esquema distinto entre réplicas."
            )

        if detail_rows_per_replica is None:
            detail_rows_per_replica = len(detail)
        elif len(detail) != detail_rows_per_replica:
            raise RuntimeError(
                f"{detail_file}: {len(detail)} filas; "
                f"esperadas {detail_rows_per_replica}."
            )

        if summary_rows_per_replica is None:
            summary_rows_per_replica = len(summary)
        elif len(summary) != summary_rows_per_replica:
            raise RuntimeError(
                f"{summary_file}: {len(summary)} filas; "
                f"esperadas {summary_rows_per_replica}."
            )

        detail_parts.append(detail)
        summary_parts.append(summary)

        metadata["replica_outputs"].append(
            {
                "replica": replica,
                "detail_file": str(detail_file.resolve()),
                "summary_file": str(summary_file.resolve()),
                "detail_rows": int(len(detail)),
                "summary_rows": int(len(summary)),
            }
        )

    merged_detail = pd.concat(detail_parts, ignore_index=True)
    merged_summary = pd.concat(summary_parts, ignore_index=True)

    duplicate_detail = merged_detail.duplicated(
        subset=["system", "replica", "metric", "resolution", "nskip"],
        keep=False,
    )
    if duplicate_detail.any():
        raise RuntimeError(
            "Hay filas duplicadas en la sensibilidad fusionada."
        )

    if len(merged_detail) != detail_rows_per_replica * len(replicas):
        raise RuntimeError("Número inesperado de filas de detalle fusionadas.")

    if len(merged_summary) != summary_rows_per_replica * len(replicas):
        raise RuntimeError("Número inesperado de filas de resumen fusionadas.")

    detail_output = output_dir / "equilibration_sensitivity.csv"
    summary_output = output_dir / "equilibration_sensitivity_summary.csv"
    metadata_output = output_dir / "equilibration_sensitivity_merge_metadata.json"

    merged_detail.to_csv(detail_output, index=False)
    merged_summary.to_csv(summary_output, index=False)

    metadata.update(
        {
            "detail_rows": int(len(merged_detail)),
            "summary_rows": int(len(merged_summary)),
            "detail_output": str(detail_output),
            "summary_output": str(summary_output),
        }
    )
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )

    print("=" * 100)
    print("MERGE SENSIBILIDAD ATLAS")
    print("=" * 100)
    print(f"System:   {args.system}")
    print(f"Replicas: {','.join(map(str, replicas))}")
    print(f"Detalle:  {detail_output}")
    print(f"Resumen:  {summary_output}")
    print(f"Filas detalle: {len(merged_detail)}")
    print(f"Filas resumen: {len(merged_summary)}")
    print("SENSITIVITY_MERGE=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
