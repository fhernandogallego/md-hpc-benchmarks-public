#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VALUE_COLUMNS = (
    "time_ps",
    "time_ns",
    "rmsd_backbone_A",
    "radius_gyration_A",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara la extracción parametrizada contra "
            "los resultados canónicos del POC."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--canonical-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
    )
    return parser.parse_args()


def compare_file(
    canonical_file: Path,
    candidate_file: Path,
    replica: int,
    resolution: str,
    tolerance: float,
) -> dict[str, object]:
    canonical = pd.read_csv(canonical_file)
    candidate = pd.read_csv(candidate_file)

    missing_canonical = set(VALUE_COLUMNS).difference(
        canonical.columns
    )
    missing_candidate = set(VALUE_COLUMNS).difference(
        candidate.columns
    )

    if missing_canonical:
        raise RuntimeError(
            f"Faltan columnas en {canonical_file}: "
            f"{sorted(missing_canonical)}"
        )

    if missing_candidate:
        raise RuntimeError(
            f"Faltan columnas en {candidate_file}: "
            f"{sorted(missing_candidate)}"
        )

    if len(canonical) != len(candidate):
        raise RuntimeError(
            f"R{replica} {resolution}: "
            f"{len(canonical)} frente a {len(candidate)} filas."
        )

    record: dict[str, object] = {
        "system": canonical["system"].iloc[0]
        if "system" in canonical.columns
        else "",
        "replica": replica,
        "resolution": resolution,
        "n_rows": int(len(canonical)),
    }

    all_valid = True

    for column in VALUE_COLUMNS:
        first = canonical[column].to_numpy(
            dtype=float
        )
        second = candidate[column].to_numpy(
            dtype=float
        )

        difference = np.abs(first - second)
        maximum = float(np.max(difference))
        mean = float(np.mean(difference))

        record[f"{column}_max_abs_diff"] = maximum
        record[f"{column}_mean_abs_diff"] = mean

        if maximum > tolerance:
            all_valid = False

    record["regression_pass"] = all_valid
    return record


def main() -> int:
    args = parse_args()

    canonical_dir = args.canonical_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []

    suffixes = {
        "10ps": "_basic_observables.csv",
        "100ps": "_basic_observables_100ps.csv",
    }

    for replica in (1, 2, 3):
        for resolution, suffix in suffixes.items():
            filename = (
                f"{args.system}_R{replica}{suffix}"
            )

            canonical_file = (
                canonical_dir / filename
            )
            candidate_file = (
                candidate_dir / filename
            )

            if not canonical_file.is_file():
                raise FileNotFoundError(
                    canonical_file
                )

            if not candidate_file.is_file():
                raise FileNotFoundError(
                    candidate_file
                )

            records.append(
                compare_file(
                    canonical_file=canonical_file,
                    candidate_file=candidate_file,
                    replica=replica,
                    resolution=resolution,
                    tolerance=args.tolerance,
                )
            )

    results = pd.DataFrame(records)

    output_csv = (
        output_dir
        / "basic_parameterization_regression.csv"
    )
    output_json = (
        output_dir
        / "basic_parameterization_regression.json"
    )

    results.to_csv(output_csv, index=False)

    overall_pass = bool(
        results["regression_pass"].all()
    )

    output_json.write_text(
        json.dumps(
            {
                "system": args.system,
                "tolerance": args.tolerance,
                "overall_pass": overall_pass,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    display_columns = [
        "replica",
        "resolution",
        "n_rows",
        "rmsd_backbone_A_max_abs_diff",
        "radius_gyration_A_max_abs_diff",
        "regression_pass",
    ]

    print("=" * 100)
    print("REGRESIÓN DEL EXTRACTOR PARAMETRIZADO")
    print("=" * 100)
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.12e}",
        )
    )
    print()
    print(
        "Estado global: "
        + ("PASS" if overall_pass else "FAIL")
    )
    print(f"Resultado:     {output_csv}")
    print(f"Metadatos:     {output_json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
