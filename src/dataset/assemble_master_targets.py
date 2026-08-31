#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_ORDER = (
    "rmsd",
    "rg",
    "sasa",
    "helix",
    "strand",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unifica los targets validados de RMSD, Rg, SASA "
            "y estructura secundaria."
        )
    )
    parser.add_argument(
        "--basic-long",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--structural-long",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--composition-csv",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {"true", "false"}

    if not set(normalized.unique()).issubset(allowed):
        raise ValueError(
            "Valores booleanos no reconocidos: "
            f"{sorted(normalized.unique())}"
        )

    return normalized.eq("true")


def safe_relative_difference(
    difference: pd.Series,
    reference: pd.Series,
) -> pd.Series:
    difference_values = difference.to_numpy(dtype=float)
    reference_values = reference.to_numpy(dtype=float)

    result = np.full(
        len(difference_values),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(difference_values)
        & np.isfinite(reference_values)
        & (np.abs(reference_values) > 1e-12)
    )

    result[valid] = (
        difference_values[valid]
        / np.abs(reference_values[valid])
    )

    return pd.Series(
        result,
        index=difference.index,
    )


def normalize_basic_targets(
    dataframe: pd.DataFrame,
    source_file: Path,
) -> pd.DataFrame:
    required = {
        "system",
        "replica",
        "metric",
        "t0_ns",
        "equilibrated_duration_ns",
        "auto_neff",
        "auto_mean_A",
        "auto_sem_A",
        "target_valid",
        "target_status",
        "target_mean_A",
        "target_sem_A",
        "tail_mean_A",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"Faltan columnas básicas: {sorted(missing)}"
        )

    metric_mapping = {
        "RMSD": "rmsd",
        "Rg": "rg",
    }

    result = dataframe.loc[
        dataframe["metric"].isin(metric_mapping)
    ].copy()

    result["target_name"] = result[
        "metric"
    ].map(metric_mapping)

    result["unit"] = "A"
    result["window_policy"] = (
        "independent_observable_t0"
    )
    result["target_valid"] = normalize_boolean(
        result["target_valid"]
    )

    normalized = pd.DataFrame(
        {
            "system": result["system"],
            "replica": result["replica"].astype(int),
            "target_name": result["target_name"],
            "unit": result["unit"],
            "window_policy": result[
                "window_policy"
            ],
            "target_value": result[
                "target_mean_A"
            ],
            "target_sem": result[
                "target_sem_A"
            ],
            "target_valid": result[
                "target_valid"
            ],
            "target_status": result[
                "target_status"
            ],
            "t0_ns": result["t0_ns"],
            "duration_ns": result[
                "equilibrated_duration_ns"
            ],
            "neff": result["auto_neff"],
            "automatic_window_value": result[
                "auto_mean_A"
            ],
            "automatic_window_sem": result[
                "auto_sem_A"
            ],
            "tail_80_100_value": result[
                "tail_mean_A"
            ],
            "source_file": str(source_file),
        }
    )

    return normalized


def normalize_structural_targets(
    dataframe: pd.DataFrame,
    source_file: Path,
) -> pd.DataFrame:
    required = {
        "system",
        "replica",
        "source_column",
        "unit",
        "window_type",
        "t0_ns",
        "equilibrated_duration_ns",
        "auto_neff",
        "auto_mean",
        "auto_sem",
        "target_valid",
        "target_status",
        "target_mean",
        "target_sem",
        "tail_mean",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise RuntimeError(
            "Faltan columnas estructurales: "
            f"{sorted(missing)}"
        )

    target_mapping = {
        "total_sasa_A2": "sasa",
        "helix_fraction": "helix",
        "strand_fraction": "strand",
    }

    result = dataframe.loc[
        dataframe["source_column"].isin(
            target_mapping
        )
    ].copy()

    result["target_name"] = result[
        "source_column"
    ].map(target_mapping)

    result["target_valid"] = normalize_boolean(
        result["target_valid"]
    )

    normalized = pd.DataFrame(
        {
            "system": result["system"],
            "replica": result["replica"].astype(int),
            "target_name": result["target_name"],
            "unit": result["unit"],
            "window_policy": result[
                "window_type"
            ],
            "target_value": result[
                "target_mean"
            ],
            "target_sem": result[
                "target_sem"
            ],
            "target_valid": result[
                "target_valid"
            ],
            "target_status": result[
                "target_status"
            ],
            "t0_ns": result["t0_ns"],
            "duration_ns": result[
                "equilibrated_duration_ns"
            ],
            "neff": result["auto_neff"],
            "automatic_window_value": result[
                "auto_mean"
            ],
            "automatic_window_sem": result[
                "auto_sem"
            ],
            "tail_80_100_value": result[
                "tail_mean"
            ],
            "source_file": str(source_file),
        }
    )

    return normalized


def validate_long_targets(
    targets: pd.DataFrame,
) -> None:
    duplicates = targets.duplicated(
        subset=[
            "system",
            "replica",
            "target_name",
        ],
        keep=False,
    )

    if duplicates.any():
        duplicate_rows = targets.loc[
            duplicates,
            [
                "system",
                "replica",
                "target_name",
            ],
        ]

        raise RuntimeError(
            "Hay targets duplicados:\n"
            + duplicate_rows.to_string(index=False)
        )

    valid_rows = targets["target_valid"]

    invalid_valid_values = targets.loc[
        valid_rows
        & (
            targets["target_value"].isna()
            | targets["target_sem"].isna()
        )
    ]

    if not invalid_valid_values.empty:
        raise RuntimeError(
            "Hay targets válidos sin valor o SEM:\n"
            + invalid_valid_values.to_string(index=False)
        )

    invalid_rows_with_target = targets.loc[
        (~valid_rows)
        & targets["target_value"].notna()
    ]

    if not invalid_rows_with_target.empty:
        raise RuntimeError(
            "Hay targets inválidos con valor final:\n"
            + invalid_rows_with_target.to_string(index=False)
        )

    observed = set(
        targets["target_name"].unique()
    )
    expected = set(TARGET_ORDER)

    if observed != expected:
        raise RuntimeError(
            "Targets observados diferentes de los esperados. "
            f"Observados={sorted(observed)}, "
            f"esperados={sorted(expected)}"
        )


def add_sensitivity_diagnostics(
    targets: pd.DataFrame,
) -> pd.DataFrame:
    result = targets.copy()

    result["automatic_minus_tail"] = (
        result["automatic_window_value"]
        - result["tail_80_100_value"]
    )

    result["absolute_automatic_tail_difference"] = (
        result["automatic_minus_tail"].abs()
    )

    result["relative_automatic_tail_difference"] = (
        safe_relative_difference(
            result["automatic_minus_tail"],
            result["automatic_window_value"],
        )
    )

    return result


def build_wide_table(
    targets: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    grouped = targets.groupby(
        ["system", "replica"],
        sort=True,
    )

    for (system, replica), group in grouped:
        indexed = group.set_index("target_name")

        missing = set(TARGET_ORDER).difference(
            indexed.index
        )

        if missing:
            raise RuntimeError(
                f"{system} R{replica}: faltan targets "
                f"{sorted(missing)}"
            )

        record: dict[str, object] = {
            "system": system,
            "replica": int(replica),
        }

        mask_bits: list[str] = []
        valid_count = 0

        for target_name in TARGET_ORDER:
            row = indexed.loc[target_name]
            target_valid = bool(
                row["target_valid"]
            )

            record[f"{target_name}_target"] = (
                row["target_value"]
            )
            record[f"{target_name}_sem"] = (
                row["target_sem"]
            )
            record[f"{target_name}_valid"] = (
                target_valid
            )
            record[f"{target_name}_status"] = (
                row["target_status"]
            )
            record[f"{target_name}_t0_ns"] = (
                row["t0_ns"]
            )
            record[
                f"{target_name}_duration_ns"
            ] = row["duration_ns"]
            record[f"{target_name}_neff"] = (
                row["neff"]
            )
            record[
                f"{target_name}_tail_80_100"
            ] = row["tail_80_100_value"]
            record[
                f"{target_name}_auto_minus_tail"
            ] = row["automatic_minus_tail"]
            record[
                f"{target_name}_auto_tail_abs_diff"
            ] = row[
                "absolute_automatic_tail_difference"
            ]
            record[
                f"{target_name}_auto_tail_rel_diff"
            ] = row[
                "relative_automatic_tail_difference"
            ]

            mask_bits.append(
                "1" if target_valid else "0"
            )
            valid_count += int(target_valid)

        record["target_mask_order"] = ",".join(
            TARGET_ORDER
        )
        record["target_mask_signature"] = "".join(
            mask_bits
        )
        record["n_primary_targets"] = len(
            TARGET_ORDER
        )
        record["n_valid_targets"] = valid_count
        record["valid_target_fraction"] = (
            valid_count / len(TARGET_ORDER)
        )
        record["has_any_valid_target"] = (
            valid_count > 0
        )
        record["all_primary_targets_valid"] = (
            valid_count == len(TARGET_ORDER)
        )

        records.append(record)

    return pd.DataFrame(records)


def validate_composition(
    composition: pd.DataFrame,
) -> None:
    required = {
        "system",
        "replica",
        "composition_sum",
        "structured_sum",
        "structured_direct",
    }

    missing = required.difference(
        composition.columns
    )

    if missing:
        raise RuntimeError(
            f"Faltan columnas de composición: {sorted(missing)}"
        )

    composition_error = np.abs(
        composition["composition_sum"].to_numpy(
            dtype=float
        )
        - 1.0
    )

    structured_error = np.abs(
        composition["structured_sum"].to_numpy(
            dtype=float
        )
        - composition[
            "structured_direct"
        ].to_numpy(dtype=float)
    )

    if np.max(composition_error) > 1e-10:
        raise RuntimeError(
            "La composición H/E/C no suma uno."
        )

    if np.max(structured_error) > 1e-10:
        raise RuntimeError(
            "La fracción estructurada es inconsistente."
        )


def main() -> int:
    args = parse_args()

    basic_file = args.basic_long.resolve()
    structural_file = args.structural_long.resolve()
    composition_file = args.composition_csv.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    basic = pd.read_csv(basic_file)
    structural = pd.read_csv(structural_file)
    composition = pd.read_csv(composition_file)

    validate_composition(composition)

    basic_normalized = normalize_basic_targets(
        basic,
        source_file=basic_file,
    )
    structural_normalized = (
        normalize_structural_targets(
            structural,
            source_file=structural_file,
        )
    )

    targets = pd.concat(
        [
            basic_normalized,
            structural_normalized,
        ],
        ignore_index=True,
    )

    targets = add_sensitivity_diagnostics(
        targets
    )

    targets = targets.sort_values(
        [
            "system",
            "replica",
            "target_name",
        ]
    ).reset_index(drop=True)

    validate_long_targets(targets)

    wide = build_wide_table(targets)

    long_csv = output_dir / "master_targets_long.csv"
    wide_csv = output_dir / "master_targets_wide.csv"
    masks_csv = output_dir / "master_target_masks.csv"
    metadata_json = output_dir / "master_targets.json"

    targets.to_csv(long_csv, index=False)
    wide.to_csv(wide_csv, index=False)

    mask_columns = [
        "system",
        "replica",
        "target_mask_order",
        "target_mask_signature",
        "n_primary_targets",
        "n_valid_targets",
        "valid_target_fraction",
        "has_any_valid_target",
        "all_primary_targets_valid",
        *[
            f"{target_name}_valid"
            for target_name in TARGET_ORDER
        ],
    ]

    wide[mask_columns].to_csv(
        masks_csv,
        index=False,
    )

    metadata_json.write_text(
        json.dumps(
            {
                "target_order": TARGET_ORDER,
                "mask_encoding": {
                    "1": "valid target",
                    "0": "invalid or unavailable target",
                },
                "primary_targets": {
                    "rmsd": "backbone RMSD in A",
                    "rg": "protein radius of gyration in A",
                    "sasa": "total SASA in A^2",
                    "helix": "DSSP helix fraction",
                    "strand": "DSSP strand fraction",
                },
                "training_policy": (
                    "Use per-output masks in the loss. "
                    "Do not replace missing targets with zero."
                ),
                "normalization_policy": (
                    "Target normalization must be fitted only "
                    "on the training split after adding more proteins."
                ),
                "sensitivity_policy": (
                    "Automatic-window versus 80-100 ns differences "
                    "are diagnostics and do not currently alter "
                    "target_valid."
                ),
                "source_files": {
                    "basic_long": str(basic_file),
                    "structural_long": str(
                        structural_file
                    ),
                    "composition": str(
                        composition_file
                    ),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 130)
    print("TABLA MAESTRA DE TARGETS")
    print("=" * 130)

    display_columns = [
        "system",
        "replica",
        "target_mask_signature",
        "n_valid_targets",
        "rmsd_target",
        "rg_target",
        "sasa_target",
        "helix_target",
        "strand_target",
    ]

    print(
        wide[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 130)
    print("MÁSCARAS")
    print("=" * 130)

    print(
        wide[mask_columns].to_string(
            index=False
        )
    )

    print()
    print("Resumen de targets válidos:")
    print(
        targets.groupby(
            ["target_name", "target_valid"]
        ).size().to_string()
    )

    print()
    print(f"Formato largo: {long_csv}")
    print(f"Formato ancho: {wide_csv}")
    print(f"Máscaras:      {masks_csv}")
    print(f"Metadatos:     {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
