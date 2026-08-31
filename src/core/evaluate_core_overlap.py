#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara núcleos rígidos obtenidos con RMSF del prefijo "
            "y de la trayectoria completa."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--quantiles",
        default="0.30,0.40,0.50",
        help="Percentiles separados por comas.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Réplicas necesarias para el núcleo consenso.",
    )
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=3,
        help="Longitud mínima de un segmento rígido.",
    )
    return parser.parse_args()


def parse_quantiles(value: str) -> list[float]:
    quantiles = [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]

    if not quantiles:
        raise ValueError("No se proporcionaron percentiles.")

    for quantile in quantiles:
        if not 0.0 < quantile < 1.0:
            raise ValueError(
                f"Percentil fuera de rango: {quantile}"
            )

    return sorted(set(quantiles))


def filter_short_segments(
    residue_indices: np.ndarray,
    selected: np.ndarray,
    min_length: int,
) -> np.ndarray:
    residue_indices = np.asarray(residue_indices, dtype=int)
    selected = np.asarray(selected, dtype=bool)

    if len(residue_indices) != len(selected):
        raise ValueError("Las dimensiones no coinciden.")

    keep = np.zeros(len(selected), dtype=bool)
    position = 0

    while position < len(selected):
        if not selected[position]:
            position += 1
            continue

        end = position + 1

        while (
            end < len(selected)
            and selected[end]
            and residue_indices[end]
            == residue_indices[end - 1] + 1
        ):
            end += 1

        if end - position >= min_length:
            keep[position:end] = True

        position = end

    return keep


def segment_lengths(
    residue_indices: np.ndarray,
    selected: np.ndarray,
) -> list[int]:
    residue_indices = np.asarray(residue_indices, dtype=int)
    selected = np.asarray(selected, dtype=bool)

    lengths: list[int] = []
    position = 0

    while position < len(selected):
        if not selected[position]:
            position += 1
            continue

        end = position + 1

        while (
            end < len(selected)
            and selected[end]
            and residue_indices[end]
            == residue_indices[end - 1] + 1
        ):
            end += 1

        lengths.append(end - position)
        position = end

    return lengths


def overlap_metrics(
    residue_indices: np.ndarray,
    prefix_selected: np.ndarray,
    full_selected: np.ndarray,
) -> dict[str, float | int]:
    prefix_set = set(
        residue_indices[prefix_selected].tolist()
    )
    full_set = set(
        residue_indices[full_selected].tolist()
    )

    intersection = prefix_set & full_set
    union = prefix_set | full_set

    precision = (
        len(intersection) / len(prefix_set)
        if prefix_set
        else 0.0
    )
    recall = (
        len(intersection) / len(full_set)
        if full_set
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    jaccard = (
        len(intersection) / len(union)
        if union
        else 0.0
    )

    prefix_segments = segment_lengths(
        residue_indices,
        prefix_selected,
    )
    full_segments = segment_lengths(
        residue_indices,
        full_selected,
    )

    return {
        "n_residues": int(len(residue_indices)),
        "prefix_core_n": int(len(prefix_set)),
        "full_core_n": int(len(full_set)),
        "intersection_n": int(len(intersection)),
        "union_n": int(len(union)),
        "prefix_coverage": float(
            len(prefix_set) / len(residue_indices)
        ),
        "full_coverage": float(
            len(full_set) / len(residue_indices)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
        "prefix_segment_count": int(
            len(prefix_segments)
        ),
        "full_segment_count": int(
            len(full_segments)
        ),
        "prefix_largest_segment": int(
            max(prefix_segments, default=0)
        ),
        "full_largest_segment": int(
            max(full_segments, default=0)
        ),
    }


def replica_profiles(
    dataframe: pd.DataFrame,
    replica: int,
) -> pd.DataFrame:
    subset = dataframe.loc[
        dataframe["replica"] == replica
    ].copy()

    metadata = (
        subset[
            [
                "residue_index",
                "topology_resid",
                "resname",
            ]
        ]
        .drop_duplicates()
        .sort_values("residue_index")
    )

    pivot = subset.pivot(
        index="residue_index",
        columns="window",
        values="rmsf_A",
    ).reset_index()

    result = metadata.merge(
        pivot,
        on="residue_index",
        how="inner",
        validate="one_to_one",
    )

    required = {"prefix", "full"}

    if not required.issubset(result.columns):
        raise RuntimeError(
            f"R{replica}: faltan ventanas RMSF."
        )

    return result.sort_values(
        "residue_index"
    ).reset_index(drop=True)


def main() -> int:
    args = parse_args()

    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    quantiles = parse_quantiles(args.quantiles)

    dataframe = pd.read_csv(input_csv)

    required = {
        "replica",
        "window",
        "residue_index",
        "topology_resid",
        "resname",
        "rmsf_A",
    }
    missing = required.difference(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"Faltan columnas: {sorted(missing)}"
        )

    summary_records: list[dict[str, object]] = []
    threshold_records: list[dict[str, object]] = []
    replica_flag_records: list[dict[str, object]] = []

    for replica in sorted(
        dataframe["replica"].unique()
    ):
        replica = int(replica)
        profiles = replica_profiles(
            dataframe,
            replica,
        )

        residue_indices = profiles[
            "residue_index"
        ].to_numpy(dtype=int)

        for quantile in quantiles:
            prefix_threshold = float(
                profiles["prefix"].quantile(quantile)
            )
            full_threshold = float(
                profiles["full"].quantile(quantile)
            )

            prefix_raw = (
                profiles["prefix"].to_numpy(dtype=float)
                <= prefix_threshold
            )
            full_raw = (
                profiles["full"].to_numpy(dtype=float)
                <= full_threshold
            )

            prefix_core = filter_short_segments(
                residue_indices,
                prefix_raw,
                args.min_segment_length,
            )
            full_core = filter_short_segments(
                residue_indices,
                full_raw,
                args.min_segment_length,
            )

            metrics = overlap_metrics(
                residue_indices,
                prefix_core,
                full_core,
            )

            summary_records.append(
                {
                    "system": "1k5n_A",
                    "scope": "replica",
                    "replica": replica,
                    "quantile": quantile,
                    "min_segment_length": (
                        args.min_segment_length
                    ),
                    **metrics,
                }
            )

            threshold_records.extend(
                [
                    {
                        "system": "1k5n_A",
                        "replica": replica,
                        "window": "prefix",
                        "quantile": quantile,
                        "rmsf_threshold_A": (
                            prefix_threshold
                        ),
                    },
                    {
                        "system": "1k5n_A",
                        "replica": replica,
                        "window": "full",
                        "quantile": quantile,
                        "rmsf_threshold_A": (
                            full_threshold
                        ),
                    },
                ]
            )

            for position, row in profiles.iterrows():
                replica_flag_records.append(
                    {
                        "system": "1k5n_A",
                        "replica": replica,
                        "quantile": quantile,
                        "residue_index": int(
                            row["residue_index"]
                        ),
                        "topology_resid": int(
                            row["topology_resid"]
                        ),
                        "resname": str(row["resname"]),
                        "prefix_rmsf_A": float(
                            row["prefix"]
                        ),
                        "full_rmsf_A": float(
                            row["full"]
                        ),
                        "prefix_threshold_A": (
                            prefix_threshold
                        ),
                        "full_threshold_A": (
                            full_threshold
                        ),
                        "prefix_core": bool(
                            prefix_core[position]
                        ),
                        "full_core": bool(
                            full_core[position]
                        ),
                    }
                )

    replica_flags = pd.DataFrame(
        replica_flag_records
    )

    consensus_records: list[dict[str, object]] = []

    for quantile in quantiles:
        subset = replica_flags.loc[
            replica_flags["quantile"] == quantile
        ].copy()

        consensus = (
            subset.groupby(
                [
                    "system",
                    "residue_index",
                    "topology_resid",
                    "resname",
                ],
                as_index=False,
            )
            .agg(
                replica_count=("replica", "nunique"),
                prefix_votes=("prefix_core", "sum"),
                full_votes=("full_core", "sum"),
                prefix_rmsf_median_A=(
                    "prefix_rmsf_A",
                    "median",
                ),
                full_rmsf_median_A=(
                    "full_rmsf_A",
                    "median",
                ),
            )
            .sort_values("residue_index")
            .reset_index(drop=True)
        )

        residue_indices = consensus[
            "residue_index"
        ].to_numpy(dtype=int)

        prefix_vote_core = (
            consensus["prefix_votes"].to_numpy(dtype=int)
            >= args.min_votes
        )
        full_vote_core = (
            consensus["full_votes"].to_numpy(dtype=int)
            >= args.min_votes
        )

        prefix_consensus = filter_short_segments(
            residue_indices,
            prefix_vote_core,
            args.min_segment_length,
        )
        full_consensus = filter_short_segments(
            residue_indices,
            full_vote_core,
            args.min_segment_length,
        )

        metrics = overlap_metrics(
            residue_indices,
            prefix_consensus,
            full_consensus,
        )

        summary_records.append(
            {
                "system": "1k5n_A",
                "scope": "consensus",
                "replica": "consensus",
                "quantile": quantile,
                "min_votes": args.min_votes,
                "min_segment_length": (
                    args.min_segment_length
                ),
                **metrics,
            }
        )

        for position, row in consensus.iterrows():
            consensus_records.append(
                {
                    **row.to_dict(),
                    "quantile": quantile,
                    "min_votes": args.min_votes,
                    "prefix_core": bool(
                        prefix_consensus[position]
                    ),
                    "full_core": bool(
                        full_consensus[position]
                    ),
                    "core_agreement": bool(
                        prefix_consensus[position]
                        == full_consensus[position]
                    ),
                }
            )

    summary = pd.DataFrame(summary_records)
    thresholds = pd.DataFrame(threshold_records)
    consensus_dataframe = pd.DataFrame(
        consensus_records
    )

    consensus_ranking = (
        summary.loc[
            summary["scope"] == "consensus"
        ]
        .sort_values(
            ["jaccard", "f1", "precision", "recall"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_quantile = float(
        consensus_ranking.iloc[0]["quantile"]
    )

    summary_csv = output_dir / "core_overlap_summary.csv"
    thresholds_csv = output_dir / "core_thresholds.csv"
    replica_flags_csv = (
        output_dir / "core_replica_residues.csv"
    )
    consensus_csv = (
        output_dir / "core_consensus_residues.csv"
    )
    metadata_json = output_dir / "core_overlap.json"

    summary.to_csv(summary_csv, index=False)
    thresholds.to_csv(thresholds_csv, index=False)
    replica_flags.to_csv(
        replica_flags_csv,
        index=False,
    )
    consensus_dataframe.to_csv(
        consensus_csv,
        index=False,
    )

    metadata_json.write_text(
        json.dumps(
            {
                "quantiles": quantiles,
                "min_votes": args.min_votes,
                "min_segment_length": (
                    args.min_segment_length
                ),
                "diagnostic_best_quantile": (
                    best_quantile
                ),
                "warning": (
                    "The diagnostic best quantile must not be "
                    "made canonical using only one protein."
                ),
                "leakage_policy": {
                    "prefix_core": (
                        "eligible for model input"
                    ),
                    "full_core": (
                        "diagnostic reference only"
                    ),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 110)
    print("SOLAPAMIENTO DE NÚCLEOS POR RÉPLICA")
    print("=" * 110)
    print(
        summary.loc[
            summary["scope"] == "replica",
            [
                "replica",
                "quantile",
                "prefix_core_n",
                "full_core_n",
                "precision",
                "recall",
                "f1",
                "jaccard",
                "prefix_segment_count",
                "full_segment_count",
            ],
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("=" * 110)
    print("NÚCLEO CONSENSO")
    print("=" * 110)
    print(
        consensus_ranking[
            [
                "quantile",
                "prefix_core_n",
                "full_core_n",
                "precision",
                "recall",
                "f1",
                "jaccard",
                "prefix_segment_count",
                "full_segment_count",
                "prefix_largest_segment",
                "full_largest_segment",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print(
        "Mejor percentil diagnóstico para 1k5n_A: "
        f"{best_quantile:.2f}"
    )
    print(
        "No debe fijarse como percentil global usando "
        "únicamente esta proteína."
    )

    print()
    print(f"Resumen:            {summary_csv}")
    print(f"Umbrales:           {thresholds_csv}")
    print(f"Residuos réplica:   {replica_flags_csv}")
    print(f"Residuos consenso:  {consensus_csv}")
    print(f"Metadatos:          {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
