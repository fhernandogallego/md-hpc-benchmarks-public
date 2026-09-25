#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge independent ATLAS SASA/DSSP replica outputs."
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--dataset-group", required=True)
    parser.add_argument("--replicas", default="1,2,3")
    parser.add_argument("--input-parent", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_replicas(value: str) -> list[int]:
    replicas = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not replicas or any(r < 1 for r in replicas):
        raise ValueError("Invalid replica list")
    return replicas


def main() -> int:
    args = parse_args()
    replicas = parse_replicas(args.replicas)
    input_parent = args.input_parent.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    profiles: list[pd.DataFrame] = []
    replica_metadata: list[dict[str, object]] = []
    common_metadata: dict[str, object] | None = None

    for replica in replicas:
        source = (
            input_parent
            / f"R{replica}"
            / args.dataset_group
            / args.system
            / "sasa_dssp"
        )

        frame_csv = source / f"{args.system}_R{replica}_sasa_dssp_100ps.csv"
        summary_csv = source / "sasa_dssp_summary.csv"
        profiles_csv = source / "sasa_dssp_residue_profiles.csv"
        metadata_json = source / "sasa_dssp_metadata.json"

        for path in (frame_csv, summary_csv, profiles_csv, metadata_json):
            if not path.is_file():
                raise FileNotFoundError(path)

        frame = pd.read_csv(frame_csv)
        if set(frame["replica"].astype(int).unique()) != {replica}:
            raise RuntimeError(f"{frame_csv}: unexpected replica values")
        frame.to_csv(output_dir / frame_csv.name, index=False)

        summary = pd.read_csv(summary_csv)
        if set(summary["replica"].astype(int).unique()) != {replica}:
            raise RuntimeError(f"{summary_csv}: unexpected replica values")
        summaries.append(summary)

        profile = pd.read_csv(profiles_csv)
        if set(profile["replica"].astype(int).unique()) != {replica}:
            raise RuntimeError(f"{profiles_csv}: unexpected replica values")
        profiles.append(profile)

        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
        if metadata.get("system") != args.system:
            raise RuntimeError(f"{metadata_json}: system mismatch")
        if metadata.get("dataset_group") != args.dataset_group:
            raise RuntimeError(f"{metadata_json}: dataset_group mismatch")
        if metadata.get("replicas") != [replica]:
            raise RuntimeError(f"{metadata_json}: expected replicas=[{replica}]")

        if common_metadata is None:
            common_metadata = {
                key: value
                for key, value in metadata.items()
                if key not in {"replicas", "replica_metadata"}
            }
        else:
            current_common = {
                key: value
                for key, value in metadata.items()
                if key not in {"replicas", "replica_metadata"}
            }
            if current_common != common_metadata:
                raise RuntimeError(
                    f"{metadata_json}: extraction configuration differs across replicas"
                )

        current_replica_metadata = metadata.get("replica_metadata", [])
        if len(current_replica_metadata) != 1:
            raise RuntimeError(f"{metadata_json}: expected one replica metadata record")
        if int(current_replica_metadata[0].get("replica")) != replica:
            raise RuntimeError(f"{metadata_json}: replica metadata mismatch")
        replica_metadata.extend(current_replica_metadata)

    summary = pd.concat(summaries, ignore_index=True)
    summary = summary.sort_values(["replica", "window"]).reset_index(drop=True)
    profiles_df = pd.concat(profiles, ignore_index=True)
    profiles_df = profiles_df.sort_values(["replica", "residue_index"]).reset_index(drop=True)

    summary.to_csv(output_dir / "sasa_dssp_summary.csv", index=False)
    profiles_df.to_csv(output_dir / "sasa_dssp_residue_profiles.csv", index=False)

    merged_metadata = dict(common_metadata or {})
    merged_metadata["replicas"] = replicas
    merged_metadata["replica_metadata"] = sorted(
        replica_metadata,
        key=lambda record: int(record["replica"]),
    )
    (output_dir / "sasa_dssp_metadata.json").write_text(
        json.dumps(merged_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 100)
    print("SASA/DSSP REPLICA MERGE")
    print("=" * 100)
    print(f"System:   {args.system}")
    print(f"Replicas: {replicas}")
    print(f"Frames:   {len(replicas)} replica CSV files")
    print(f"Summary:  {len(summary)} rows")
    print(f"Profiles: {len(profiles_df)} rows")
    print(f"Output:   {output_dir}")
    print("SASA_DSSP_MERGE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
