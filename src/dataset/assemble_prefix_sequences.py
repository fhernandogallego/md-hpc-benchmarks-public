#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "rmsd_backbone_A",
    "radius_gyration_A",
    "core_rmsd_q30_A",
    "core_rmsd_q40_A",
    "core_rmsd_q50_A",
    "total_sasa_A2",
    "helix_fraction",
    "strand_fraction",
]

TARGET_COLUMNS = [
    "rmsd",
    "rg",
    "sasa",
    "helix",
    "strand",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensambla secuencias temporales del prefijo con "
            "sus targets multisalida y máscaras."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--replicas", default="1,2,3")
    parser.add_argument("--basic-dir", required=True, type=Path)
    parser.add_argument("--core-rmsd-dir", required=True, type=Path)
    parser.add_argument("--sasa-dssp-dir", required=True, type=Path)
    parser.add_argument("--master-targets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix-end-ns", type=float, default=20.0)
    parser.add_argument("--expected-step-ps", type=int, default=100)
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


def normalize_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(f"Booleano no reconocido: {value!r}")


def add_time_key(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    if "time_ps" in result.columns:
        time_ps = result["time_ps"].to_numpy(dtype=float)
    elif "time_ns" in result.columns:
        time_ps = (
            result["time_ns"].to_numpy(dtype=float)
            * 1000.0
        )
    else:
        raise RuntimeError(
            "No existe una columna time_ps o time_ns."
        )

    result["time_ps_key"] = np.rint(
        time_ps
    ).astype(np.int64)

    if result["time_ps_key"].duplicated().any():
        duplicated = result.loc[
            result["time_ps_key"].duplicated(
                keep=False
            ),
            "time_ps_key",
        ].tolist()

        raise RuntimeError(
            f"Hay tiempos duplicados: {duplicated[:10]}"
        )

    return result


def load_basic(
    basic_dir: Path,
    system: str,
    replica: int,
) -> pd.DataFrame:
    filename = (
        basic_dir
        / f"{system}_R{replica}_basic_observables_100ps.csv"
    )

    if not filename.is_file():
        raise FileNotFoundError(filename)

    dataframe = pd.read_csv(filename)
    dataframe = add_time_key(dataframe)

    required = {
        "time_ps_key",
        "time_ns",
        "rmsd_backbone_A",
        "radius_gyration_A",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"Faltan columnas en {filename}: {sorted(missing)}"
        )

    return dataframe[
        [
            "time_ps_key",
            "time_ns",
            "rmsd_backbone_A",
            "radius_gyration_A",
        ]
    ].copy()


def load_core_rmsd(
    core_dir: Path,
    system: str,
    replica: int,
    quantile: int,
) -> pd.DataFrame:
    filename = (
        core_dir
        / f"{system}_R{replica}_q{quantile:02d}_core_rmsd.csv"
    )

    if not filename.is_file():
        raise FileNotFoundError(filename)

    dataframe = pd.read_csv(filename)
    dataframe = add_time_key(dataframe)

    required = {
        "time_ps_key",
        "backbone_rmsd_A",
        "replica_prefix_rmsd_A",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"Faltan columnas en {filename}: {sorted(missing)}"
        )

    return dataframe[
        [
            "time_ps_key",
            "backbone_rmsd_A",
            "replica_prefix_rmsd_A",
        ]
    ].rename(
        columns={
            "backbone_rmsd_A": (
                f"backbone_check_q{quantile:02d}_A"
            ),
            "replica_prefix_rmsd_A": (
                f"core_rmsd_q{quantile:02d}_A"
            ),
        }
    )


def load_sasa_dssp(
    sasa_dir: Path,
    system: str,
    replica: int,
) -> pd.DataFrame:
    filename = (
        sasa_dir
        / f"{system}_R{replica}_sasa_dssp_100ps.csv"
    )

    if not filename.is_file():
        raise FileNotFoundError(filename)

    dataframe = pd.read_csv(filename)
    dataframe = add_time_key(dataframe)

    required = {
        "time_ps_key",
        "total_sasa_A2",
        "helix_fraction",
        "strand_fraction",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"Faltan columnas en {filename}: {sorted(missing)}"
        )

    return dataframe[
        [
            "time_ps_key",
            "total_sasa_A2",
            "helix_fraction",
            "strand_fraction",
        ]
    ].copy()


def validate_time_grid(
    dataframe: pd.DataFrame,
    prefix_end_ns: float,
    expected_step_ps: int,
) -> None:
    expected_steps = (
        int(
            round(
                prefix_end_ns * 1000.0
                / expected_step_ps
            )
        )
        + 1
    )

    if len(dataframe) != expected_steps:
        raise RuntimeError(
            f"Se esperaban {expected_steps} pasos y "
            f"se obtuvieron {len(dataframe)}."
        )

    time_ps = dataframe[
        "time_ps_key"
    ].to_numpy(dtype=np.int64)

    if time_ps[0] != 0:
        raise RuntimeError(
            f"La secuencia comienza en {time_ps[0]} ps."
        )

    expected_final_ps = int(
        round(prefix_end_ns * 1000.0)
    )

    if time_ps[-1] != expected_final_ps:
        raise RuntimeError(
            f"La secuencia termina en {time_ps[-1]} ps, "
            f"no en {expected_final_ps} ps."
        )

    increments = np.diff(time_ps)

    if not np.all(increments == expected_step_ps):
        invalid = increments[
            increments != expected_step_ps
        ]

        raise RuntimeError(
            "La rejilla temporal no es uniforme. "
            f"Incrementos incorrectos: {invalid[:10]}"
        )


def validate_features(
    dataframe: pd.DataFrame,
) -> None:
    missing = set(FEATURE_COLUMNS).difference(
        dataframe.columns
    )

    if missing:
        raise RuntimeError(
            f"Faltan características: {sorted(missing)}"
        )

    feature_values = dataframe[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    if not np.all(np.isfinite(feature_values)):
        positions = np.argwhere(
            ~np.isfinite(feature_values)
        )

        raise RuntimeError(
            "Hay valores no finitos en las características: "
            f"{positions[:10].tolist()}"
        )

    for column in (
        "helix_fraction",
        "strand_fraction",
    ):
        values = dataframe[column].to_numpy(
            dtype=float
        )

        if np.any(values < -1e-10) or np.any(
            values > 1.0 + 1e-10
        ):
            raise RuntimeError(
                f"{column} contiene valores fuera de [0,1]."
            )

    secondary_sum = (
        dataframe["helix_fraction"]
        + dataframe["strand_fraction"]
    ).to_numpy(dtype=float)

    if np.any(secondary_sum > 1.0 + 1e-10):
        raise RuntimeError(
            "helix_fraction + strand_fraction es mayor que uno."
        )


def assemble_replica(
    system: str,
    replica: int,
    basic_dir: Path,
    core_dir: Path,
    sasa_dir: Path,
    output_dir: Path,
    prefix_end_ns: float,
    expected_step_ps: int,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
    list[dict[str, object]],
]:
    dataframe = load_basic(
        basic_dir=basic_dir,
        system=system,
        replica=replica,
    )

    for quantile in (30, 40, 50):
        core = load_core_rmsd(
            core_dir=core_dir,
            system=system,
            replica=replica,
            quantile=quantile,
        )

        dataframe = dataframe.merge(
            core,
            on="time_ps_key",
            how="inner",
            validate="one_to_one",
        )

    sasa = load_sasa_dssp(
        sasa_dir=sasa_dir,
        system=system,
        replica=replica,
    )

    dataframe = dataframe.merge(
        sasa,
        on="time_ps_key",
        how="inner",
        validate="one_to_one",
    )

    dataframe = dataframe.loc[
        dataframe["time_ns"]
        <= prefix_end_ns + 1e-9
    ].copy()

    dataframe = dataframe.sort_values(
        "time_ps_key"
    ).reset_index(drop=True)

    validation_differences: dict[str, float] = {}

    for quantile in (30, 40, 50):
        check_column = (
            f"backbone_check_q{quantile:02d}_A"
        )

        difference = np.abs(
            dataframe["rmsd_backbone_A"].to_numpy(
                dtype=float
            )
            - dataframe[check_column].to_numpy(
                dtype=float
            )
        )

        maximum_difference = float(
            np.max(difference)
        )

        validation_differences[
            f"max_backbone_difference_q{quantile:02d}_A"
        ] = maximum_difference

        if maximum_difference > 1e-6:
            raise RuntimeError(
                f"R{replica}: el RMSD básico y core q"
                f"{quantile:02d} difieren hasta "
                f"{maximum_difference:.8f} Å."
            )

    validate_time_grid(
        dataframe=dataframe,
        prefix_end_ns=prefix_end_ns,
        expected_step_ps=expected_step_ps,
    )
    validate_features(dataframe)

    output_dataframe = pd.DataFrame(
        {
            "system": system,
            "replica": replica,
            "sample_id": f"{system}_R{replica}",
            "step_index": np.arange(
                len(dataframe),
                dtype=int,
            ),
            "time_ps": dataframe[
                "time_ps_key"
            ].astype(int),
            "time_ns": dataframe[
                "time_ns"
            ].astype(float),
            "normalized_time": (
                dataframe["time_ns"].astype(float)
                / prefix_end_ns
            ),
        }
    )

    for column in FEATURE_COLUMNS:
        output_dataframe[column] = dataframe[
            column
        ].to_numpy(dtype=float)

    output_file = (
        output_dir
        / f"{system}_R{replica}_prefix_sequence_100ps.csv"
    )

    output_dataframe.to_csv(
        output_file,
        index=False,
    )

    feature_summaries: list[dict[str, object]] = []

    for column in FEATURE_COLUMNS:
        values = output_dataframe[
            column
        ].to_numpy(dtype=float)

        feature_summaries.append(
            {
                "system": system,
                "replica": replica,
                "sample_id": f"{system}_R{replica}",
                "feature": column,
                "n_steps": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "first": float(values[0]),
                "last": float(values[-1]),
                "last_minus_first": float(
                    values[-1] - values[0]
                ),
            }
        )

    manifest_record = {
        "system": system,
        "replica": replica,
        "sample_id": f"{system}_R{replica}",
        "sequence_csv": str(output_file),
        "n_steps": int(len(output_dataframe)),
        "n_features": len(FEATURE_COLUMNS),
        "start_ns": float(
            output_dataframe["time_ns"].iloc[0]
        ),
        "end_ns": float(
            output_dataframe["time_ns"].iloc[-1]
        ),
        "step_ps": expected_step_ps,
        **validation_differences,
    }

    return (
        output_dataframe,
        manifest_record,
        feature_summaries,
    )


def load_targets(
    filename: Path,
    system: str,
) -> pd.DataFrame:
    dataframe = pd.read_csv(
        filename,
        dtype={"target_mask_signature": "string"},
    )

    required = {
        "system",
        "replica",
        "target_mask_signature",
    }

    for target in TARGET_COLUMNS:
        required.update(
            {
                f"{target}_target",
                f"{target}_valid",
            }
        )

    missing = required.difference(
        dataframe.columns
    )

    if missing:
        raise RuntimeError(
            f"Faltan columnas de targets: {sorted(missing)}"
        )

    dataframe = dataframe.loc[
        dataframe["system"] == system
    ].copy()

    if len(dataframe) != 3:
        raise RuntimeError(
            f"Se esperaban tres réplicas y hay "
            f"{len(dataframe)}."
        )

    signatures = (
        dataframe["target_mask_signature"]
        .astype(str)
    )

    valid_signatures = signatures.str.fullmatch(
        r"[01]{5}"
    )

    if not valid_signatures.all():
        invalid = signatures.loc[
            ~valid_signatures
        ].tolist()

        raise RuntimeError(
            "Firmas de máscara no válidas: "
            f"{invalid}"
        )

    return dataframe.sort_values(
        "replica"
    ).reset_index(drop=True)


def build_target_arrays(
    targets: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    target_values = np.full(
        (
            len(targets),
            len(TARGET_COLUMNS),
        ),
        np.nan,
        dtype=np.float64,
    )

    target_mask = np.zeros(
        (
            len(targets),
            len(TARGET_COLUMNS),
        ),
        dtype=bool,
    )

    for row_index, row in targets.iterrows():
        for target_index, target in enumerate(
            TARGET_COLUMNS
        ):
            valid = normalize_boolean(
                row[f"{target}_valid"]
            )
            value = row[f"{target}_target"]

            target_mask[
                row_index,
                target_index,
            ] = valid

            if valid:
                if not np.isfinite(float(value)):
                    raise RuntimeError(
                        f"Target válido no finito: "
                        f"R{row['replica']} {target}"
                    )

                target_values[
                    row_index,
                    target_index,
                ] = float(value)
            else:
                if pd.notna(value):
                    raise RuntimeError(
                        f"Target inválido con valor: "
                        f"R{row['replica']} {target}"
                    )

    return target_values, target_mask


def main() -> int:
    args = parse_args()

    basic_dir = args.basic_dir.resolve()
    core_dir = args.core_rmsd_dir.resolve()
    sasa_dir = args.sasa_dssp_dir.resolve()
    target_file = args.master_targets.resolve()
    output_dir = args.output_dir.resolve()

    replicas = np.asarray(
        parse_replicas(args.replicas),
        dtype=np.int64,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    sequences: list[pd.DataFrame] = []
    manifest_records: list[dict[str, object]] = []
    feature_summary_records: list[dict[str, object]] = []

    for replica in replicas.tolist():
        (
            sequence,
            manifest,
            summaries,
        ) = assemble_replica(
            system=args.system,
            replica=replica,
            basic_dir=basic_dir,
            core_dir=core_dir,
            sasa_dir=sasa_dir,
            output_dir=output_dir,
            prefix_end_ns=args.prefix_end_ns,
            expected_step_ps=args.expected_step_ps,
        )

        sequences.append(sequence)
        manifest_records.append(manifest)
        feature_summary_records.extend(summaries)

    reference_times = sequences[0][
        "time_ns"
    ].to_numpy(dtype=float)

    for replica, sequence in zip(
        replicas[1:],
        sequences[1:],
        strict=True,
    ):
        candidate_times = sequence[
            "time_ns"
        ].to_numpy(dtype=float)

        if not np.allclose(
            reference_times,
            candidate_times,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"R{int(replica)}: la rejilla temporal "
                f"difiere de R{int(replicas[0])}."
            )

    X_raw = np.stack(
        [
            sequence[
                FEATURE_COLUMNS
            ].to_numpy(dtype=np.float64)
            for sequence in sequences
        ],
        axis=0,
    )

    targets = load_targets(
        filename=target_file,
        system=args.system,
    )

    requested_replicas = [
        int(replica)
        for replica in replicas.tolist()
    ]

    targets = targets.loc[
        targets["replica"]
        .astype(int)
        .isin(requested_replicas)
    ].copy()

    targets["replica"] = (
        targets["replica"].astype(int)
    )

    targets = (
        targets.sort_values("replica")
        .reset_index(drop=True)
    )

    observed_replicas = (
        targets["replica"].tolist()
    )

    if observed_replicas != requested_replicas:
        raise RuntimeError(
            "Las réplicas de los targets no coinciden. "
            f"Esperadas: {requested_replicas}; "
            f"observadas: {observed_replicas}."
        )

    target_values, target_mask = (
        build_target_arrays(targets)
    )

    sample_ids = np.array(
        [
            f"{args.system}_R{replica}"
            for replica in replicas
        ],
        dtype="U64",
    )

    feature_names = np.array(
        FEATURE_COLUMNS,
        dtype="U64",
    )
    target_names = np.array(
        TARGET_COLUMNS,
        dtype="U32",
    )

    npz_file = output_dir / "prefix_sequences_raw.npz"

    np.savez_compressed(
        npz_file,
        X_raw=X_raw,
        time_ns=reference_times,
        replicas=replicas,
        sample_ids=sample_ids,
        feature_names=feature_names,
        target_values=target_values,
        target_mask=target_mask,
        target_names=target_names,
    )

    manifest = pd.DataFrame(
        manifest_records
    ).merge(
        targets[
            [
                "system",
                "replica",
                "target_mask_signature",
                "n_valid_targets",
            ]
        ],
        on=["system", "replica"],
        how="left",
        validate="one_to_one",
    )

    manifest_file = (
        output_dir / "prefix_sequence_manifest.csv"
    )
    feature_summary_file = (
        output_dir / "prefix_feature_summary.csv"
    )
    metadata_file = (
        output_dir / "prefix_sequences_metadata.json"
    )

    manifest.to_csv(
        manifest_file,
        index=False,
    )
    pd.DataFrame(
        feature_summary_records
    ).to_csv(
        feature_summary_file,
        index=False,
    )

    metadata_file.write_text(
        json.dumps(
            {
                "system": args.system,
                "prefix_start_ns": 0.0,
                "prefix_end_ns": args.prefix_end_ns,
                "temporal_resolution_ps": (
                    args.expected_step_ps
                ),
                "n_steps": int(X_raw.shape[1]),
                "feature_order": FEATURE_COLUMNS,
                "target_order": TARGET_COLUMNS,
                "array_shapes": {
                    "X_raw": list(X_raw.shape),
                    "target_values": list(
                        target_values.shape
                    ),
                    "target_mask": list(
                        target_mask.shape
                    ),
                    "time_ns": list(
                        reference_times.shape
                    ),
                },
                "leakage_policy": {
                    "maximum_input_time_ns": (
                        args.prefix_end_ns
                    ),
                    "core_rmsd_variant": (
                        "replica_prefix_rmsd_A"
                    ),
                    "full_oracle_features_included": False,
                    "target_values_in_input": False,
                },
                "normalization_policy": (
                    "Raw values only. Fit scalers using "
                    "the future training split, never using "
                    "validation or test proteins."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 110)
    print("SECUENCIAS DEL PREFIJO")
    print("=" * 110)
    print(f"Forma X_raw:         {X_raw.shape}")
    print(f"Forma targets:       {target_values.shape}")
    print(f"Forma máscara:       {target_mask.shape}")
    print(f"Pasos temporales:    {X_raw.shape[1]}")
    print(f"Características:     {X_raw.shape[2]}")
    print(f"Tiempo inicial:      {reference_times[0]:.1f} ns")
    print(f"Tiempo final:        {reference_times[-1]:.1f} ns")
    print()

    print("Orden de características:")
    for index, feature in enumerate(
        FEATURE_COLUMNS
    ):
        print(f"  {index}: {feature}")

    print()
    print("Targets y máscaras:")
    for sample_index, sample_id in enumerate(
        sample_ids
    ):
        values = target_values[sample_index]
        mask = target_mask[sample_index]

        print(
            f"{sample_id}: "
            f"mask={''.join('1' if x else '0' for x in mask)}"
        )

        for target_index, target_name in enumerate(
            TARGET_COLUMNS
        ):
            print(
                f"  {target_name:<8} "
                f"valid={bool(mask[target_index])!s:<5} "
                f"value={values[target_index]}"
            )

    print()
    print("=" * 110)
    print("MANIFIESTO")
    print("=" * 110)
    print(manifest.to_string(index=False))

    print()
    print(f"NPZ:        {npz_file}")
    print(f"Manifiesto: {manifest_file}")
    print(f"Resumen:    {feature_summary_file}")
    print(f"Metadatos:  {metadata_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
