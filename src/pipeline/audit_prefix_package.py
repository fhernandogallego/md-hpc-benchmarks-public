#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "rmsd_backbone_A",
    "radius_gyration_A",
    "core_rmsd_q30_A",
    "core_rmsd_q40_A",
    "core_rmsd_q50_A",
    "total_sasa_A2",
    "helix_fraction",
    "strand_fraction",
)

TARGET_NAMES = (
    "rmsd",
    "rg",
    "sasa",
    "helix",
    "strand",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audita la consistencia interna, las máscaras "
            "y la política antileakage de un paquete de prefijos."
        )
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--replicas", default="1,2,3")
    parser.add_argument(
        "--package-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--master-targets",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-masks",
        default="",
        help=(
            "Máscaras opcionales con formato "
            "1:11100,2:01111,3:10111"
        ),
    )
    parser.add_argument(
        "--prefix-end-ns",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--step-ps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=1e-10,
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
            "Las réplicas deben ser positivas."
        )

    return replicas


def parse_expected_masks(
    value: str,
) -> dict[int, str]:
    if not value.strip():
        return {}

    masks: dict[int, str] = {}

    for item in value.split(","):
        replica_text, signature = (
            item.strip().split(":", maxsplit=1)
        )

        replica = int(replica_text)
        signature = signature.strip()

        if (
            len(signature) != len(TARGET_NAMES)
            or set(signature).difference({"0", "1"})
        ):
            raise ValueError(
                f"Máscara inválida para R{replica}: "
                f"{signature!r}"
            )

        masks[replica] = signature

    return masks


def normalize_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Booleano no reconocido: {value!r}"
    )


def sha256_file(filename: Path) -> str:
    digest = hashlib.sha256()

    with filename.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def run_audit(
    args: argparse.Namespace,
) -> dict[str, object]:
    package_dir = args.package_dir.resolve()
    master_file = args.master_targets.resolve()

    replicas = parse_replicas(
        args.replicas
    )
    expected_masks = parse_expected_masks(
        args.expected_masks
    )

    checks: list[dict[str, object]] = []

    def check(
        condition: bool,
        name: str,
        details: object = None,
    ) -> None:
        record = {
            "check": name,
            "pass": bool(condition),
        }

        if details is not None:
            record["details"] = details

        checks.append(record)

        if not condition:
            raise RuntimeError(
                f"Fallo en auditoría: {name}. "
                f"Detalles: {details}"
            )

    sequence_files = [
        package_dir
        / (
            f"{args.system}_R{replica}"
            "_prefix_sequence_100ps.csv"
        )
        for replica in replicas
    ]

    required_files = [
        *sequence_files,
        package_dir
        / "prefix_feature_summary.csv",
        package_dir
        / "prefix_sequence_manifest.csv",
        package_dir
        / "prefix_sequences_metadata.json",
        package_dir
        / "prefix_sequences_raw.npz",
        master_file,
    ]

    for filename in required_files:
        check(
            filename.is_file()
            and filename.stat().st_size > 0,
            f"archivo presente: {filename.name}",
        )

    metadata_file = (
        package_dir
        / "prefix_sequences_metadata.json"
    )

    metadata = json.loads(
        metadata_file.read_text(
            encoding="utf-8"
        )
    )

    expected_n_steps_float = (
        args.prefix_end_ns
        * 1000.0
        / args.step_ps
    )

    check(
        abs(
            expected_n_steps_float
            - round(expected_n_steps_float)
        )
        <= args.numeric_tolerance,
        "la duración es divisible por el paso temporal",
        expected_n_steps_float,
    )

    expected_n_steps = (
        int(round(expected_n_steps_float))
        + 1
    )

    expected_time_ps = (
        np.arange(
            expected_n_steps,
            dtype=np.int64,
        )
        * args.step_ps
    )

    expected_time_ns = (
        expected_time_ps.astype(float)
        / 1000.0
    )

    npz_file = (
        package_dir
        / "prefix_sequences_raw.npz"
    )

    with np.load(
        npz_file,
        allow_pickle=False,
    ) as archive:
        required_arrays = {
            "X_raw",
            "time_ns",
            "replicas",
            "sample_ids",
            "feature_names",
            "target_values",
            "target_mask",
            "target_names",
        }

        check(
            set(archive.files) == required_arrays,
            "arrays requeridos en NPZ",
            sorted(archive.files),
        )

        X_raw = archive["X_raw"].copy()
        time_ns = archive["time_ns"].copy()
        npz_replicas = archive[
            "replicas"
        ].astype(int)
        sample_ids = archive[
            "sample_ids"
        ].astype(str)
        feature_names = archive[
            "feature_names"
        ].astype(str)
        target_values = archive[
            "target_values"
        ].copy()
        target_mask = archive[
            "target_mask"
        ].astype(bool)
        target_names = archive[
            "target_names"
        ].astype(str)

    expected_shapes = {
        "X_raw": (
            len(replicas),
            expected_n_steps,
            len(FEATURE_NAMES),
        ),
        "target_values": (
            len(replicas),
            len(TARGET_NAMES),
        ),
        "target_mask": (
            len(replicas),
            len(TARGET_NAMES),
        ),
        "time_ns": (
            expected_n_steps,
        ),
    }

    check(
        X_raw.shape
        == expected_shapes["X_raw"],
        "forma de X_raw",
        {
            "observed": list(X_raw.shape),
            "expected": list(
                expected_shapes["X_raw"]
            ),
        },
    )

    check(
        target_values.shape
        == expected_shapes["target_values"],
        "forma de target_values",
        list(target_values.shape),
    )

    check(
        target_mask.shape
        == expected_shapes["target_mask"],
        "forma de target_mask",
        list(target_mask.shape),
    )

    check(
        time_ns.shape
        == expected_shapes["time_ns"],
        "forma de time_ns",
        list(time_ns.shape),
    )

    check(
        X_raw.dtype == np.float64,
        "X_raw usa float64",
        str(X_raw.dtype),
    )

    check(
        target_values.dtype == np.float64,
        "target_values usa float64",
        str(target_values.dtype),
    )

    check(
        tuple(feature_names)
        == FEATURE_NAMES,
        "orden exacto de features",
        feature_names.tolist(),
    )

    check(
        tuple(target_names)
        == TARGET_NAMES,
        "orden exacto de targets",
        target_names.tolist(),
    )

    check(
        np.array_equal(
            npz_replicas,
            np.asarray(
                replicas,
                dtype=int,
            ),
        ),
        "orden de réplicas",
        npz_replicas.tolist(),
    )

    expected_sample_ids = np.asarray(
        [
            f"{args.system}_R{replica}"
            for replica in replicas
        ]
    )

    check(
        np.array_equal(
            sample_ids,
            expected_sample_ids,
        ),
        "sample_ids",
        sample_ids.tolist(),
    )

    check(
        np.allclose(
            time_ns,
            expected_time_ns,
            rtol=0.0,
            atol=args.numeric_tolerance,
        ),
        "rejilla temporal exacta del NPZ",
        {
            "start_ns": float(time_ns[0]),
            "end_ns": float(time_ns[-1]),
            "n_steps": int(len(time_ns)),
        },
    )

    check(
        float(np.max(time_ns))
        <= (
            args.prefix_end_ns
            + args.numeric_tolerance
        ),
        "ningún input supera el final del prefijo",
        float(np.max(time_ns)),
    )

    forbidden_feature_tokens = (
        "target",
        "mask",
        "valid",
        "neff",
        "t0",
        "tail",
        "oracle",
        "full_",
    )

    forbidden_features = [
        feature
        for feature in feature_names
        if any(
            token in feature.lower()
            for token in forbidden_feature_tokens
        )
    ]

    check(
        not forbidden_features,
        "ausencia de variables de target o equilibrio "
        "en las features",
        forbidden_features,
    )

    check(
        "normalized_time"
        not in set(feature_names),
        "normalized_time no está incluido en X_raw",
    )

    check(
        set(feature_names).isdisjoint(
            set(target_names)
        ),
        "features y targets son conjuntos separados",
    )

    check(
        metadata.get("system")
        == args.system,
        "sistema de metadatos",
        metadata.get("system"),
    )

    check(
        float(
            metadata.get(
                "prefix_start_ns"
            )
        )
        == 0.0,
        "inicio del prefijo en metadatos",
    )

    check(
        abs(
            float(
                metadata.get(
                    "prefix_end_ns"
                )
            )
            - args.prefix_end_ns
        )
        <= args.numeric_tolerance,
        "final del prefijo en metadatos",
    )

    check(
        int(
            metadata.get(
                "temporal_resolution_ps"
            )
        )
        == args.step_ps,
        "resolución temporal en metadatos",
    )

    check(
        int(metadata.get("n_steps"))
        == expected_n_steps,
        "número de pasos en metadatos",
    )

    check(
        tuple(
            metadata.get(
                "feature_order",
                [],
            )
        )
        == FEATURE_NAMES,
        "orden de features en metadatos",
    )

    check(
        tuple(
            metadata.get(
                "target_order",
                [],
            )
        )
        == TARGET_NAMES,
        "orden de targets en metadatos",
    )

    metadata_shapes = metadata.get(
        "array_shapes",
        {},
    )

    for array_name, expected_shape in (
        expected_shapes.items()
    ):
        check(
            metadata_shapes.get(
                array_name
            )
            == list(expected_shape),
            f"forma declarada de {array_name}",
            metadata_shapes.get(array_name),
        )

    leakage_policy = metadata.get(
        "leakage_policy",
        {},
    )

    check(
        abs(
            float(
                leakage_policy.get(
                    "maximum_input_time_ns"
                )
            )
            - args.prefix_end_ns
        )
        <= args.numeric_tolerance,
        "maximum_input_time_ns",
    )

    check(
        leakage_policy.get(
            "core_rmsd_variant"
        )
        == "replica_prefix_rmsd_A",
        "core-RMSD usa la variante del prefijo",
        leakage_policy.get(
            "core_rmsd_variant"
        ),
    )

    check(
        leakage_policy.get(
            "full_oracle_features_included"
        )
        is False,
        "features oracle excluidas",
    )

    check(
        leakage_policy.get(
            "target_values_in_input"
        )
        is False,
        "targets excluidos de X_raw",
    )

    normalization_policy = str(
        metadata.get(
            "normalization_policy",
            "",
        )
    )

    check(
        "Raw values only"
        in normalization_policy,
        "política de datos crudos",
        normalization_policy,
    )

    manifest_file = (
        package_dir
        / "prefix_sequence_manifest.csv"
    )

    manifest = pd.read_csv(
        manifest_file,
        dtype={
            "system": "string",
            "sample_id": "string",
            "sequence_csv": "string",
            "target_mask_signature": "string",
        },
    )

    manifest = (
        manifest.sort_values("replica")
        .reset_index(drop=True)
    )

    check(
        len(manifest)
        == len(replicas),
        "número de filas del manifiesto",
        len(manifest),
    )

    check(
        manifest["replica"]
        .astype(int)
        .tolist()
        == replicas,
        "réplicas del manifiesto",
    )

    check(
        manifest["sample_id"]
        .astype(str)
        .tolist()
        == expected_sample_ids.tolist(),
        "sample_ids del manifiesto",
    )

    check(
        (
            manifest["n_steps"]
            .astype(int)
            == expected_n_steps
        ).all(),
        "n_steps del manifiesto",
    )

    check(
        (
            manifest["n_features"]
            .astype(int)
            == len(FEATURE_NAMES)
        ).all(),
        "n_features del manifiesto",
    )

    check(
        np.allclose(
            manifest["start_ns"]
            .to_numpy(dtype=float),
            0.0,
            rtol=0.0,
            atol=args.numeric_tolerance,
        ),
        "inicio temporal del manifiesto",
    )

    check(
        np.allclose(
            manifest["end_ns"]
            .to_numpy(dtype=float),
            args.prefix_end_ns,
            rtol=0.0,
            atol=args.numeric_tolerance,
        ),
        "final temporal del manifiesto",
    )

    check(
        (
            manifest["step_ps"]
            .astype(int)
            == args.step_ps
        ).all(),
        "paso temporal del manifiesto",
    )

    for quantile in (30, 40, 50):
        column = (
            "max_backbone_difference_"
            f"q{quantile}_A"
        )

        check(
            float(
                manifest[column]
                .abs()
                .max()
            )
            <= args.numeric_tolerance,
            f"consistencia backbone/core q{quantile}",
            float(
                manifest[column]
                .abs()
                .max()
            ),
        )

    sequence_matrices: list[np.ndarray] = []

    required_sequence_columns = {
        "system",
        "replica",
        "sample_id",
        "step_index",
        "time_ps",
        "time_ns",
        "normalized_time",
        *FEATURE_NAMES,
    }

    for sample_index, (
        replica,
        filename,
    ) in enumerate(
        zip(
            replicas,
            sequence_files,
            strict=True,
        )
    ):
        dataframe = pd.read_csv(
            filename
        )

        missing = (
            required_sequence_columns
            .difference(dataframe.columns)
        )

        check(
            not missing,
            f"columnas del CSV R{replica}",
            sorted(missing),
        )

        check(
            len(dataframe)
            == expected_n_steps,
            f"número de pasos del CSV R{replica}",
            len(dataframe),
        )

        check(
            set(
                dataframe["system"]
                .astype(str)
                .unique()
            )
            == {args.system},
            f"sistema del CSV R{replica}",
        )

        check(
            set(
                dataframe["replica"]
                .astype(int)
                .unique()
            )
            == {replica},
            f"réplica del CSV R{replica}",
        )

        check(
            set(
                dataframe["sample_id"]
                .astype(str)
                .unique()
            )
            == {
                f"{args.system}_R{replica}"
            },
            f"sample_id del CSV R{replica}",
        )

        check(
            np.array_equal(
                dataframe["step_index"]
                .to_numpy(dtype=int),
                np.arange(
                    expected_n_steps,
                    dtype=int,
                ),
            ),
            f"step_index del CSV R{replica}",
        )

        check(
            np.array_equal(
                dataframe["time_ps"]
                .to_numpy(dtype=np.int64),
                expected_time_ps,
            ),
            f"time_ps del CSV R{replica}",
        )

        check(
            np.allclose(
                dataframe["time_ns"]
                .to_numpy(dtype=float),
                expected_time_ns,
                rtol=0.0,
                atol=args.numeric_tolerance,
            ),
            f"time_ns del CSV R{replica}",
        )

        expected_normalized_time = (
            expected_time_ns
            / args.prefix_end_ns
        )

        check(
            np.allclose(
                dataframe[
                    "normalized_time"
                ].to_numpy(dtype=float),
                expected_normalized_time,
                rtol=0.0,
                atol=args.numeric_tolerance,
            ),
            f"normalized_time del CSV R{replica}",
        )

        matrix = dataframe[
            list(FEATURE_NAMES)
        ].to_numpy(dtype=np.float64)

        sequence_matrices.append(
            matrix
        )

        check(
            np.all(
                np.isfinite(matrix)
            ),
            f"features finitas del CSV R{replica}",
        )

        check(
            np.allclose(
                matrix,
                X_raw[sample_index],
                rtol=0.0,
                atol=args.numeric_tolerance,
            ),
            f"igualdad CSV/NPZ R{replica}",
            float(
                np.max(
                    np.abs(
                        matrix
                        - X_raw[sample_index]
                    )
                )
            ),
        )

        expected_sequence_name = (
            f"{args.system}_R{replica}"
            "_prefix_sequence_100ps.csv"
        )

        observed_sequence_name = Path(
            str(
                manifest.loc[
                    sample_index,
                    "sequence_csv",
                ]
            )
        ).name

        check(
            observed_sequence_name
            == expected_sequence_name,
            f"ruta del manifiesto R{replica}",
            observed_sequence_name,
        )

    stacked_csv = np.stack(
        sequence_matrices,
        axis=0,
    )

    check(
        np.allclose(
            stacked_csv,
            X_raw,
            rtol=0.0,
            atol=args.numeric_tolerance,
        ),
        "X_raw es la concatenación de los CSV crudos",
    )

    summary_file = (
        package_dir
        / "prefix_feature_summary.csv"
    )

    feature_summary = pd.read_csv(
        summary_file
    )

    check(
        len(feature_summary)
        == (
            len(replicas)
            * len(FEATURE_NAMES)
        ),
        "número de filas del resumen de features",
        len(feature_summary),
    )

    for sample_index, replica in enumerate(
        replicas
    ):
        for feature_index, feature in enumerate(
            FEATURE_NAMES
        ):
            subset = feature_summary.loc[
                (
                    feature_summary["replica"]
                    .astype(int)
                    == replica
                )
                & (
                    feature_summary["feature"]
                    .astype(str)
                    == feature
                )
            ]

            check(
                len(subset) == 1,
                (
                    "fila única del resumen "
                    f"R{replica} {feature}"
                ),
                len(subset),
            )

            row = subset.iloc[0]
            values = X_raw[
                sample_index,
                :,
                feature_index,
            ]

            expected_statistics = {
                "n_steps": int(
                    len(values)
                ),
                "mean": float(
                    np.mean(values)
                ),
                "std": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
                "min": float(
                    np.min(values)
                ),
                "max": float(
                    np.max(values)
                ),
                "first": float(
                    values[0]
                ),
                "last": float(
                    values[-1]
                ),
                "last_minus_first": float(
                    values[-1]
                    - values[0]
                ),
            }

            check(
                int(row["n_steps"])
                == expected_statistics[
                    "n_steps"
                ],
                (
                    "n_steps del resumen "
                    f"R{replica} {feature}"
                ),
            )

            for statistic in (
                "mean",
                "std",
                "min",
                "max",
                "first",
                "last",
                "last_minus_first",
            ):
                observed = float(
                    row[statistic]
                )
                expected = (
                    expected_statistics[
                        statistic
                    ]
                )

                check(
                    abs(observed - expected)
                    <= args.numeric_tolerance,
                    (
                        f"{statistic} del resumen "
                        f"R{replica} {feature}"
                    ),
                    {
                        "observed": observed,
                        "expected": expected,
                    },
                )

    master = pd.read_csv(
        master_file,
        dtype={
            "system": "string",
            "target_mask_signature": "string",
        },
    )

    master = master.loc[
        (
            master["system"]
            .astype(str)
            == args.system
        )
        & (
            master["replica"]
            .astype(int)
            .isin(replicas)
        )
    ].copy()

    master["replica"] = (
        master["replica"].astype(int)
    )

    master = (
        master.sort_values("replica")
        .reset_index(drop=True)
    )

    check(
        master["replica"].tolist()
        == replicas,
        "réplicas de master_targets",
        master["replica"].tolist(),
    )

    observed_signatures: dict[int, str] = {}

    for sample_index, replica in enumerate(
        replicas
    ):
        row = master.iloc[
            sample_index
        ]

        reconstructed_bits: list[str] = []

        for target_index, target in enumerate(
            TARGET_NAMES
        ):
            valid_column = (
                f"{target}_valid"
            )
            value_column = (
                f"{target}_target"
            )

            valid = normalize_boolean(
                row[valid_column]
            )
            value = row[value_column]

            reconstructed_bits.append(
                "1" if valid else "0"
            )

            check(
                bool(
                    target_mask[
                        sample_index,
                        target_index,
                    ]
                )
                == valid,
                (
                    "máscara NPZ/master "
                    f"R{replica} {target}"
                ),
            )

            if valid:
                check(
                    pd.notna(value)
                    and np.isfinite(
                        float(value)
                    ),
                    (
                        "target válido finito "
                        f"R{replica} {target}"
                    ),
                    value,
                )

                check(
                    np.isfinite(
                        target_values[
                            sample_index,
                            target_index,
                        ]
                    ),
                    (
                        "target válido finito "
                        f"en NPZ R{replica} {target}"
                    ),
                )

                check(
                    abs(
                        float(value)
                        - float(
                            target_values[
                                sample_index,
                                target_index,
                            ]
                        )
                    )
                    <= args.numeric_tolerance,
                    (
                        "target NPZ/master "
                        f"R{replica} {target}"
                    ),
                )
            else:
                check(
                    pd.isna(value),
                    (
                        "target inválido NaN "
                        f"en master R{replica} {target}"
                    ),
                    value,
                )

                check(
                    np.isnan(
                        target_values[
                            sample_index,
                            target_index,
                        ]
                    ),
                    (
                        "target inválido NaN "
                        f"en NPZ R{replica} {target}"
                    ),
                )

        signature = "".join(
            reconstructed_bits
        )

        observed_signatures[
            replica
        ] = signature

        check(
            str(
                row[
                    "target_mask_signature"
                ]
            )
            == signature,
            f"firma de máscara en master R{replica}",
            signature,
        )

        check(
            str(
                manifest.loc[
                    sample_index,
                    "target_mask_signature",
                ]
            )
            == signature,
            (
                "firma de máscara en manifiesto "
                f"R{replica}"
            ),
            signature,
        )

        check(
            int(
                manifest.loc[
                    sample_index,
                    "n_valid_targets",
                ]
            )
            == signature.count("1"),
            (
                "n_valid_targets del manifiesto "
                f"R{replica}"
            ),
        )

        if replica in expected_masks:
            check(
                signature
                == expected_masks[replica],
                f"máscara esperada R{replica}",
                {
                    "observed": signature,
                    "expected": (
                        expected_masks[
                            replica
                        ]
                    ),
                },
            )

    check(
        int(target_mask.sum()) == int(np.isfinite(target_values).sum()),
        "número total de targets válidos",
        int(target_mask.sum()),
    )

    return {
        "overall_pass": True,
        "system": args.system,
        "replicas": replicas,
        "package_dir": str(
            package_dir
        ),
        "prefix_end_ns": (
            args.prefix_end_ns
        ),
        "step_ps": args.step_ps,
        "n_steps": expected_n_steps,
        "feature_order": list(
            FEATURE_NAMES
        ),
        "target_order": list(
            TARGET_NAMES
        ),
        "array_shapes": {
            "X_raw": list(
                X_raw.shape
            ),
            "target_values": list(
                target_values.shape
            ),
            "target_mask": list(
                target_mask.shape
            ),
            "time_ns": list(
                time_ns.shape
            ),
        },
        "mask_signatures": {
            str(replica): signature
            for replica, signature in (
                observed_signatures.items()
            )
        },
        "n_valid_targets": int(
            target_mask.sum()
        ),
        "maximum_input_time_ns": float(
            np.max(time_ns)
        ),
        "core_rmsd_variant": (
            leakage_policy.get(
                "core_rmsd_variant"
            )
        ),
        "full_oracle_features_included": (
            leakage_policy.get(
                "full_oracle_features_included"
            )
        ),
        "target_values_in_input": (
            leakage_policy.get(
                "target_values_in_input"
            )
        ),
        "checks_passed": len(checks),
        "checks": checks,
    }


def main() -> int:
    args = parse_args()

    package_dir = args.package_dir.resolve()
    audit_dir = (
        package_dir / "audit"
    )
    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_file = (
        audit_dir
        / "prefix_package_audit.json"
    )
    checksum_file = (
        audit_dir
        / "prefix_package_checksums.sha256"
    )

    try:
        result = run_audit(args)
        exit_code = 0
    except Exception as error:
        result = {
            "overall_pass": False,
            "system": args.system,
            "package_dir": str(
                package_dir
            ),
            "error_type": type(
                error
            ).__name__,
            "error": str(error),
        }
        exit_code = 1

    audit_file.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    files_to_hash = sorted(
        [
            filename
            for filename in (
                package_dir.glob("*")
            )
            if filename.is_file()
        ]
        + [audit_file]
    )

    checksum_lines = [
        (
            f"{sha256_file(filename)}  "
            f"{filename.relative_to(package_dir)}"
        )
        for filename in files_to_hash
    ]

    checksum_file.write_text(
        "\n".join(checksum_lines)
        + "\n",
        encoding="utf-8",
    )

    print("=" * 110)
    print(
        "AUDITORÍA DEL PAQUETE DE PREFIJOS"
    )
    print("=" * 110)

    if result["overall_pass"]:
        print("Estado:                    PASS")
        print(
            "Comprobaciones superadas: "
            f"{result['checks_passed']}"
        )
        print(
            "Forma X_raw:               "
            f"{tuple(result['array_shapes']['X_raw'])}"
        )
        print(
            "Forma targets:             "
            f"{tuple(result['array_shapes']['target_values'])}"
        )
        print(
            "Forma máscara:             "
            f"{tuple(result['array_shapes']['target_mask'])}"
        )
        print(
            "Tiempo máximo de entrada:  "
            f"{result['maximum_input_time_ns']:.1f} ns"
        )
        print(
            "Targets válidos:           "
            f"{result['n_valid_targets']}"
        )
        print(
            "Máscaras:                  "
            f"{result['mask_signatures']}"
        )
        print(
            "Core-RMSD:                 "
            f"{result['core_rmsd_variant']}"
        )
        print(
            "Oracle incluido:           "
            f"{result['full_oracle_features_included']}"
        )
        print(
            "Targets incluidos en X:    "
            f"{result['target_values_in_input']}"
        )
    else:
        print("Estado: FAIL")
        print(
            f"Error:  {result['error']}"
        )

    print()
    print(f"Auditoría: {audit_file}")
    print(f"Checksums: {checksum_file}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
