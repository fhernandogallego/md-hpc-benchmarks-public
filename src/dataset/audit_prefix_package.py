#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FEATURES = [
    "rmsd_backbone_A",
    "radius_gyration_A",
    "core_rmsd_q30_A",
    "core_rmsd_q40_A",
    "core_rmsd_q50_A",
    "total_sasa_A2",
    "helix_fraction",
    "strand_fraction",
]

EXPECTED_TARGETS = [
    "rmsd",
    "rg",
    "sasa",
    "helix",
    "strand",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita el paquete de secuencias del prefijo."
    )
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-steps", type=int, default=201)
    parser.add_argument("--expected-step-ps", type=int, default=100)
    parser.add_argument("--maximum-time-ns", type=float, default=20.0)
    return parser.parse_args()


def sha256(filename: Path) -> str:
    digest = hashlib.sha256()

    with filename.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    args = parse_args()

    npz_file = args.npz.resolve()
    manifest_file = args.manifest.resolve()
    metadata_file = args.metadata.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    package = np.load(
        npz_file,
        allow_pickle=False,
    )

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

    missing_arrays = required_arrays.difference(
        package.files
    )
    require(
        not missing_arrays,
        f"Faltan arrays en NPZ: {sorted(missing_arrays)}",
    )

    X_raw = package["X_raw"]
    time_ns = package["time_ns"]
    replicas = package["replicas"]
    sample_ids = [
        str(value)
        for value in package["sample_ids"]
    ]
    feature_names = [
        str(value)
        for value in package["feature_names"]
    ]
    target_values = package["target_values"]
    target_mask = package["target_mask"].astype(bool)
    target_names = [
        str(value)
        for value in package["target_names"]
    ]

    require(
        X_raw.ndim == 3,
        f"X_raw no tiene tres dimensiones: {X_raw.shape}",
    )
    require(
        X_raw.shape[1] == args.expected_steps,
        f"Número incorrecto de pasos: {X_raw.shape}",
    )
    require(
        X_raw.shape[2] == len(EXPECTED_FEATURES),
        f"Número incorrecto de características: {X_raw.shape}",
    )
    require(
        feature_names == EXPECTED_FEATURES,
        f"Orden de características incorrecto: {feature_names}",
    )
    require(
        target_names == EXPECTED_TARGETS,
        f"Orden de targets incorrecto: {target_names}",
    )
    require(
        target_values.shape == target_mask.shape,
        "Targets y máscaras tienen formas diferentes.",
    )
    require(
        target_values.shape == (
            X_raw.shape[0],
            len(EXPECTED_TARGETS),
        ),
        f"Forma de targets incorrecta: {target_values.shape}",
    )

    require(
        np.all(np.isfinite(X_raw)),
        "X_raw contiene NaN o infinitos.",
    )
    require(
        np.all(np.isfinite(target_values[target_mask])),
        "Hay targets válidos no finitos.",
    )
    require(
        np.all(np.isnan(target_values[~target_mask])),
        "Hay targets inválidos que no son NaN.",
    )

    require(
        len(time_ns) == args.expected_steps,
        "La longitud temporal no coincide.",
    )
    require(
        np.isclose(time_ns[0], 0.0),
        f"El tiempo inicial no es cero: {time_ns[0]}",
    )
    require(
        np.isclose(
            time_ns[-1],
            args.maximum_time_ns,
        ),
        f"El tiempo final no es 20 ns: {time_ns[-1]}",
    )

    expected_increment_ns = (
        args.expected_step_ps / 1000.0
    )
    require(
        np.allclose(
            np.diff(time_ns),
            expected_increment_ns,
            rtol=0.0,
            atol=1e-12,
        ),
        "La rejilla temporal no es uniforme.",
    )

    manifest = pd.read_csv(
        manifest_file,
        dtype={"target_mask_signature": "string"},
    )

    require(
        len(manifest) == X_raw.shape[0],
        "El manifiesto no tiene una fila por muestra.",
    )
    require(
        manifest["sample_id"].is_unique,
        "Hay sample_id duplicados.",
    )

    metadata = json.loads(
        metadata_file.read_text(encoding="utf-8")
    )

    require(
        metadata["leakage_policy"][
            "full_oracle_features_included"
        ] is False,
        "Los metadatos indican presencia de features oracle.",
    )
    require(
        float(
            metadata["leakage_policy"][
                "maximum_input_time_ns"
            ]
        )
        <= args.maximum_time_ns,
        "Los metadatos permiten información posterior a 20 ns.",
    )

    sequence_checks: list[dict[str, object]] = []
    checksum_files = [
        npz_file,
        manifest_file,
        metadata_file,
    ]

    for sample_index, sample_id in enumerate(sample_ids):
        matching = manifest.loc[
            manifest["sample_id"] == sample_id
        ]

        require(
            len(matching) == 1,
            f"{sample_id}: no existe una fila única en el manifiesto.",
        )

        row = matching.iloc[0]
        sequence_file = Path(
            str(row["sequence_csv"])
        )

        require(
            sequence_file.is_file(),
            f"No existe {sequence_file}",
        )

        checksum_files.append(sequence_file)

        sequence = pd.read_csv(sequence_file)

        require(
            len(sequence) == args.expected_steps,
            f"{sample_id}: longitud CSV incorrecta.",
        )
        require(
            set(EXPECTED_FEATURES).issubset(
                sequence.columns
            ),
            f"{sample_id}: faltan características.",
        )
        require(
            not any(
                "oracle" in column.lower()
                for column in sequence.columns
            ),
            f"{sample_id}: hay una columna oracle.",
        )
        require(
            float(sequence["time_ns"].max())
            <= args.maximum_time_ns + 1e-12,
            f"{sample_id}: contiene tiempos posteriores a 20 ns.",
        )

        csv_values = sequence[
            EXPECTED_FEATURES
        ].to_numpy(dtype=float)

        maximum_array_difference = float(
            np.max(
                np.abs(
                    csv_values
                    - X_raw[sample_index]
                )
            )
        )

        require(
            maximum_array_difference <= 1e-10,
            (
                f"{sample_id}: CSV y NPZ difieren hasta "
                f"{maximum_array_difference}"
            ),
        )

        expected_signature = "".join(
            "1" if valid else "0"
            for valid in target_mask[sample_index]
        )
        manifest_signature = str(
            row["target_mask_signature"]
        )

        require(
            manifest_signature == expected_signature,
            (
                f"{sample_id}: firma {manifest_signature}, "
                f"esperada {expected_signature}"
            ),
        )

        sequence_checks.append(
            {
                "sample_id": sample_id,
                "replica": int(replicas[sample_index]),
                "n_steps": int(len(sequence)),
                "target_mask_signature": expected_signature,
                "maximum_csv_npz_difference": (
                    maximum_array_difference
                ),
                "maximum_time_ns": float(
                    sequence["time_ns"].max()
                ),
                "sequence_csv": str(sequence_file),
            }
        )

    checksums: list[dict[str, str]] = []

    for filename in checksum_files:
        checksums.append(
            {
                "sha256": sha256(filename),
                "file": str(filename),
            }
        )

    audit = {
        "status": "PASS",
        "npz": str(npz_file),
        "manifest": str(manifest_file),
        "metadata": str(metadata_file),
        "X_raw_shape": list(X_raw.shape),
        "target_values_shape": list(target_values.shape),
        "target_mask_shape": list(target_mask.shape),
        "feature_order": feature_names,
        "target_order": target_names,
        "time_start_ns": float(time_ns[0]),
        "time_end_ns": float(time_ns[-1]),
        "time_step_ps": args.expected_step_ps,
        "samples": sequence_checks,
        "checksums": checksums,
    }

    audit_json = output_dir / "prefix_package_audit.json"
    checksums_file = (
        output_dir / "prefix_package_checksums.sha256"
    )

    audit_json.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    checksums_file.write_text(
        "".join(
            f"{record['sha256']}  {record['file']}\n"
            for record in checksums
        ),
        encoding="utf-8",
    )

    print("=" * 90)
    print("AUDITORÍA DEL PAQUETE")
    print("=" * 90)
    print("Estado:             PASS")
    print(f"X_raw:              {X_raw.shape}")
    print(f"Targets:            {target_values.shape}")
    print(f"Máscaras:           {target_mask.shape}")
    print(f"Tiempo:             {time_ns[0]}–{time_ns[-1]} ns")
    print(f"Resolución:         {args.expected_step_ps} ps")
    print()

    for record in sequence_checks:
        print(
            f"{record['sample_id']}: "
            f"mask={record['target_mask_signature']} "
            f"max_diff={record['maximum_csv_npz_difference']:.3e}"
        )

    print()
    print(f"Auditoría: {audit_json}")
    print(f"Hashes:    {checksums_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
