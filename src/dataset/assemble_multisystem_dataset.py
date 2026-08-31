from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_KEYS = {
    "X_raw",
    "time_ns",
    "replicas",
    "sample_ids",
    "feature_names",
    "target_values",
    "target_mask",
    "target_names",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine validated single-system prefix packages "
            "into one raw multi-system dataset."
        )
    )

    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="SYSTEM=RELEASE",
        help=(
            "Validated source release. Repeat once per system. "
            "Example: --source 1k5n_A=/path/to/release"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for the combined dataset.",
    )

    return parser.parse_args()


def decode_strings(array: np.ndarray) -> list[str]:
    result = []

    for value in array.tolist():
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))

    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def locate_npz(release: Path) -> Path:
    matches = sorted(
        release.rglob("prefix_sequences_raw.npz")
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"{release}: expected exactly one "
            f"prefix_sequences_raw.npz, found "
            f"{len(matches)}."
        )

    return matches[0]


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            f"Invalid --source value: {value!r}. "
            "Expected SYSTEM=RELEASE."
        )

    system, release_text = value.split("=", 1)

    system = system.strip()
    release_text = release_text.strip()

    if not system:
        raise ValueError(
            f"Empty system in --source {value!r}."
        )

    if not release_text:
        raise ValueError(
            f"Empty release path in --source {value!r}."
        )

    return system, Path(release_text).expanduser()


def mask_signature(mask: np.ndarray) -> str:
    return "".join(
        "1" if bool(value) else "0"
        for value in mask
    )


def load_source(
    system: str,
    release: Path,
) -> dict:
    if not release.is_dir():
        raise FileNotFoundError(release)

    npz_path = locate_npz(release)

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as archive:
        missing = REQUIRED_KEYS.difference(
            archive.files
        )

        if missing:
            raise RuntimeError(
                f"{system}: missing arrays "
                f"{sorted(missing)}."
            )

        data = {
            "system": system,
            "release": release.resolve(),
            "npz_path": npz_path.resolve(),
            "npz_sha256": sha256_file(npz_path),
            "X_raw": archive["X_raw"].copy(),
            "time_ns": archive["time_ns"].copy(),
            "replicas": archive["replicas"].copy(),
            "sample_ids": decode_strings(
                archive["sample_ids"]
            ),
            "feature_names": decode_strings(
                archive["feature_names"]
            ),
            "target_values": (
                archive["target_values"].copy()
            ),
            "target_mask": (
                archive["target_mask"]
                .astype(bool)
                .copy()
            ),
            "target_names": decode_strings(
                archive["target_names"]
            ),
        }

    X_raw = data["X_raw"]
    time_ns = data["time_ns"]
    replicas = data["replicas"]
    target_values = data["target_values"]
    target_mask = data["target_mask"]
    sample_ids = data["sample_ids"]

    if X_raw.ndim != 3:
        raise RuntimeError(
            f"{system}: X_raw shape {X_raw.shape}."
        )

    n_samples = X_raw.shape[0]

    if target_values.ndim != 2:
        raise RuntimeError(
            f"{system}: target_values shape "
            f"{target_values.shape}."
        )

    if target_mask.shape != target_values.shape:
        raise RuntimeError(
            f"{system}: target mask/value shapes differ."
        )

    if target_values.shape[0] != n_samples:
        raise RuntimeError(
            f"{system}: target sample count differs."
        )

    if len(replicas) != n_samples:
        raise RuntimeError(
            f"{system}: replica count differs."
        )

    if len(sample_ids) != n_samples:
        raise RuntimeError(
            f"{system}: sample-id count differs."
        )

    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(
            f"{system}: duplicate sample IDs."
        )

    if time_ns.shape != (X_raw.shape[1],):
        raise RuntimeError(
            f"{system}: time axis shape mismatch."
        )

    if X_raw.dtype != np.float64:
        raise RuntimeError(
            f"{system}: X_raw dtype is {X_raw.dtype}, "
            "expected float64."
        )

    if target_values.dtype != np.float64:
        raise RuntimeError(
            f"{system}: target_values dtype is "
            f"{target_values.dtype}, expected float64."
        )

    if not np.all(np.isfinite(X_raw)):
        raise RuntimeError(
            f"{system}: X_raw contains non-finite values."
        )

    if not np.all(
        np.isfinite(target_values[target_mask])
    ):
        raise RuntimeError(
            f"{system}: valid targets contain "
            "non-finite values."
        )

    if not np.all(
        np.isnan(target_values[~target_mask])
    ):
        raise RuntimeError(
            f"{system}: invalid targets are not all NaN."
        )

    return data


def main() -> None:
    args = parse_args()

    parsed_sources = [
        parse_source(value)
        for value in args.source
    ]

    systems = [
        system
        for system, _ in parsed_sources
    ]

    if len(set(systems)) != len(systems):
        raise RuntimeError(
            "Each system may appear only once."
        )

    sources = [
        load_source(system, release)
        for system, release in parsed_sources
    ]

    reference = sources[0]

    for source in sources[1:]:
        system = source["system"]

        if (
            source["feature_names"]
            != reference["feature_names"]
        ):
            raise RuntimeError(
                f"{system}: feature order differs."
            )

        if (
            source["target_names"]
            != reference["target_names"]
        ):
            raise RuntimeError(
                f"{system}: target order differs."
            )

        if source["X_raw"].shape[1:] != (
            reference["X_raw"].shape[1:]
        ):
            raise RuntimeError(
                f"{system}: sequence shape differs."
            )

        if source["target_values"].shape[1] != (
            reference["target_values"].shape[1]
        ):
            raise RuntimeError(
                f"{system}: target width differs."
            )

        if not np.array_equal(
            source["time_ns"],
            reference["time_ns"],
        ):
            raise RuntimeError(
                f"{system}: time grid differs."
            )

    X_raw = np.concatenate(
        [source["X_raw"] for source in sources],
        axis=0,
    )

    target_values = np.concatenate(
        [
            source["target_values"]
            for source in sources
        ],
        axis=0,
    )

    target_mask = np.concatenate(
        [
            source["target_mask"]
            for source in sources
        ],
        axis=0,
    )

    replicas = np.concatenate(
        [
            np.asarray(
                source["replicas"],
                dtype=np.int64,
            )
            for source in sources
        ],
        axis=0,
    )

    system_ids: list[str] = []
    sample_ids: list[str] = []
    source_sample_ids: list[str] = []

    manifest_records: list[dict] = []

    sample_index = 0

    for source in sources:
        system = source["system"]

        for local_index, source_sample_id in enumerate(
            source["sample_ids"]
        ):
            replica = int(
                source["replicas"][local_index]
            )

            global_sample_id = (
                f"{system}_R{replica}"
            )

            system_ids.append(system)
            sample_ids.append(global_sample_id)
            source_sample_ids.append(
                source_sample_id
            )

            signature = mask_signature(
                source["target_mask"][local_index]
            )

            manifest_records.append(
                {
                    "sample_index": sample_index,
                    "sample_id": global_sample_id,
                    "system": system,
                    "replica": replica,
                    "source_sample_id": (
                        source_sample_id
                    ),
                    "target_mask_signature": (
                        signature
                    ),
                    "n_valid_targets": int(
                        source["target_mask"][
                            local_index
                        ].sum()
                    ),
                    "source_release": str(
                        source["release"]
                    ),
                    "source_npz": str(
                        source["npz_path"]
                    ),
                    "source_npz_sha256": (
                        source["npz_sha256"]
                    ),
                }
            )

            sample_index += 1

    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(
            "Global sample IDs are not unique."
        )

    output_dir = args.output_dir.expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_names = np.asarray(
        reference["feature_names"],
        dtype="U64",
    )

    target_names = np.asarray(
        reference["target_names"],
        dtype="U32",
    )

    time_ns = np.asarray(
        reference["time_ns"],
        dtype=np.float64,
    )

    npz_path = (
        output_dir
        / "multisystem_dataset_raw.npz"
    )

    np.savez_compressed(
        npz_path,
        X_raw=X_raw,
        time_ns=time_ns,
        system_ids=np.asarray(
            system_ids,
            dtype="U64",
        ),
        replicas=replicas,
        sample_ids=np.asarray(
            sample_ids,
            dtype="U128",
        ),
        source_sample_ids=np.asarray(
            source_sample_ids,
            dtype="U128",
        ),
        feature_names=feature_names,
        target_values=target_values,
        target_mask=target_mask,
        target_names=target_names,
    )

    manifest = pd.DataFrame(
        manifest_records
    )

    manifest.to_csv(
        output_dir
        / "multisystem_sample_manifest.csv",
        index=False,
    )

    feature_records: list[dict] = []

    for system in systems + ["__all__"]:
        if system == "__all__":
            values = X_raw
        else:
            selector = np.asarray(
                system_ids
            ) == system

            values = X_raw[selector]

        for feature_index, feature_name in enumerate(
            reference["feature_names"]
        ):
            feature_values = values[
                :,
                :,
                feature_index,
            ]

            feature_records.append(
                {
                    "system": system,
                    "feature": feature_name,
                    "n_values": int(
                        feature_values.size
                    ),
                    "mean": float(
                        np.mean(feature_values)
                    ),
                    "std": float(
                        np.std(feature_values)
                    ),
                    "min": float(
                        np.min(feature_values)
                    ),
                    "max": float(
                        np.max(feature_values)
                    ),
                }
            )

    pd.DataFrame(
        feature_records
    ).to_csv(
        output_dir
        / "multisystem_feature_summary.csv",
        index=False,
    )

    target_records: list[dict] = []

    system_ids_array = np.asarray(
        system_ids
    )

    for system in systems + ["__all__"]:
        if system == "__all__":
            sample_selector = np.ones(
                len(system_ids),
                dtype=bool,
            )
        else:
            sample_selector = (
                system_ids_array == system
            )

        for target_index, target_name in enumerate(
            reference["target_names"]
        ):
            valid_selector = (
                sample_selector
                & target_mask[:, target_index]
            )

            values = target_values[
                valid_selector,
                target_index,
            ]

            record = {
                "system": system,
                "target": target_name,
                "n_valid": int(values.size),
                "n_total": int(
                    sample_selector.sum()
                ),
                "valid_fraction": float(
                    values.size
                    / sample_selector.sum()
                ),
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            }

            if values.size:
                record.update(
                    {
                        "mean": float(
                            np.mean(values)
                        ),
                        "std": float(
                            np.std(values)
                        ),
                        "min": float(
                            np.min(values)
                        ),
                        "max": float(
                            np.max(values)
                        ),
                    }
                )

            target_records.append(record)

    pd.DataFrame(
        target_records
    ).to_csv(
        output_dir
        / "multisystem_target_summary.csv",
        index=False,
    )

    metadata = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_type": (
            "raw_multisystem_prefix_dataset"
        ),
        "n_systems": len(systems),
        "systems": systems,
        "n_samples": int(X_raw.shape[0]),
        "array_shapes": {
            "X_raw": list(X_raw.shape),
            "time_ns": list(time_ns.shape),
            "system_ids": [
                len(system_ids)
            ],
            "replicas": list(replicas.shape),
            "sample_ids": [
                len(sample_ids)
            ],
            "target_values": list(
                target_values.shape
            ),
            "target_mask": list(
                target_mask.shape
            ),
        },
        "feature_order": (
            reference["feature_names"]
        ),
        "target_order": (
            reference["target_names"]
        ),
        "time_start_ns": float(
            time_ns[0]
        ),
        "time_end_ns": float(
            time_ns[-1]
        ),
        "time_step_ns": float(
            np.diff(time_ns)[0]
        ),
        "n_valid_targets": int(
            target_mask.sum()
        ),
        "n_total_targets": int(
            target_mask.size
        ),
        "normalized": False,
        "imputed": False,
        "train_validation_test_split": False,
        "sample_order_randomized": False,
        "leakage_policy": {
            "maximum_input_time_ns": 20.0,
            "post_prefix_features_allowed": False,
            "oracle_features_allowed": False,
            "target_values_in_input": False,
            "scaler_fitted": False,
        },
        "sources": [
            {
                "system": source["system"],
                "release": str(
                    source["release"]
                ),
                "npz": str(
                    source["npz_path"]
                ),
                "npz_sha256": (
                    source["npz_sha256"]
                ),
                "n_samples": int(
                    source["X_raw"].shape[0]
                ),
                "n_valid_targets": int(
                    source["target_mask"].sum()
                ),
            }
            for source in sources
        ],
    }

    with (
        output_dir
        / "multisystem_dataset_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")

    print("=" * 80)
    print("MULTI-SYSTEM DATASET ASSEMBLY")
    print("=" * 80)

    for source in sources:
        print(
            f"{source['system']}: "
            f"{source['X_raw'].shape[0]} samples, "
            f"{int(source['target_mask'].sum())}/"
            f"{source['target_mask'].size} "
            "valid targets"
        )

    print()
    print(f"Systems:            {systems}")
    print(f"X_raw:              {X_raw.shape}")
    print(
        f"target_values:      "
        f"{target_values.shape}"
    )
    print(
        f"target_mask:        "
        f"{target_mask.shape}"
    )
    print(
        f"time_ns:            "
        f"{time_ns.shape}"
    )
    print(
        f"Valid targets:      "
        f"{int(target_mask.sum())}/"
        f"{target_mask.size}"
    )
    print(f"Sample IDs:         {sample_ids}")
    print()
    print(f"Output: {output_dir}")
    print()
    print("MULTI-SYSTEM ASSEMBLY: PASS")


if __name__ == "__main__":
    main()
