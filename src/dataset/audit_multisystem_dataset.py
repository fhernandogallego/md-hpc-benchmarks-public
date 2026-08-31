from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


AGGREGATE_KEYS = {
    "X_raw",
    "time_ns",
    "system_ids",
    "replicas",
    "sample_ids",
    "source_sample_ids",
    "feature_names",
    "target_values",
    "target_mask",
    "target_names",
}

SOURCE_KEYS = {
    "X_raw",
    "time_ns",
    "replicas",
    "sample_ids",
    "feature_names",
    "target_values",
    "target_mask",
    "target_names",
}

MANIFEST_COLUMNS = [
    "sample_index",
    "sample_id",
    "system",
    "replica",
    "source_sample_id",
    "target_mask_signature",
    "n_valid_targets",
    "source_release",
    "source_npz",
    "source_npz_sha256",
]

FEATURE_SUMMARY_COLUMNS = [
    "system",
    "feature",
    "n_values",
    "mean",
    "std",
    "min",
    "max",
]

TARGET_SUMMARY_COLUMNS = [
    "system",
    "target",
    "n_valid",
    "n_total",
    "valid_fraction",
    "mean",
    "std",
    "min",
    "max",
]


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def decode_strings(array: np.ndarray) -> list[str]:
    values = []

    for value in array.tolist():
        if isinstance(value, bytes):
            values.append(value.decode("utf-8"))
        else:
            values.append(str(value))

    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def mask_signature(mask: np.ndarray) -> str:
    return "".join(
        "1" if bool(value) else "0"
        for value in mask
    )


def assert_array_equal(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    equal_nan: bool = False,
) -> None:
    require(
        actual.shape == expected.shape,
        (
            f"{name}: shape mismatch: "
            f"{actual.shape} != {expected.shape}"
        ),
    )

    if equal_nan:
        equal = np.array_equal(
            actual,
            expected,
            equal_nan=True,
        )
    else:
        equal = np.array_equal(
            actual,
            expected,
        )

    require(
        equal,
        f"{name}: array contents differ.",
    )


def floats_equal(
    actual: float,
    expected: float,
    tolerance: float,
) -> bool:
    if np.isnan(actual) and np.isnan(expected):
        return True

    if not (
        np.isfinite(actual)
        and np.isfinite(expected)
    ):
        return actual == expected

    return abs(actual - expected) <= tolerance


def parse_csv_float(value: str) -> float:
    value = value.strip()

    if value == "":
        return float("nan")

    return float(value)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    return fieldnames, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strong provenance and reconstruction audit "
            "for a raw multi-system prefix dataset."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--report",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=1e-12,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    tolerance = args.numeric_tolerance

    require(
        dataset_dir.is_dir(),
        f"Dataset directory not found: {dataset_dir}",
    )

    paths = {
        "metadata": (
            dataset_dir
            / "multisystem_dataset_metadata.json"
        ),
        "npz": (
            dataset_dir
            / "multisystem_dataset_raw.npz"
        ),
        "manifest": (
            dataset_dir
            / "multisystem_sample_manifest.csv"
        ),
        "feature_summary": (
            dataset_dir
            / "multisystem_feature_summary.csv"
        ),
        "target_summary": (
            dataset_dir
            / "multisystem_target_summary.csv"
        ),
    }

    for name, path in paths.items():
        require(
            path.is_file(),
            f"Missing {name}: {path}",
        )

    metadata = json.loads(
        paths["metadata"].read_text(
            encoding="utf-8"
        )
    )

    print("===== LOAD AGGREGATE =====")

    with np.load(
        paths["npz"],
        allow_pickle=False,
    ) as archive:
        require(
            set(archive.files) == AGGREGATE_KEYS,
            (
                "Aggregate NPZ keys differ. "
                f"Found={sorted(archive.files)}"
            ),
        )

        aggregate = {
            key: archive[key].copy()
            for key in archive.files
        }

    X_raw = aggregate["X_raw"]
    time_ns = aggregate["time_ns"]
    system_ids = decode_strings(
        aggregate["system_ids"]
    )
    replicas = aggregate["replicas"]
    sample_ids = decode_strings(
        aggregate["sample_ids"]
    )
    source_sample_ids = decode_strings(
        aggregate["source_sample_ids"]
    )
    feature_names = decode_strings(
        aggregate["feature_names"]
    )
    target_values = aggregate["target_values"]
    target_mask = aggregate["target_mask"]
    target_names = decode_strings(
        aggregate["target_names"]
    )

    expected_dtypes = {
        "X_raw": np.dtype("float64"),
        "time_ns": np.dtype("float64"),
        "system_ids": np.dtype("U64"),
        "replicas": np.dtype("int64"),
        "sample_ids": np.dtype("U128"),
        "source_sample_ids": np.dtype("U128"),
        "feature_names": np.dtype("U64"),
        "target_values": np.dtype("float64"),
        "target_mask": np.dtype("bool"),
        "target_names": np.dtype("U32"),
    }

    for key, expected_dtype in expected_dtypes.items():
        require(
            aggregate[key].dtype == expected_dtype,
            (
                f"{key}: dtype "
                f"{aggregate[key].dtype} != "
                f"{expected_dtype}"
            ),
        )

    n_samples = X_raw.shape[0]

    require(
        X_raw.ndim == 3,
        f"X_raw ndim={X_raw.ndim}, expected 3.",
    )
    require(
        time_ns.shape == (X_raw.shape[1],),
        "time_ns shape does not match X_raw.",
    )
    require(
        len(feature_names) == X_raw.shape[2],
        "feature_names width mismatch.",
    )
    require(
        target_values.ndim == 2,
        "target_values must be 2-D.",
    )
    require(
        target_mask.shape == target_values.shape,
        "target_mask/value shape mismatch.",
    )
    require(
        target_values.shape[0] == n_samples,
        "Target sample count mismatch.",
    )
    require(
        len(target_names) == target_values.shape[1],
        "target_names width mismatch.",
    )

    for name, values in [
        ("system_ids", system_ids),
        ("replicas", replicas),
        ("sample_ids", sample_ids),
        ("source_sample_ids", source_sample_ids),
    ]:
        require(
            len(values) == n_samples,
            f"{name}: sample count mismatch.",
        )

    require(
        len(set(sample_ids)) == len(sample_ids),
        "Global sample IDs are not unique.",
    )
    require(
        np.all(np.isfinite(X_raw)),
        "X_raw contains non-finite values.",
    )
    require(
        np.all(
            np.isfinite(
                target_values[target_mask]
            )
        ),
        "Valid targets contain non-finite values.",
    )
    require(
        np.all(
            np.isnan(
                target_values[~target_mask]
            )
        ),
        "Invalid targets are not all NaN.",
    )

    print(
        f"X_raw={X_raw.shape}, "
        f"targets={target_values.shape}"
    )

    print()
    print("===== METADATA CONSISTENCY =====")

    require(
        metadata["dataset_type"]
        == "raw_multisystem_prefix_dataset",
        "Unexpected dataset_type.",
    )

    systems = list(metadata["systems"])

    require(
        len(systems) == len(set(systems)),
        "Metadata systems are not unique.",
    )
    require(
        metadata["n_systems"] == len(systems),
        "metadata n_systems mismatch.",
    )
    require(
        metadata["n_samples"] == n_samples,
        "metadata n_samples mismatch.",
    )
    require(
        metadata["feature_order"]
        == feature_names,
        "metadata feature_order mismatch.",
    )
    require(
        metadata["target_order"]
        == target_names,
        "metadata target_order mismatch.",
    )
    require(
        metadata["n_valid_targets"]
        == int(target_mask.sum()),
        "metadata n_valid_targets mismatch.",
    )
    require(
        metadata["n_total_targets"]
        == int(target_mask.size),
        "metadata n_total_targets mismatch.",
    )

    expected_shapes = {
        "X_raw": list(X_raw.shape),
        "time_ns": list(time_ns.shape),
        "system_ids": [n_samples],
        "replicas": list(replicas.shape),
        "sample_ids": [n_samples],
        "target_values": list(
            target_values.shape
        ),
        "target_mask": list(
            target_mask.shape
        ),
    }

    require(
        metadata["array_shapes"]
        == expected_shapes,
        "metadata array_shapes mismatch.",
    )

    require(
        float(metadata["time_start_ns"])
        == float(time_ns[0]),
        "metadata time_start_ns mismatch.",
    )
    require(
        float(metadata["time_end_ns"])
        == float(time_ns[-1]),
        "metadata time_end_ns mismatch.",
    )

    if len(time_ns) > 1:
        deltas = np.diff(time_ns)

        require(
            np.allclose(
                deltas,
                float(metadata["time_step_ns"]),
                rtol=0.0,
                atol=tolerance,
            ),
            (
                "Aggregate time grid is not uniform "
                f"within atol={tolerance}."
            ),
        )
        require(
            abs(
                float(np.mean(deltas))
                - float(metadata["time_step_ns"])
            )
            <= tolerance,
            "metadata time_step_ns mismatch.",
        )

    for key in [
        "normalized",
        "imputed",
        "train_validation_test_split",
        "sample_order_randomized",
    ]:
        require(
            metadata[key] is False,
            f"{key} must be false.",
        )

    leakage = metadata["leakage_policy"]

    require(
        float(
            leakage["maximum_input_time_ns"]
        )
        == float(time_ns[-1]),
        (
            "leakage maximum_input_time_ns "
            "does not match input endpoint."
        ),
    )

    for key in [
        "post_prefix_features_allowed",
        "oracle_features_allowed",
        "target_values_in_input",
        "scaler_fitted",
    ]:
        require(
            leakage[key] is False,
            f"leakage_policy.{key} must be false.",
        )

    print("Metadata: PASS")

    print()
    print("===== SOURCE HASH + RECONSTRUCTION =====")

    sources = metadata["sources"]

    require(
        len(sources) == len(systems),
        "metadata source count mismatch.",
    )
    require(
        [source["system"] for source in sources]
        == systems,
        "Source order differs from systems order.",
    )

    expected_X = []
    expected_targets = []
    expected_masks = []
    expected_replicas = []

    expected_system_ids: list[str] = []
    expected_sample_ids: list[str] = []
    expected_source_sample_ids: list[str] = []
    expected_manifest: list[dict[str, str]] = []

    source_audit = []
    global_index = 0

    for source in sources:
        system = str(source["system"])
        release = Path(source["release"])
        npz_path = Path(source["npz"])

        require(
            release.is_dir(),
            f"{system}: release missing: {release}",
        )
        require(
            npz_path.is_file(),
            f"{system}: source NPZ missing: {npz_path}",
        )

        actual_sha = sha256_file(npz_path)
        recorded_sha = str(
            source["npz_sha256"]
        )

        require(
            actual_sha == recorded_sha,
            (
                f"{system}: source SHA256 mismatch: "
                f"{actual_sha} != {recorded_sha}"
            ),
        )

        with np.load(
            npz_path,
            allow_pickle=False,
        ) as archive:
            missing = SOURCE_KEYS.difference(
                archive.files
            )

            require(
                not missing,
                (
                    f"{system}: missing source keys "
                    f"{sorted(missing)}"
                ),
            )

            source_X = archive["X_raw"].copy()
            source_time = archive["time_ns"].copy()
            source_replicas = (
                archive["replicas"]
                .astype(np.int64)
                .copy()
            )
            source_ids = decode_strings(
                archive["sample_ids"]
            )
            source_features = decode_strings(
                archive["feature_names"]
            )
            source_targets = (
                archive["target_values"].copy()
            )
            source_mask = (
                archive["target_mask"]
                .astype(bool)
                .copy()
            )
            source_target_names = decode_strings(
                archive["target_names"]
            )

        require(
            source_X.dtype == np.float64,
            f"{system}: X_raw is not float64.",
        )
        require(
            source_targets.dtype == np.float64,
            (
                f"{system}: target_values "
                "is not float64."
            ),
        )
        require(
            source_X.shape[1:]
            == X_raw.shape[1:],
            (
                f"{system}: sequence shape "
                "differs from aggregate."
            ),
        )
        require(
            source_targets.shape[1]
            == target_values.shape[1],
            (
                f"{system}: target width "
                "differs."
            ),
        )

        assert_array_equal(
            f"{system}.time_ns",
            source_time,
            time_ns,
        )

        require(
            source_features == feature_names,
            f"{system}: feature order differs.",
        )
        require(
            source_target_names == target_names,
            f"{system}: target order differs.",
        )
        require(
            source_targets.shape
            == source_mask.shape,
            f"{system}: target/mask shape mismatch.",
        )
        require(
            source_X.shape[0]
            == source_targets.shape[0],
            f"{system}: sample count mismatch.",
        )
        require(
            len(source_ids)
            == source_X.shape[0],
            f"{system}: sample_ids mismatch.",
        )
        require(
            len(source_replicas)
            == source_X.shape[0],
            f"{system}: replicas mismatch.",
        )
        require(
            len(set(source_ids))
            == len(source_ids),
            f"{system}: duplicate source IDs.",
        )
        require(
            np.all(np.isfinite(source_X)),
            f"{system}: source X has non-finite values.",
        )
        require(
            np.all(
                np.isfinite(
                    source_targets[source_mask]
                )
            ),
            (
                f"{system}: valid source targets "
                "contain non-finite values."
            ),
        )
        require(
            np.all(
                np.isnan(
                    source_targets[~source_mask]
                )
            ),
            (
                f"{system}: invalid source targets "
                "are not NaN."
            ),
        )

        require(
            int(source["n_samples"])
            == source_X.shape[0],
            f"{system}: metadata n_samples mismatch.",
        )
        require(
            int(source["n_valid_targets"])
            == int(source_mask.sum()),
            (
                f"{system}: metadata "
                "n_valid_targets mismatch."
            ),
        )

        expected_X.append(source_X)
        expected_targets.append(source_targets)
        expected_masks.append(source_mask)
        expected_replicas.append(
            source_replicas
        )

        for local_index, source_id in enumerate(
            source_ids
        ):
            replica = int(
                source_replicas[local_index]
            )
            global_sample_id = (
                f"{system}_R{replica}"
            )

            expected_system_ids.append(system)
            expected_sample_ids.append(
                global_sample_id
            )
            expected_source_sample_ids.append(
                source_id
            )

            expected_manifest.append(
                {
                    "sample_index": str(
                        global_index
                    ),
                    "sample_id": global_sample_id,
                    "system": system,
                    "replica": str(replica),
                    "source_sample_id": source_id,
                    "target_mask_signature": (
                        mask_signature(
                            source_mask[
                                local_index
                            ]
                        )
                    ),
                    "n_valid_targets": str(
                        int(
                            source_mask[
                                local_index
                            ].sum()
                        )
                    ),
                    "source_release": str(
                        source["release"]
                    ),
                    "source_npz": str(
                        source["npz"]
                    ),
                    "source_npz_sha256": (
                        recorded_sha
                    ),
                }
            )

            global_index += 1

        source_audit.append(
            {
                "system": system,
                "npz": str(npz_path),
                "sha256": actual_sha,
                "n_samples": int(
                    source_X.shape[0]
                ),
                "n_valid_targets": int(
                    source_mask.sum()
                ),
            }
        )

        print(
            f"{system}: "
            f"SHA256 PASS, "
            f"samples={source_X.shape[0]}, "
            f"valid={int(source_mask.sum())}"
        )

    reconstructed_X = np.concatenate(
        expected_X,
        axis=0,
    )
    reconstructed_targets = np.concatenate(
        expected_targets,
        axis=0,
    )
    reconstructed_masks = np.concatenate(
        expected_masks,
        axis=0,
    )
    reconstructed_replicas = np.concatenate(
        expected_replicas,
        axis=0,
    )

    assert_array_equal(
        "aggregate.X_raw",
        X_raw,
        reconstructed_X,
    )
    assert_array_equal(
        "aggregate.target_values",
        target_values,
        reconstructed_targets,
        equal_nan=True,
    )
    assert_array_equal(
        "aggregate.target_mask",
        target_mask,
        reconstructed_masks,
    )
    assert_array_equal(
        "aggregate.replicas",
        replicas,
        reconstructed_replicas,
    )

    require(
        system_ids == expected_system_ids,
        "aggregate system_ids differ.",
    )
    require(
        sample_ids == expected_sample_ids,
        "aggregate sample_ids differ.",
    )
    require(
        source_sample_ids
        == expected_source_sample_ids,
        "aggregate source_sample_ids differ.",
    )

    first_occurrence_systems = list(
        dict.fromkeys(system_ids)
    )

    require(
        first_occurrence_systems == systems,
        "Aggregate system ordering differs.",
    )

    print("Exact source reconstruction: PASS")

    print()
    print("===== SAMPLE MANIFEST =====")

    manifest_columns, manifest_rows = read_csv(
        paths["manifest"]
    )

    require(
        manifest_columns == MANIFEST_COLUMNS,
        "Manifest columns differ.",
    )
    require(
        len(manifest_rows)
        == len(expected_manifest),
        "Manifest row count differs.",
    )

    for index, (actual, expected) in enumerate(
        zip(
            manifest_rows,
            expected_manifest,
            strict=True,
        )
    ):
        require(
            actual == expected,
            (
                f"Manifest row {index} differs.\n"
                f"Actual:   {actual}\n"
                f"Expected: {expected}"
            ),
        )

    print("Sample manifest: PASS")

    print()
    print("===== FEATURE SUMMARY =====")

    feature_columns, feature_rows = read_csv(
        paths["feature_summary"]
    )

    require(
        feature_columns
        == FEATURE_SUMMARY_COLUMNS,
        "Feature summary columns differ.",
    )

    expected_feature_rows = []

    system_ids_array = np.asarray(
        system_ids
    )

    for system in systems + ["__all__"]:
        if system == "__all__":
            values = X_raw
        else:
            values = X_raw[
                system_ids_array == system
            ]

        for feature_index, feature_name in enumerate(
            feature_names
        ):
            feature_values = values[
                :,
                :,
                feature_index,
            ]

            expected_feature_rows.append(
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

    require(
        len(feature_rows)
        == len(expected_feature_rows),
        "Feature summary row count differs.",
    )

    for index, (actual, expected) in enumerate(
        zip(
            feature_rows,
            expected_feature_rows,
            strict=True,
        )
    ):
        require(
            actual["system"]
            == expected["system"],
            (
                f"Feature row {index}: "
                "system differs."
            ),
        )
        require(
            actual["feature"]
            == expected["feature"],
            (
                f"Feature row {index}: "
                "feature differs."
            ),
        )
        require(
            int(actual["n_values"])
            == expected["n_values"],
            (
                f"Feature row {index}: "
                "n_values differs."
            ),
        )

        for field in [
            "mean",
            "std",
            "min",
            "max",
        ]:
            require(
                floats_equal(
                    parse_csv_float(actual[field]),
                    expected[field],
                    tolerance,
                ),
                (
                    f"Feature row {index} "
                    f"{field} differs: "
                    f"{actual[field]} != "
                    f"{expected[field]}"
                ),
            )

    print("Feature summary: PASS")

    print()
    print("===== TARGET SUMMARY =====")

    target_columns, target_rows = read_csv(
        paths["target_summary"]
    )

    require(
        target_columns
        == TARGET_SUMMARY_COLUMNS,
        "Target summary columns differ.",
    )

    expected_target_rows = []

    for system in systems + ["__all__"]:
        if system == "__all__":
            sample_selector = np.ones(
                n_samples,
                dtype=bool,
            )
        else:
            sample_selector = (
                system_ids_array == system
            )

        n_total = int(
            sample_selector.sum()
        )

        for target_index, target_name in enumerate(
            target_names
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
                "n_total": n_total,
                "valid_fraction": float(
                    values.size / n_total
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

            expected_target_rows.append(
                record
            )

    require(
        len(target_rows)
        == len(expected_target_rows),
        "Target summary row count differs.",
    )

    for index, (actual, expected) in enumerate(
        zip(
            target_rows,
            expected_target_rows,
            strict=True,
        )
    ):
        require(
            actual["system"]
            == expected["system"],
            (
                f"Target row {index}: "
                "system differs."
            ),
        )
        require(
            actual["target"]
            == expected["target"],
            (
                f"Target row {index}: "
                "target differs."
            ),
        )

        for field in [
            "n_valid",
            "n_total",
        ]:
            require(
                int(actual[field])
                == expected[field],
                (
                    f"Target row {index} "
                    f"{field} differs."
                ),
            )

        for field in [
            "valid_fraction",
            "mean",
            "std",
            "min",
            "max",
        ]:
            require(
                floats_equal(
                    parse_csv_float(actual[field]),
                    float(expected[field]),
                    tolerance,
                ),
                (
                    f"Target row {index} "
                    f"{field} differs: "
                    f"{actual[field]} != "
                    f"{expected[field]}"
                ),
            )

    print("Target summary: PASS")

    print()
    print("===== ARTIFACT HASHES =====")

    artifact_hashes = {}

    for name, path in paths.items():
        digest = sha256_file(path)
        artifact_hashes[name] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }

        print(
            f"{name}: {digest}"
        )

    report = {
        "audited_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "dataset_dir": str(dataset_dir),
        "numeric_tolerance": tolerance,
        "n_systems": len(systems),
        "systems": systems,
        "n_samples": n_samples,
        "X_raw_shape": list(X_raw.shape),
        "target_values_shape": list(
            target_values.shape
        ),
        "n_valid_targets": int(
            target_mask.sum()
        ),
        "n_total_targets": int(
            target_mask.size
        ),
        "feature_order": feature_names,
        "target_order": target_names,
        "time_start_ns": float(
            time_ns[0]
        ),
        "time_end_ns": float(
            time_ns[-1]
        ),
        "checks": {
            "aggregate_schema": True,
            "metadata_consistency": True,
            "source_sha256": True,
            "exact_source_reconstruction": True,
            "sample_manifest": True,
            "feature_summary": True,
            "target_summary": True,
            "raw_dataset_policy": True,
            "leakage_policy": True,
        },
        "sources": source_audit,
        "artifacts": artifact_hashes,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("========================================")
    print("MULTISYSTEM DATASET AUDIT: PASS")
    print("========================================")
    print(f"Systems:       {len(systems)}")
    print(f"Samples:       {n_samples}")
    print(f"X_raw:         {X_raw.shape}")
    print(
        "Valid targets: "
        f"{int(target_mask.sum())}/"
        f"{int(target_mask.size)}"
    )
    print(f"Report:        {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("========================================")
        print("MULTISYSTEM DATASET AUDIT: FAIL")
        print("========================================")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise
