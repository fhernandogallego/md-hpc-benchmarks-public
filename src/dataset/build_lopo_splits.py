from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic leave-one-protein-out "
            "splits from a frozen multisystem dataset."
        )
    )

    parser.add_argument(
        "--dataset-npz",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
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


def mask_signature(mask: np.ndarray) -> str:
    return "".join(
        "1" if bool(value) else "0"
        for value in mask
    )


def main() -> None:
    args = parse_args()

    dataset_npz = args.dataset_npz.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not dataset_npz.is_file():
        raise FileNotFoundError(dataset_npz)

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: {output_dir}"
        )

    with np.load(
        dataset_npz,
        allow_pickle=False,
    ) as archive:
        required = {
            "X_raw",
            "system_ids",
            "replicas",
            "sample_ids",
            "target_mask",
            "target_names",
        }

        missing = required.difference(
            archive.files
        )

        if missing:
            raise RuntimeError(
                f"Missing arrays: {sorted(missing)}"
            )

        X_raw = archive["X_raw"].copy()
        system_ids = np.asarray(
            decode_strings(
                archive["system_ids"]
            )
        )
        replicas = (
            archive["replicas"]
            .astype(np.int64)
            .copy()
        )
        sample_ids = decode_strings(
            archive["sample_ids"]
        )
        target_mask = (
            archive["target_mask"]
            .astype(bool)
            .copy()
        )
        target_names = decode_strings(
            archive["target_names"]
        )

    n_samples = X_raw.shape[0]

    if len(system_ids) != n_samples:
        raise RuntimeError(
            "system_ids sample count mismatch."
        )

    if len(sample_ids) != n_samples:
        raise RuntimeError(
            "sample_ids sample count mismatch."
        )

    if len(replicas) != n_samples:
        raise RuntimeError(
            "replicas sample count mismatch."
        )

    if target_mask.shape[0] != n_samples:
        raise RuntimeError(
            "target_mask sample count mismatch."
        )

    if target_mask.shape[1] != len(target_names):
        raise RuntimeError(
            "target_mask target width mismatch."
        )

    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(
            "Duplicate sample IDs."
        )

    systems = list(
        dict.fromkeys(system_ids.tolist())
    )

    if len(systems) < 2:
        raise RuntimeError(
            "LOPO requires at least two systems."
        )

    fold_records = []
    sample_records = []
    test_appearance = np.zeros(
        n_samples,
        dtype=np.int64,
    )

    for fold_index, test_system in enumerate(
        systems,
        start=1,
    ):
        test_selector = (
            system_ids == test_system
        )
        train_selector = ~test_selector

        train_indices = np.where(
            train_selector
        )[0]
        test_indices = np.where(
            test_selector
        )[0]

        if not len(test_indices):
            raise RuntimeError(
                f"{test_system}: empty test fold."
            )

        train_systems = list(
            dict.fromkeys(
                system_ids[
                    train_indices
                ].tolist()
            )
        )

        test_systems = list(
            dict.fromkeys(
                system_ids[
                    test_indices
                ].tolist()
            )
        )

        overlap = (
            set(train_systems)
            & set(test_systems)
        )

        if overlap:
            raise RuntimeError(
                f"Fold {fold_index}: "
                f"group leakage {sorted(overlap)}"
            )

        train_counts = (
            target_mask[train_indices]
            .sum(axis=0)
            .astype(int)
        )

        test_counts = (
            target_mask[test_indices]
            .sum(axis=0)
            .astype(int)
        )

        if np.any(train_counts == 0):
            missing_targets = [
                target
                for target, count
                in zip(
                    target_names,
                    train_counts,
                )
                if count == 0
            ]

            raise RuntimeError(
                f"Fold {fold_index}: no training "
                f"examples for {missing_targets}"
            )

        test_appearance[
            test_indices
        ] += 1

        fold_id = f"fold_{fold_index:02d}"

        fold_records.append(
            {
                "fold_id": fold_id,
                "fold_index": fold_index,
                "test_system": test_system,
                "train_systems": train_systems,
                "test_systems": test_systems,
                "train_indices": (
                    train_indices.tolist()
                ),
                "test_indices": (
                    test_indices.tolist()
                ),
                "train_sample_ids": [
                    sample_ids[i]
                    for i in train_indices
                ],
                "test_sample_ids": [
                    sample_ids[i]
                    for i in test_indices
                ],
                "train_target_counts": {
                    target: int(count)
                    for target, count
                    in zip(
                        target_names,
                        train_counts,
                    )
                },
                "test_target_counts": {
                    target: int(count)
                    for target, count
                    in zip(
                        target_names,
                        test_counts,
                    )
                },
                "test_unobserved_targets": [
                    target
                    for target, count
                    in zip(
                        target_names,
                        test_counts,
                    )
                    if count == 0
                ],
            }
        )

        for sample_index in range(
            n_samples
        ):
            role = (
                "test"
                if test_selector[sample_index]
                else "train"
            )

            sample_records.append(
                {
                    "fold_id": fold_id,
                    "fold_index": fold_index,
                    "test_system": test_system,
                    "sample_index": (
                        sample_index
                    ),
                    "sample_id": (
                        sample_ids[
                            sample_index
                        ]
                    ),
                    "system": (
                        str(
                            system_ids[
                                sample_index
                            ]
                        )
                    ),
                    "replica": int(
                        replicas[
                            sample_index
                        ]
                    ),
                    "role": role,
                    "target_mask_signature": (
                        mask_signature(
                            target_mask[
                                sample_index
                            ]
                        )
                    ),
                    "n_valid_targets": int(
                        target_mask[
                            sample_index
                        ].sum()
                    ),
                }
            )

    if not np.all(
        test_appearance == 1
    ):
        raise RuntimeError(
            "Every sample must appear exactly "
            "once as outer test."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    folds_json = (
        output_dir
        / "lopo_folds.json"
    )

    folds_json.write_text(
        json.dumps(
            {
                "strategy": (
                    "leave_one_protein_out"
                ),
                "group_key": "system_id",
                "n_folds": len(
                    fold_records
                ),
                "folds": fold_records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = (
        output_dir
        / "lopo_sample_manifest.csv"
    )

    manifest_columns = [
        "fold_id",
        "fold_index",
        "test_system",
        "sample_index",
        "sample_id",
        "system",
        "replica",
        "role",
        "target_mask_signature",
        "n_valid_targets",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=manifest_columns,
        )
        writer.writeheader()
        writer.writerows(
            sample_records
        )

    metadata = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "split_type": (
            "deterministic_outer_lopo"
        ),
        "dataset_npz": str(
            dataset_npz
        ),
        "dataset_npz_sha256": (
            sha256_file(dataset_npz)
        ),
        "group_key": "system_id",
        "shuffle": False,
        "random_seed": None,
        "n_systems": len(systems),
        "systems": systems,
        "n_samples": n_samples,
        "n_folds": len(
            fold_records
        ),
        "samples_per_outer_test": {
            fold["fold_id"]: len(
                fold["test_indices"]
            )
            for fold in fold_records
        },
        "target_names": target_names,
        "global_target_counts": {
            target: int(count)
            for target, count
            in zip(
                target_names,
                target_mask.sum(
                    axis=0
                ),
            )
        },
        "invariants": {
            "protein_group_leakage": False,
            "each_sample_test_exactly_once": True,
            "training_has_each_target": True,
            "normalization_applied": False,
            "imputation_applied": False,
            "target_mask_modified": False,
        },
    }

    metadata_path = (
        output_dir
        / "split_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    checksums_path = (
        output_dir
        / "SHA256SUMS.txt"
    )

    checksum_files = [
        folds_json,
        manifest_path,
        metadata_path,
    ]

    with checksums_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            checksum_files,
            key=lambda p: p.name,
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )

    print(
        "========================================"
    )
    print("LOPO SPLIT BUILD: PASS")
    print(
        "========================================"
    )
    print(
        f"Systems: {len(systems)}"
    )
    print(
        f"Samples: {n_samples}"
    )
    print(
        f"Folds:   {len(fold_records)}"
    )
    print()

    for fold in fold_records:
        counts = " ".join(
            f"{target}={count}"
            for target, count
            in fold[
                "test_target_counts"
            ].items()
        )

        missing = (
            ",".join(
                fold[
                    "test_unobserved_targets"
                ]
            )
            or "none"
        )

        print(
            f'{fold["fold_id"]} '
            f'TEST={fold["test_system"]} '
            f'n_test='
            f'{len(fold["test_indices"])} '
            f'{counts} '
            f'unobserved={missing}'
        )

    print()
    print(
        "Each sample test exactly once: PASS"
    )
    print(
        "Protein group leakage: PASS"
    )
    print(
        "Training target coverage: PASS"
    )
    print(
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
