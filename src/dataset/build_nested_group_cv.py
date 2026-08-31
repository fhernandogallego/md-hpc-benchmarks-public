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
            "Build deterministic nested protein-group CV "
            "inside frozen outer LOPO folds."
        )
    )

    parser.add_argument(
        "--dataset-npz",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--outer-folds",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def decode_strings(
    array: np.ndarray,
) -> list[str]:
    result = []

    for value in array.tolist():
        if isinstance(value, bytes):
            result.append(
                value.decode("utf-8")
            )
        else:
            result.append(str(value))

    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def mask_signature(
    mask: np.ndarray,
) -> str:
    return "".join(
        "1" if bool(value) else "0"
        for value in mask
    )


def main() -> None:
    args = parse_args()

    dataset_npz = (
        args.dataset_npz
        .expanduser()
        .resolve()
    )

    outer_folds_path = (
        args.outer_folds
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not dataset_npz.is_file():
        raise FileNotFoundError(
            dataset_npz
        )

    if not outer_folds_path.is_file():
        raise FileNotFoundError(
            outer_folds_path
        )

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: "
            f"{output_dir}"
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
                f"Missing arrays: "
                f"{sorted(missing)}"
            )

        X_raw = (
            archive["X_raw"]
            .copy()
        )

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
            "system_ids sample count "
            "mismatch."
        )

    if len(sample_ids) != n_samples:
        raise RuntimeError(
            "sample_ids sample count "
            "mismatch."
        )

    if len(replicas) != n_samples:
        raise RuntimeError(
            "replicas sample count "
            "mismatch."
        )

    if target_mask.shape[0] != n_samples:
        raise RuntimeError(
            "target_mask sample count "
            "mismatch."
        )

    if (
        target_mask.shape[1]
        != len(target_names)
    ):
        raise RuntimeError(
            "target width mismatch."
        )

    if len(set(sample_ids)) != n_samples:
        raise RuntimeError(
            "Duplicate sample IDs."
        )

    systems = list(
        dict.fromkeys(
            system_ids.tolist()
        )
    )

    outer_data = json.loads(
        outer_folds_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        outer_data.get("strategy")
        != "leave_one_protein_out"
    ):
        raise RuntimeError(
            "Outer strategy is not LOPO."
        )

    outer_folds = outer_data[
        "folds"
    ]

    if len(outer_folds) != len(systems):
        raise RuntimeError(
            "Outer fold count does not "
            "match system count."
        )

    nested_outer_records = []
    manifest_records = []

    total_inner_folds = 0

    global_train_min = np.full(
        len(target_names),
        np.iinfo(np.int64).max,
        dtype=np.int64,
    )

    global_train_max = np.zeros(
        len(target_names),
        dtype=np.int64,
    )

    global_valid_min = np.full(
        len(target_names),
        np.iinfo(np.int64).max,
        dtype=np.int64,
    )

    global_valid_max = np.zeros(
        len(target_names),
        dtype=np.int64,
    )

    for expected_outer_index, outer in enumerate(
        outer_folds,
        start=1,
    ):
        outer_fold_id = (
            f"fold_{expected_outer_index:02d}"
        )

        if (
            outer["fold_id"]
            != outer_fold_id
        ):
            raise RuntimeError(
                f"{outer_fold_id}: "
                "outer fold ordering mismatch."
            )

        outer_test_system = str(
            outer["test_system"]
        )

        outer_train_systems = list(
            outer["train_systems"]
        )

        if (
            len(outer_train_systems)
            != 10
        ):
            raise RuntimeError(
                f"{outer_fold_id}: "
                "expected 10 outer-train "
                "systems."
            )

        if (
            outer_test_system
            in outer_train_systems
        ):
            raise RuntimeError(
                f"{outer_fold_id}: "
                "outer-test leakage."
            )

        outer_test_selector = (
            system_ids
            == outer_test_system
        )

        outer_test_indices = np.where(
            outer_test_selector
        )[0]

        if len(outer_test_indices) != 3:
            raise RuntimeError(
                f"{outer_fold_id}: "
                "expected 3 outer-test "
                "samples."
            )

        recorded_outer_test = (
            outer["test_indices"]
        )

        if (
            recorded_outer_test
            != outer_test_indices.tolist()
        ):
            raise RuntimeError(
                f"{outer_fold_id}: "
                "outer test indices differ "
                "from dataset."
            )

        validation_appearances = {
            system: 0
            for system
            in outer_train_systems
        }

        inner_records = []

        for inner_index in range(5):
            inner_fold_id = (
                f"inner_{inner_index + 1:02d}"
            )

            nested_fold_id = (
                f"{outer_fold_id}_"
                f"{inner_fold_id}"
            )

            start = (
                inner_index * 2
            )

            valid_systems = (
                outer_train_systems[
                    start:start + 2
                ]
            )

            train_systems = [
                system
                for system
                in outer_train_systems
                if system
                not in valid_systems
            ]

            if len(valid_systems) != 2:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "validation group size "
                    "is not 2."
                )

            if len(train_systems) != 8:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "training group size "
                    "is not 8."
                )

            for system in valid_systems:
                validation_appearances[
                    system
                ] += 1

            train_selector = np.isin(
                system_ids,
                train_systems,
            )

            valid_selector = np.isin(
                system_ids,
                valid_systems,
            )

            train_indices = np.where(
                train_selector
            )[0]

            valid_indices = np.where(
                valid_selector
            )[0]

            if len(train_indices) != 24:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "expected 24 train "
                    "samples."
                )

            if len(valid_indices) != 6:
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "expected 6 validation "
                    "samples."
                )

            train_set = set(
                system_ids[
                    train_indices
                ].tolist()
            )

            valid_set = set(
                system_ids[
                    valid_indices
                ].tolist()
            )

            test_set = set(
                system_ids[
                    outer_test_indices
                ].tolist()
            )

            if (
                train_set & valid_set
                or train_set & test_set
                or valid_set & test_set
            ):
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "protein leakage."
                )

            if (
                outer_test_system
                in train_set
                or outer_test_system
                in valid_set
            ):
                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "outer test is not blind."
                )

            train_counts = (
                target_mask[
                    train_indices
                ]
                .sum(axis=0)
                .astype(int)
            )

            valid_counts = (
                target_mask[
                    valid_indices
                ]
                .sum(axis=0)
                .astype(int)
            )

            test_counts = (
                target_mask[
                    outer_test_indices
                ]
                .sum(axis=0)
                .astype(int)
            )

            if np.any(
                train_counts == 0
            ):
                missing = [
                    target
                    for target, count
                    in zip(
                        target_names,
                        train_counts,
                    )
                    if count == 0
                ]

                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "inner-train missing "
                    f"{missing}"
                )

            if np.any(
                valid_counts == 0
            ):
                missing = [
                    target
                    for target, count
                    in zip(
                        target_names,
                        valid_counts,
                    )
                    if count == 0
                ]

                raise RuntimeError(
                    f"{nested_fold_id}: "
                    "inner-valid missing "
                    f"{missing}"
                )

            global_train_min = np.minimum(
                global_train_min,
                train_counts,
            )

            global_train_max = np.maximum(
                global_train_max,
                train_counts,
            )

            global_valid_min = np.minimum(
                global_valid_min,
                valid_counts,
            )

            global_valid_max = np.maximum(
                global_valid_max,
                valid_counts,
            )

            inner_record = {
                "inner_fold_id": (
                    inner_fold_id
                ),
                "nested_fold_id": (
                    nested_fold_id
                ),
                "outer_fold_id": (
                    outer_fold_id
                ),
                "outer_test_system": (
                    outer_test_system
                ),
                "inner_train_systems": (
                    train_systems
                ),
                "inner_valid_systems": (
                    valid_systems
                ),
                "outer_test_systems": [
                    outer_test_system
                ],
                "inner_train_indices": (
                    train_indices.tolist()
                ),
                "inner_valid_indices": (
                    valid_indices.tolist()
                ),
                "outer_test_indices": (
                    outer_test_indices.tolist()
                ),
                "inner_train_sample_ids": [
                    sample_ids[i]
                    for i
                    in train_indices
                ],
                "inner_valid_sample_ids": [
                    sample_ids[i]
                    for i
                    in valid_indices
                ],
                "outer_test_sample_ids": [
                    sample_ids[i]
                    for i
                    in outer_test_indices
                ],
                "inner_train_target_counts": {
                    target: int(count)
                    for target, count
                    in zip(
                        target_names,
                        train_counts,
                    )
                },
                "inner_valid_target_counts": {
                    target: int(count)
                    for target, count
                    in zip(
                        target_names,
                        valid_counts,
                    )
                },
                "outer_test_target_counts": {
                    target: int(count)
                    for target, count
                    in zip(
                        target_names,
                        test_counts,
                    )
                },
                "outer_test_unobserved_targets": [
                    target
                    for target, count
                    in zip(
                        target_names,
                        test_counts,
                    )
                    if count == 0
                ],
            }

            inner_records.append(
                inner_record
            )

            for sample_index in range(
                n_samples
            ):
                if (
                    sample_index
                    in set(
                        train_indices.tolist()
                    )
                ):
                    role = "inner_train"
                elif (
                    sample_index
                    in set(
                        valid_indices.tolist()
                    )
                ):
                    role = "inner_valid"
                elif (
                    sample_index
                    in set(
                        outer_test_indices.tolist()
                    )
                ):
                    role = "outer_test"
                else:
                    raise RuntimeError(
                        f"{nested_fold_id}: "
                        f"sample {sample_index} "
                        "has no role."
                    )

                manifest_records.append(
                    {
                        "outer_fold_id": (
                            outer_fold_id
                        ),
                        "inner_fold_id": (
                            inner_fold_id
                        ),
                        "nested_fold_id": (
                            nested_fold_id
                        ),
                        "outer_test_system": (
                            outer_test_system
                        ),
                        "sample_index": (
                            sample_index
                        ),
                        "sample_id": (
                            sample_ids[
                                sample_index
                            ]
                        ),
                        "system": str(
                            system_ids[
                                sample_index
                            ]
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

            total_inner_folds += 1

        if not all(
            count == 1
            for count
            in validation_appearances.values()
        ):
            raise RuntimeError(
                f"{outer_fold_id}: "
                "each outer-train protein "
                "must be validation exactly "
                "once."
            )

        nested_outer_records.append(
            {
                "outer_fold_id": (
                    outer_fold_id
                ),
                "outer_test_system": (
                    outer_test_system
                ),
                "outer_train_systems": (
                    outer_train_systems
                ),
                "outer_test_indices": (
                    outer_test_indices.tolist()
                ),
                "inner_pairing_rule": (
                    "consecutive_pairs_in_"
                    "outer_train_system_order"
                ),
                "inner_folds": (
                    inner_records
                ),
            }
        )

    if total_inner_folds != 55:
        raise RuntimeError(
            f"Expected 55 inner folds, "
            f"found {total_inner_folds}."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    nested_path = (
        output_dir
        / "nested_cv.json"
    )

    nested_path.write_text(
        json.dumps(
            {
                "strategy": (
                    "nested_protein_group_cv"
                ),
                "outer_strategy": (
                    "leave_one_protein_out"
                ),
                "inner_strategy": (
                    "deterministic_5fold_"
                    "protein_group_cv"
                ),
                "inner_pairing_rule": (
                    "consecutive_pairs_in_"
                    "outer_train_system_order"
                ),
                "n_outer_folds": (
                    len(nested_outer_records)
                ),
                "n_inner_folds_per_outer": 5,
                "n_nested_folds": (
                    total_inner_folds
                ),
                "outer_folds": (
                    nested_outer_records
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = (
        output_dir
        / "nested_sample_manifest.csv"
    )

    manifest_columns = [
        "outer_fold_id",
        "inner_fold_id",
        "nested_fold_id",
        "outer_test_system",
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
            manifest_records
        )

    metadata = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "split_type": (
            "deterministic_nested_"
            "protein_group_cv"
        ),
        "dataset_npz": str(
            dataset_npz
        ),
        "dataset_npz_sha256": (
            sha256_file(
                dataset_npz
            )
        ),
        "outer_folds_file": str(
            outer_folds_path
        ),
        "outer_folds_sha256": (
            sha256_file(
                outer_folds_path
            )
        ),
        "group_key": "system_id",
        "shuffle": False,
        "random_seed": None,
        "inner_pairing_rule": (
            "consecutive_pairs_in_"
            "outer_train_system_order"
        ),
        "n_systems": len(systems),
        "systems": systems,
        "n_samples": n_samples,
        "n_outer_folds": (
            len(nested_outer_records)
        ),
        "n_inner_folds_per_outer": 5,
        "n_nested_folds": (
            total_inner_folds
        ),
        "inner_train_proteins": 8,
        "inner_valid_proteins": 2,
        "outer_test_proteins": 1,
        "inner_train_samples": 24,
        "inner_valid_samples": 6,
        "outer_test_samples": 3,
        "target_names": target_names,
        "global_inner_train_min_counts": {
            target: int(count)
            for target, count
            in zip(
                target_names,
                global_train_min,
            )
        },
        "global_inner_train_max_counts": {
            target: int(count)
            for target, count
            in zip(
                target_names,
                global_train_max,
            )
        },
        "global_inner_valid_min_counts": {
            target: int(count)
            for target, count
            in zip(
                target_names,
                global_valid_min,
            )
        },
        "global_inner_valid_max_counts": {
            target: int(count)
            for target, count
            in zip(
                target_names,
                global_valid_max,
            )
        },
        "invariants": {
            "outer_test_always_blind": True,
            "protein_group_leakage": False,
            "inner_train_has_each_target": True,
            "inner_valid_has_each_target": True,
            "each_outer_train_protein_valid_once": True,
            "normalization_applied": False,
            "imputation_applied": False,
            "target_mask_modified": False,
            "scaler_fitted": False,
        },
    }

    metadata_path = (
        output_dir
        / "nested_cv_metadata.json"
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

    checksum_files = [
        nested_path,
        manifest_path,
        metadata_path,
    ]

    checksums_path = (
        output_dir
        / "SHA256SUMS.txt"
    )

    with checksums_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            checksum_files,
            key=lambda item: (
                item.name
            ),
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )

    print(
        "========================================"
    )
    print("NESTED GROUP CV BUILD: PASS")
    print(
        "========================================"
    )
    print(
        f"Outer folds:  "
        f"{len(nested_outer_records)}"
    )
    print(
        "Inner/outer:  5"
    )
    print(
        f"Nested folds: "
        f"{total_inner_folds}"
    )
    print(
        "Train/valid/test samples: "
        "24/6/3"
    )
    print()

    for outer in nested_outer_records:
        print(
            f'{outer["outer_fold_id"]} '
            f'OUTER_TEST='
            f'{outer["outer_test_system"]}'
        )

        for inner in outer[
            "inner_folds"
        ]:
            valid_text = "+".join(
                inner[
                    "inner_valid_systems"
                ]
            )

            print(
                f'  '
                f'{inner["inner_fold_id"]} '
                f'VALID={valid_text}'
            )

    print()
    print(
        "Outer test always blind: PASS"
    )
    print(
        "Protein group leakage: PASS"
    )
    print(
        "Inner train coverage: PASS"
    )
    print(
        "Inner valid coverage: PASS"
    )
    print(
        "Each outer-train protein "
        "validation once: PASS"
    )
    print(
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
