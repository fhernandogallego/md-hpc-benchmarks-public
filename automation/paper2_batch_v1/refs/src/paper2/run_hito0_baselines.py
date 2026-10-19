#!/usr/bin/env python3
"""Protein-wise ATLAS scale baselines on a frozen HITO 0 split.

Development evaluates validation proteins. Final evaluates test proteins only
after model choices are fixed. The held-out proteins never enter fitting,
feature scaling, target scaling, or Ridge alpha selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.modeling.train_ridge_summary_outer import summarize_sequence


TARGETS = ("rmsd", "rg", "sasa", "helix", "strand")
FEATURES = (
    "rmsd_backbone_A", "radius_gyration_A", "core_rmsd_q30_A",
    "core_rmsd_q40_A", "core_rmsd_q50_A", "total_sasa_A2",
    "helix_fraction", "strand_fraction",
)
PERSISTENCE_CHANNEL = (0, 1, 5, 6, 7)
MODELS = ("trained_mean", "last_value_persistence", "ridge_summary")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strings(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in values.tolist()]


def load_dataset(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"X_raw", "time_ns", "system_ids", "sample_ids",
                    "feature_names", "target_names", "target_values", "target_mask"}
        if missing := required - set(archive.files):
            raise ValueError(f"Missing dataset arrays: {sorted(missing)}")
        data = {key: archive[key].copy() for key in required}
    x, y, mask = data["X_raw"], data["target_values"], data["target_mask"]
    n = x.shape[0]
    if x.shape != (n, 201, 8) or y.shape != (n, 5) or mask.shape != y.shape:
        raise ValueError(f"Unexpected X/y/mask shapes: {x.shape}/{y.shape}/{mask.shape}")
    if data["time_ns"].shape != (201,) or not np.allclose(
        data["time_ns"], np.linspace(0, 20, 201), rtol=0, atol=1e-8
    ):
        raise ValueError("Expected the frozen 0–20 ns, 100 ps grid")
    if tuple(strings(data["feature_names"])) != FEATURES:
        raise ValueError("Feature order differs from the ATLAS20 representation")
    if tuple(strings(data["target_names"])) != TARGETS:
        raise ValueError("Target order differs from the ATLAS20 representation")
    if len(data["system_ids"]) != n or len(data["sample_ids"]) != n:
        raise ValueError("System/sample identifier length mismatch")
    if len(set(strings(data["sample_ids"]))) != n:
        raise ValueError("Duplicate sample identifiers")
    mask = mask.astype(bool)
    if not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
        raise ValueError("Non-finite inputs or valid targets")
    if not np.isnan(y[~mask]).all():
        raise ValueError("Invalid targets must be NaN")
    data["target_mask"] = mask
    data["groups"] = np.asarray(strings(data["system_ids"]))
    data["ids"] = strings(data["sample_ids"])
    return data


def read_split(path: Path, groups: np.ndarray) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"system_id", "split", "is_prior_atlas20"}.issubset(reader.fieldnames or []):
            raise ValueError("Split needs system_id, split, is_prior_atlas20")
        rows = list(reader)
    mapping = {}
    for row in rows:
        group, part = row["system_id"], row["split"]
        if group in mapping or part not in {"train", "validation", "test"}:
            raise ValueError("Duplicate group or unknown split")
        if row["is_prior_atlas20"] not in ("true", "false"):
            raise ValueError("Invalid prior-ATLAS20 flag")
        if row["is_prior_atlas20"] == "true" and part != "train":
            raise ValueError(f"Prior ATLAS20 protein {group} assigned to {part}")
        mapping[group] = part
    if set(mapping) != set(groups):
        raise ValueError("Split and dataset protein groups do not match exactly")
    for part in ("train", "validation", "test"):
        if not any(value == part for value in mapping.values()):
            raise ValueError(f"Empty {part} split")
    return mapping


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1
    y_mean = float(y.mean())
    y_std = float(y.std()) or 1.0
    z = (x - mean) / std
    design = np.column_stack((np.ones(len(z)), z))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ ((y - y_mean) / y_std),
    )
    return mean, std, y_mean, y_std, coefficients


def predict_ridge(fit: tuple, x: np.ndarray) -> np.ndarray:
    mean, std, y_mean, y_std, coefficients = fit
    return y_mean + y_std * (coefficients[0] + ((x - mean) / std) @ coefficients[1:])


def protein_mean_absolute_error(
    truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray,
) -> float:
    return float(np.mean([
        abs(float(truth[groups == group].mean() - prediction[groups == group].mean()))
        for group in np.unique(groups)
    ]))


def choose_alpha(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    alphas: list[float], folds: int, seed: int,
) -> tuple[float, dict[str, float]]:
    distinct = sorted(set(groups), key=lambda group: (
        hashlib.sha256(f"{seed}:{group}".encode()).digest(), group
    ))
    if len(distinct) < folds:
        raise ValueError(f"Only {len(distinct)} valid training proteins for {folds} folds")
    bins = np.array_split(distinct, folds)
    scores = {}
    for alpha in alphas:
        truths, predictions, group_ids = [], [], []
        for held_out in bins:
            validation = np.isin(groups, held_out)
            training = ~validation
            fit = fit_ridge(x[training], y[training], alpha)
            truths.extend(y[validation])
            predictions.extend(predict_ridge(fit, x[validation]))
            group_ids.extend(groups[validation])
        scores[str(alpha)] = protein_mean_absolute_error(
            np.asarray(truths), np.asarray(predictions), np.asarray(group_ids)
        )
    return min(alphas, key=lambda alpha: (scores[str(alpha)], alpha)), scores


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metrics(truth: np.ndarray, predicted: np.ndarray, seed: int) -> dict:
    error = predicted - truth
    n = len(error)
    rng = np.random.default_rng(seed)
    boot = np.abs(error)[rng.integers(0, n, size=(2000, n))].mean(axis=1)
    total = np.sum((truth - truth.mean()) ** 2)
    return {
        "n_proteins": n,
        "mae": float(np.mean(np.abs(error))),
        "mae_ci_low": float(np.quantile(boot, 0.025)),
        "mae_ci_high": float(np.quantile(boot, 0.975)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1 - np.sum(error ** 2) / total) if total > 0 else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "final"),
                        default="development")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.inner_folds < 2 or not args.alphas or any(a <= 0 for a in args.alphas):
        parser.error("Need >=2 inner folds and strictly positive alphas")
    dataset = args.dataset_npz.expanduser().resolve()
    split = args.split_csv.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    metadata_path = split.with_name("split_metadata.json")
    frozen = json.loads(metadata_path.read_text(encoding="utf-8"))
    if frozen["dataset_sha256"] != sha256(dataset) or frozen["split_sha256"] != sha256(split):
        raise ValueError("Dataset or split changed since the split was frozen")
    data = load_dataset(dataset)
    groups = data["groups"]
    partitions = read_split(split, groups)
    partition = np.asarray([partitions[group] for group in groups])
    eval_part = "validation" if args.stage == "development" else "test"
    summary = summarize_sequence(data["X_raw"])
    if summary.shape != (len(groups), 48):
        raise ValueError("Unexpected summary shape")

    replica_rows, protein_rows, score_rows, selected = [], [], [], {}
    for index, target in enumerate(TARGETS):
        valid = data["target_mask"][:, index]
        train = np.flatnonzero(valid & (partition == "train"))
        evaluation = np.flatnonzero(valid & (partition == eval_part))
        if len(set(groups[train])) < args.inner_folds or len(set(groups[evaluation])) < 2:
            raise ValueError(f"{target}: insufficient valid groups in train/{eval_part}")
        y = data["target_values"][:, index]
        alpha, inner_scores = choose_alpha(
            summary[train], y[train], groups[train], args.alphas,
            args.inner_folds, args.seed + index,
        )
        fit = fit_ridge(summary[train], y[train], alpha)
        by_train_group = [y[train][groups[train] == group].mean()
                          for group in np.unique(groups[train])]
        mean_prediction = float(np.mean(by_train_group))
        predictions = {
            "trained_mean": np.full(len(evaluation), mean_prediction),
            "last_value_persistence": data["X_raw"][
                evaluation, -1, PERSISTENCE_CHANNEL[index]
            ],
            "ridge_summary": predict_ridge(fit, summary[evaluation]),
        }
        selected[target] = {
            "ridge_alpha": alpha,
            "inner_protein_mae_by_alpha": inner_scores,
            "n_valid_train_replicas": len(train),
            "n_valid_train_proteins": len(set(groups[train])),
            "n_valid_eval_replicas": len(evaluation),
            "n_valid_eval_proteins": len(set(groups[evaluation])),
        }
        for pos, sample_index in enumerate(evaluation):
            for model in MODELS:
                replica_rows.append({
                    "stage": args.stage, "split": eval_part,
                    "sample_id": data["ids"][sample_index],
                    "system_id": groups[sample_index],
                    "target": target, "model": model,
                    "true": float(y[sample_index]),
                    "predicted": float(predictions[model][pos]),
                })
        for group in sorted(set(groups[evaluation])):
            selected_rows = np.flatnonzero(groups[evaluation] == group)
            for model in MODELS:
                protein_rows.append({
                    "stage": args.stage, "split": eval_part,
                    "system_id": group, "target": target, "model": model,
                    "n_valid_replicas": len(selected_rows),
                    "true": float(np.mean(y[evaluation[selected_rows]])),
                    "predicted": float(np.mean(predictions[model][selected_rows])),
                })
        for model in MODELS:
            rows = [row for row in protein_rows if row["target"] == target
                    and row["model"] == model]
            values = metrics(np.array([row["true"] for row in rows]),
                             np.array([row["predicted"] for row in rows]),
                             args.seed + index)
            score_rows.append({"stage": args.stage, "target": target,
                               "model": model, **values})

    output.mkdir(parents=True)
    write_csv(output / "replica_predictions.csv", replica_rows,
              ["stage", "split", "sample_id", "system_id", "target", "model",
               "true", "predicted"])
    write_csv(output / "protein_predictions.csv", protein_rows,
              ["stage", "split", "system_id", "target", "model",
               "n_valid_replicas", "true", "predicted"])
    write_csv(output / "protein_metrics.csv", score_rows,
              ["stage", "target", "model", "n_proteins", "mae", "mae_ci_low",
               "mae_ci_high", "rmse", "r2"])
    (output / "run_metadata.json").write_text(json.dumps({
        "stage": args.stage, "evaluated_split": eval_part,
        "dataset_sha256": frozen["dataset_sha256"],
        "split_sha256": frozen["split_sha256"],
        "code_sha256": sha256(Path(__file__)),
        "seed": args.seed, "inner_folds": args.inner_folds,
        "alphas": args.alphas, "selection": selected,
        "protein_aggregation": "mean of valid replicas per protein before metric",
        "ridge_alpha_objective": "mean protein-level absolute error in train-only grouped CV",
        "trained_mean": "mean of train protein means for each valid target",
        "test_labels_used_for_metrics": args.stage == "final",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Evaluated {eval_part} proteins; results: {output}")


if __name__ == "__main__":
    main()

