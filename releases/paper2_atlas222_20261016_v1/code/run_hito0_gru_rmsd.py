#!/usr/bin/env python3
"""Train-only epoch selection for a small residual GRU; evaluate ATLAS80 validation RMSD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
from torch import nn

from src.paper2.run_hito0_baselines import load_dataset, metrics, read_split, sha256


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(8, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1]).squeeze(-1)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def prepare(x: np.ndarray, y: np.ndarray, fit_indices: np.ndarray):
    mean = x[fit_indices].mean(axis=(0, 1))
    std = x[fit_indices].std(axis=(0, 1))
    std[std < 1e-8] = 1.0
    residual = y[fit_indices] - x[fit_indices, -1, 0]
    residual_scale = float(np.std(residual))
    if not np.isfinite(residual_scale) or residual_scale < 1e-8:
        raise ValueError("Residual RMSD without usable training variance")
    return mean, std, residual_scale


def fit_model(x, y, indices, mean, std, scale, seed, epochs, hidden_size):
    setup_seed(seed)
    model = ResidualGRU(hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    features = torch.as_tensor(((x[indices] - mean) / std).astype(np.float32))
    residual = torch.as_tensor(
        ((y[indices] - x[indices, -1, 0]) / scale).astype(np.float32)
    )
    for _ in range(epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = torch.mean((model(features) - residual).square())
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
    return model


def predict(model, x, indices, mean, std, scale):
    model.eval()
    features = torch.as_tensor(((x[indices] - mean) / std).astype(np.float32))
    with torch.no_grad():
        residual = model(features).numpy().astype(np.float64) * scale
    return x[indices, -1, 0] + residual


def protein_errors(y, predicted, groups):
    errors = []
    for group in sorted(set(groups)):
        selected = groups == group
        errors.append(abs(float(y[selected].mean() - predicted[selected].mean())))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=150)
    args = parser.parse_args()
    dataset, split, output = (p.expanduser().resolve() for p in
                              (args.dataset_npz, args.split_csv, args.output_dir))
    if output.exists():
        raise FileExistsError(output)
    if args.hidden_size < 1 or args.max_epochs < 1:
        parser.error("hidden-size and max-epochs must be positive")
    frozen = json.loads(split.with_name("split_metadata.json").read_text())
    if frozen["dataset_sha256"] != sha256(dataset) or frozen["split_sha256"] != sha256(split):
        raise ValueError("Dataset or split differs from frozen hashes")
    data = load_dataset(dataset)
    x = data["X_raw"].astype(np.float64)
    groups = data["groups"]
    partition = read_split(split, groups)
    parts = np.asarray([partition[group] for group in groups])
    valid = data["target_mask"][:, 0]
    y = data["target_values"][:, 0]

    train_groups = sorted({g for g in groups[parts == "train"]},
                          key=lambda g: (hashlib.sha256(f"{args.seed}:{g}".encode()).digest(), g))
    n_inner = max(8, round(len(train_groups) * 0.2))
    inner_groups = set(train_groups[:n_inner])
    inner_fit = np.flatnonzero(valid & (parts == "train") & ~np.isin(groups, list(inner_groups)))
    inner_eval = np.flatnonzero(valid & np.isin(groups, list(inner_groups)))
    full_fit = np.flatnonzero(valid & (parts == "train"))
    evaluation = np.flatnonzero(valid & (parts == "validation"))
    if min(len(set(groups[i])) for i in (inner_fit, inner_eval, evaluation)) < 2:
        raise ValueError("Insufficient labeled proteins in inner/validation partitions")

    mean, std, scale = prepare(x, y, inner_fit)
    setup_seed(args.seed)
    model = ResidualGRU(args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    features = torch.as_tensor(((x[inner_fit] - mean) / std).astype(np.float32))
    residual = torch.as_tensor(((y[inner_fit] - x[inner_fit, -1, 0]) / scale).astype(np.float32))
    best_epoch, best_score, stale = 0, float("inf"), 0
    history = []
    for epoch in range(args.max_epochs + 1):
        if epoch:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(features) - residual).square())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        predicted = predict(model, x, inner_eval, mean, std, scale)
        score = float(np.mean(protein_errors(y[inner_eval], predicted, groups[inner_eval])))
        history.append({"epoch": epoch, "inner_protein_mae": score})
        if score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        if stale >= 20:
            break

    full_mean, full_std, full_scale = prepare(x, y, full_fit)
    final = fit_model(x, y, full_fit, full_mean, full_std, full_scale,
                      args.seed, best_epoch, args.hidden_size)
    prediction = predict(final, x, evaluation, full_mean, full_std, full_scale)
    protein_rows = []
    for group in sorted(set(groups[evaluation])):
        selected = groups[evaluation] == group
        protein_rows.append({
            "stage": "development", "split": "validation", "system_id": group,
            "target": "rmsd", "model": "gru_residual_rmsd_pilot",
            "n_valid_replicas": int(selected.sum()),
            "true": float(y[evaluation][selected].mean()),
            "predicted": float(prediction[selected].mean()),
        })
    score = metrics(np.asarray([r["true"] for r in protein_rows]),
                    np.asarray([r["predicted"] for r in protein_rows]), args.seed)
    output.mkdir(parents=True)
    with (output / "protein_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(protein_rows[0]))
        writer.writeheader()
        writer.writerows(protein_rows)
    with (output / "inner_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save({"state_dict": final.state_dict(), "feature_mean": full_mean,
                "feature_std": full_std, "residual_scale": full_scale}, output / "model.pt")
    (output / "run_metadata.json").write_text(json.dumps({
        "dataset_sha256": sha256(dataset), "split_sha256": sha256(split),
        "code_sha256": sha256(Path(__file__)), "seed": args.seed,
        "train_only_inner_groups": sorted(inner_groups), "selected_epoch": best_epoch,
        "inner_best_protein_mae": best_score, "hidden_size": args.hidden_size,
        "n_valid_train_proteins": len(set(groups[full_fit])),
        "n_valid_validation_proteins": len(protein_rows),
        "validation_used_for_epoch_selection": False, "test_used_for_evaluation": False,
        "protein_metric": score,
    }, indent=2, sort_keys=True) + "\n")
    print(f"GRU_RMSD_DEV=PASS selected_epoch={best_epoch} "
          f"n={len(protein_rows)} mae={score['mae']:.6f} r2={score['r2']:.6f}")
    print(f"PREDICTIONS={output / 'protein_predictions.csv'}")


if __name__ == "__main__":
    main()
