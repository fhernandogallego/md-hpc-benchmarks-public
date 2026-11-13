"""Shared helpers for the paper-2 extension calculations.

Ridge, prefix summaries and the residual GRU reproduce the definitions used in the
ATLAS942 run (src/paper2/run_hito0_baselines.py, src/modeling/train_ridge_summary_outer.py,
run_hito0_gru_rmsd.py), so results are comparable with the published tables.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

FEATURES = ("rmsd_backbone_A", "radius_gyration_A", "core_rmsd_q30_A",
            "core_rmsd_q40_A", "core_rmsd_q50_A", "total_sasa_A2",
            "helix_fraction", "strand_fraction")
SEED = 20261007
ALPHAS_ORIGINAL = [0.01, 0.1, 1.0, 10.0, 100.0]
ALPHAS_EXTENDED = [0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]


# ---------------------------------------------------------------- I/O
def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, default=float) + "\n")


def strings(values) -> np.ndarray:
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def load_folds(run: Path) -> dict[str, dict]:
    """outer_folds.csv of an existing run: system_id -> {group_id, outer_fold}."""
    rows = read_csv(Path(run) / "outer_folds.csv")
    folds = {r["system_id"]: {"group_id": r["group_id"], "outer_fold": int(r["outer_fold"])} for r in rows}
    for group in {r["group_id"] for r in rows}:
        if len({r["outer_fold"] for r in rows if r["group_id"] == group}) != 1:
            raise ValueError(f"Group {group} appears in more than one outer fold")
    return folds


def load_prefix(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dataset.npz of an existing run: (sample_ids, system_ids, X_raw[n, 201, 8])."""
    with np.load(Path(run) / "dataset.npz", allow_pickle=False) as z:
        if tuple(strings(z["feature_names"])) != FEATURES:
            raise ValueError("Unexpected feature order in dataset.npz")
        if not np.allclose(z["time_ns"], np.linspace(0, 20, 201), atol=1e-8):
            raise ValueError("Expected 201 prefix points between 0 and 20 ns")
        return strings(z["sample_ids"]), strings(z["system_ids"]), z["X_raw"].astype(np.float64)


# ---------------------------------------------------------------- models
def summarize(x: np.ndarray) -> np.ndarray:
    """Six summaries per channel (mean, std, min, max, last, slope on t in [-1, 1]); any prefix length."""
    t = np.linspace(-1.0, 1.0, x.shape[1])
    slope = np.einsum("t,ntf->nf", t, x) / float(np.sum(t * t))
    return np.concatenate([x.mean(1), x.std(1, ddof=0), x.min(1), x.max(1), x[:, -1, :], slope], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple:
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1
    y_mean, y_std = float(y.mean()), float(y.std()) or 1.0
    design = np.column_stack((np.ones(len(x)), (x - mean) / std))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ ((y - y_mean) / y_std))
    return mean, std, y_mean, y_std, coef


def predict_ridge(fit: tuple, x: np.ndarray) -> np.ndarray:
    mean, std, y_mean, y_std, coef = fit
    return y_mean + y_std * (coef[0] + ((x - mean) / std) @ coef[1:])


def ordered(values, seed: int) -> list:
    return sorted(set(values), key=lambda g: (hashlib.sha256(f"{seed}:{g}".encode()).digest(), g))


def system_errors(y: np.ndarray, p: np.ndarray, systems: np.ndarray) -> list[float]:
    return [abs(float(y[systems == s].mean() - p[systems == s].mean())) for s in sorted(set(systems))]


def inner_pairs(fit_idx: np.ndarray, groups: np.ndarray, seed: int, n: int = 4):
    bins = np.array_split(np.asarray(ordered(groups[fit_idx], seed)), n)
    pairs = []
    for held in bins:
        ev = fit_idx[np.isin(groups[fit_idx], held)]
        tr = fit_idx[~np.isin(groups[fit_idx], held)]
        pairs.append((tr, ev))
    return pairs


def ridge_cv_predict(features, y, systems, groups, fit_idx, eval_idx, alphas, seed):
    """Select alpha by inner grouped CV (system-level MAE), refit on fit_idx, predict eval_idx."""
    scores = {}
    for alpha in alphas:
        errs = []
        for tr, ev in inner_pairs(fit_idx, groups, seed):
            errs.extend(system_errors(y[ev], predict_ridge(fit_ridge(features[tr], y[tr], alpha), features[ev]), systems[ev]))
        scores[alpha] = float(np.mean(errs))
    alpha = min(alphas, key=lambda a: (scores[a], a))
    return predict_ridge(fit_ridge(features[fit_idx], y[fit_idx], alpha), features[eval_idx]), alpha


def run_gru(x, y, systems, groups, fit_idx, eval_idx, seed, hidden=32, max_epochs=150, patience=20):
    """Residual GRU on channel 0 (persistence = last value of channel 0), as in the ATLAS942 run."""
    import random
    import torch
    from torch import nn

    def setup(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

    class ResidualGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(x.shape[2], hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        def forward(self, t):
            _, h = self.gru(t)
            return self.head(h[-1]).squeeze(-1)

    def prepare(idx):
        mean, std = x[idx].mean((0, 1)), x[idx].std((0, 1))
        std[std < 1e-8] = 1.0
        scale = float(np.std(y[idx] - x[idx, -1, 0]))
        if not np.isfinite(scale) or scale < 1e-8:
            raise ValueError("Residual without usable variance")
        return mean, std, scale

    def predict(model, idx, mean, std, scale):
        model.eval()
        with torch.no_grad():
            r = model(torch.as_tensor(((x[idx] - mean) / std).astype(np.float32))).numpy().astype(np.float64)
        return x[idx, -1, 0] + r * scale

    def train(idx, mean, std, scale, epochs, eval_idx_inner=None):
        setup(seed)
        model = ResidualGRU()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        feats = torch.as_tensor(((x[idx] - mean) / std).astype(np.float32))
        target = torch.as_tensor(((y[idx] - x[idx, -1, 0]) / scale).astype(np.float32))
        best, best_score, stale = 0, float("inf"), 0
        for epoch in range(epochs + 1):
            if epoch:
                model.train(); opt.zero_grad(set_to_none=True)
                loss = torch.mean((model(feats) - target).square())
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if eval_idx_inner is not None:
                score = float(np.mean(system_errors(y[eval_idx_inner], predict(model, eval_idx_inner, mean, std, scale), systems[eval_idx_inner])))
                if score < best_score - 1e-5:
                    best, best_score, stale = epoch, score, 0
                else:
                    stale += 1
                if stale >= patience:
                    break
        return model, best

    distinct = ordered(groups[fit_idx], seed)
    held = distinct[:max(5, round(len(distinct) * 0.2))]
    inner_eval = fit_idx[np.isin(groups[fit_idx], held)]
    inner_fit = fit_idx[~np.isin(groups[fit_idx], held)]
    m, s, sc = prepare(inner_fit)
    _, epoch = train(inner_fit, m, s, sc, max_epochs, inner_eval)
    m, s, sc = prepare(fit_idx)
    model, _ = train(fit_idx, m, s, sc, epoch)
    return predict(model, eval_idx, m, s, sc), epoch


# ---------------------------------------------------------------- evaluation
def to_system_level(rows: list[dict]) -> list[dict]:
    """Average replica predictions/targets within (key..., system)."""
    acc = {}
    for r in rows:
        k = tuple((c, r[c]) for c in r if c not in ("sample_id", "true", "predicted"))
        a = acc.setdefault(k, [[], []])
        a[0].append(float(r["true"])); a[1].append(float(r["predicted"]))
    return [{**dict(k), "true": float(np.mean(a[0])), "predicted": float(np.mean(a[1]))} for k, a in acc.items()]


def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    total = float(np.sum((true - true.mean()) ** 2))
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": 1 - float(np.sum(err ** 2)) / total if total else float("nan")}


def group_bootstrap_delta(err_a: np.ndarray, err_b: np.ndarray, groups: np.ndarray, draws=10000, seed=20261016):
    """Paired difference in system-level MAE (a - b) with a 95% percentile interval that resamples
    whole sequence groups (same draws for both models). Vectorised through group sums."""
    names, inv = np.unique(groups, return_inverse=True)
    g = len(names)
    sum_a = np.bincount(inv, weights=np.abs(err_a), minlength=g)
    sum_b = np.bincount(inv, weights=np.abs(err_b), minlength=g)
    count = np.bincount(inv, minlength=g).astype(float)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, g, g), minlength=g) for _ in range(draws)]).astype(float)
    n = w @ count
    boot = (w @ sum_a) / n - (w @ sum_b) / n
    point = float(np.abs(err_a).mean() - np.abs(err_b).mean())
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
