from pathlib import Path
import sys, csv, json, hashlib
import numpy as np
import torch
from torch import nn
from src.paper2.run_hito0_baselines import fit_ridge, predict_ridge, FEATURES
from src.modeling.train_ridge_summary_outer import summarize_sequence
from run_hito0_gru_rmsd import (
    ResidualGRU, prepare, predict, fit_model, setup_seed, protein_errors
)

run = Path(sys.argv[1])
mode = sys.argv[2]
MODELS = ["trained_mean", "last_value_persistence", "snapshot0_ridge",
          "ridge_summary", "gru_residual_rmsd_pilot"]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 20261007

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))

def write(p, rows):
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def strings(a):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in a])

def ordered(values, seed):
    return sorted(set(values), key=lambda g: (
        hashlib.sha256(f"{seed}:{g}".encode()).digest(), g
    ))

# Todos los trabajos verifican las mismas copias de entradas y código.
manifest = json.loads((run / "manifest.json").read_text())
for relative, expected in manifest["files"].items():
    assert sha(run / relative) == expected, f"Archivo modificado: {relative}"

fold_rows = read(run / "outer_folds.csv")
fm = {r["system_id"]: r for r in fold_rows}
assert len(fold_rows) == len(fm) == 170
assert len({r["group_id"] for r in fold_rows}) == 149
for group in {r["group_id"] for r in fold_rows}:
    assert len({r["outer_fold"] for r in fold_rows if r["group_id"] == group}) == 1

if mode == "aggregate":
    result = []
    for fold in range(5):
        directory = run / f"fold_{fold}"
        assert (directory / "PASS").is_file(), f"Fold {fold} incompleto"
        rows = read(directory / "system_predictions.csv")
        assert len(rows) == 34 * len(MODELS)
        assert {r["system_id"] for r in rows} == {
            s for s, r in fm.items() if int(r["outer_fold"]) == fold
        }
        assert all(int(r["fold"]) == fold for r in rows)
        result.extend(rows)

    scores = []
    for model in MODELS:
        rows = [r for r in result if r["model"] == model]
        assert len(rows) == 170
        assert {r["system_id"] for r in rows} == set(fm)
        truth = np.asarray([float(r["true"]) for r in rows])
        pred = np.asarray([float(r["predicted"]) for r in rows])
        assert np.isfinite(truth).all() and np.isfinite(pred).all()
        error = pred - truth
        total = float(np.sum((truth - truth.mean()) ** 2))
        scores.append({
            "model": model, "n_systems": 170, "n_groups": 149,
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": 1 - float(np.sum(error ** 2)) / total if total else float("nan")
        })
    write(run / "system_predictions.csv", result)
    write(run / "system_metrics.csv", scores)
    (run / "summary.json").write_text(json.dumps({
        "scope": "train_only_grouped_cv_future_RMSD",
        "systems": 170, "groups": 149, "replicas": 288,
        "outer_folds": 5,
        "validation_test_evaluated": False,
        "confidence_intervals": "pending_group_resampling",
        "predictions_sha256": sha(run / "system_predictions.csv"),
        "metrics": scores
    }, indent=2) + "\n")
    for row in scores:
        print(f"{row['model']} MAE={row['mae']:.6f} "
              f"RMSE={row['rmse']:.6f} R2={row['r2']:.6f}")
    print("ATLAS222_GROUPED_CV=PASS")
    print(f"OUTPUT={run}")
    sys.exit(0)

fold = int(mode)
assert 0 <= fold < 5
output = run / f"fold_{fold}"
output.mkdir(exist_ok=False)
setup_seed(SEED + fold)

target_rows = [
    r for r in read(run / "train_future_rmsd_rows.csv")
    if r["future_valid"].lower() == "true"
]
target_rows.sort(key=lambda r: r["sample_id"])
assert len(target_rows) == 288
assert len({r["sample_id"] for r in target_rows}) == 288
assert {r["system_id"] for r in target_rows} == set(fm)

# Solo cargamos entradas e identificadores del ensamblado, no sus targets.
with np.load(run / "dataset.npz", allow_pickle=False) as z:
    ids = strings(z["sample_ids"])
    all_systems = strings(z["system_ids"])
    features = strings(z["feature_names"])
    time = z["time_ns"]
    assert tuple(features) == FEATURES
    assert np.allclose(time, np.linspace(0, 20, 201), rtol=0, atol=1e-8)
    lookup = {s: i for i, s in enumerate(ids)}
    assert len(lookup) == len(ids) == 666
    indices = [lookup[r["sample_id"]] for r in target_rows]
    x = z["X_raw"][indices].astype(np.float64)
    assert all(all_systems[i] == r["system_id"]
               for i, r in zip(indices, target_rows))

y = np.asarray([float(r["future_target_A"]) for r in target_rows])
systems = np.asarray([r["system_id"] for r in target_rows])
groups = np.asarray([fm[s]["group_id"] for s in systems])
folds = np.asarray([int(fm[s]["outer_fold"]) for s in systems])
assert x.shape == (288, 201, 8)
assert np.isfinite(x).all() and np.isfinite(y).all()
for s in fm:
    assert int(np.sum(systems == s)) == int(fm[s]["n_valid_replicas"])
assert all(r["group_id"] == fm[r["system_id"]]["group_id"] for r in target_rows)

fit = np.flatnonzero(folds != fold)
evaluation = np.flatnonzero(folds == fold)
assert set(groups[fit]).isdisjoint(groups[evaluation])
assert len(set(systems[evaluation])) == 34

# Cuatro folds internos por grupo para Ridge.
inner_bins = np.array_split(
    np.asarray(ordered(groups[fit], SEED + 100 * fold)), 4
)
inner_pairs = []
for held in inner_bins:
    ev = fit[np.isin(groups[fit], held)]
    tr = fit[~np.isin(groups[fit], held)]
    assert set(groups[tr]).isdisjoint(groups[ev])
    assert set(groups[ev]).isdisjoint(groups[evaluation])
    inner_pairs.append((tr, ev))

predictions = {
    "trained_mean": np.full(len(evaluation), np.mean([
        y[fit][systems[fit] == s].mean() for s in sorted(set(systems[fit]))
    ])),
    "last_value_persistence": x[evaluation, -1, 0]
}
selection = {}
for model, values in (
    ("snapshot0_ridge", x[:, 0, :][:, [1, 5, 6, 7]]),
    ("ridge_summary", summarize_sequence(x))
):
    scores = {}
    for alpha in ALPHAS:
        errors = []
        for tr, ev in inner_pairs:
            fitted = fit_ridge(values[tr], y[tr], alpha)
            p = predict_ridge(fitted, values[ev])
            errors.extend(protein_errors(y[ev], p, systems[ev]))
        scores[str(alpha)] = float(np.mean(errors))
    alpha = min(ALPHAS, key=lambda a: (scores[str(a)], a))
    fitted = fit_ridge(values[fit], y[fit], alpha)
    predictions[model] = predict_ridge(fitted, values[evaluation])
    mean, std, ymean, ystd, coef = fitted
    np.savez(output / f"{model}.npz",
             mean=mean, std=std, y_mean=ymean, y_std=ystd,
             coefficients=coef, alpha=alpha)
    selection[model] = {"alpha": alpha, "inner_system_mae": scores}

# Selección de época: holdout interno por grupo, puntuado por sistema.
seed = SEED + fold
distinct = ordered(groups[fit], seed)
held = distinct[:max(5, round(len(distinct) * 0.2))]
inner_eval = fit[np.isin(groups[fit], held)]
inner_fit = fit[~np.isin(groups[fit], held)]
assert set(groups[inner_fit]).isdisjoint(groups[inner_eval])
assert set(groups[inner_eval]).isdisjoint(groups[evaluation])

mean, std, scale = prepare(x, y, inner_fit)
setup_seed(seed)
model = ResidualGRU(32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
features = torch.as_tensor(((x[inner_fit] - mean) / std).astype(np.float32))
residual = torch.as_tensor(
    ((y[inner_fit] - x[inner_fit, -1, 0]) / scale).astype(np.float32)
)
best_epoch, best_score, stale = 0, float("inf"), 0
history = []
for epoch in range(151):
    if epoch:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(features) - residual).square())
        assert torch.isfinite(loss), "Loss no finita"
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    p = predict(model, x, inner_eval, mean, std, scale)
    score = float(np.mean(protein_errors(y[inner_eval], p, systems[inner_eval])))
    history.append({"epoch": epoch, "inner_system_mae": score})
    if score < best_score - 1e-5:
        best_epoch, best_score, stale = epoch, score, 0
    else:
        stale += 1
    if stale >= 20:
        break

mean, std, scale = prepare(x, y, fit)
model = fit_model(x, y, fit, mean, std, scale, seed, best_epoch, 32)
predictions["gru_residual_rmsd_pilot"] = predict(
    model, x, evaluation, mean, std, scale
)
torch.save({
    "state_dict": model.state_dict(), "feature_mean": mean,
    "feature_std": std, "residual_scale": scale,
    "hidden_size": 32, "selected_epoch": best_epoch,
}, output / "gru_model.pt")
write(output / "inner_history.csv", history)

rows = []
replica_rows = []
for name, predicted in predictions.items():
    assert np.isfinite(predicted).all()
    for j, i in enumerate(evaluation):
        replica_rows.append({
            "fold": fold, "sample_id": target_rows[i]["sample_id"],
            "system_id": systems[i], "group_id": groups[i],
            "model": name, "true": float(y[i]), "predicted": float(predicted[j])
        })
    for s in sorted(set(systems[evaluation])):
        mask = systems[evaluation] == s
        rows.append({
            "fold": fold, "system_id": s, "group_id": fm[s]["group_id"],
            "model": name, "n_valid_replicas": int(mask.sum()),
            "true": float(y[evaluation][mask].mean()),
            "predicted": float(predicted[mask].mean())
        })

write(output / "system_predictions.csv", rows)
write(output / "replica_predictions.csv", replica_rows)
selection["gru_residual_rmsd_pilot"] = {
    "epoch": best_epoch, "inner_system_mae": best_score,
    "inner_eval_groups": held, "seed": seed
}
(output / "metadata.json").write_text(json.dumps({
    "fold": fold, "selection": selection,
    "outer_eval_groups": sorted(set(groups[evaluation])),
    "ridge_inner_eval_groups": [b.tolist() for b in inner_bins],
    "group_separation_checked": True,
    "validation_test_evaluated": False,
    "torch_version": torch.__version__, "numpy_version": np.__version__,
    "fit_loss_weighting": "equal valid replicas",
    "selection_metric": "MAE of system means",
    "trained_mean_A": float(predictions["trained_mean"][0]),
}, indent=2) + "\n")
(output / "PASS").write_text("PASS\n")
print(f"FOLD={fold} PASS systems=34 gru_epoch={best_epoch}", flush=True)
