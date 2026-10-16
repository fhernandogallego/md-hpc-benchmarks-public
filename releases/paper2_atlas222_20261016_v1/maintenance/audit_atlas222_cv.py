from pathlib import Path
import csv, json, hashlib, sys
import numpy as np

run = Path(sys.argv[1])
out = run / "paired_group_audit_v1"
out.mkdir(exist_ok=False)

def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((run / "manifest.json").read_text())
for name, expected in manifest["files"].items():
    assert sha(run / name) == expected, f"Archivo modificado: {name}"

folds = read(run / "outer_folds.csv")
fm = {r["system_id"]: r for r in folds}
assert len(folds) == len(fm) == 170
group_ids = sorted({r["group_id"] for r in folds})
assert len(group_ids) == 149
for group in group_ids:
    assert len({r["outer_fold"] for r in folds if r["group_id"] == group}) == 1

targets = {
    r["sample_id"]: r
    for r in read(run / "train_future_rmsd_rows.csv")
    if r["future_valid"].lower() == "true"
}
assert len(targets) == 288
models = ["trained_mean", "last_value_persistence", "snapshot0_ridge",
          "ridge_summary", "gru_residual_rmsd_pilot"]

replicas, selection = {}, []
for fold in range(5):
    directory = run / f"fold_{fold}"
    assert (directory / "PASS").is_file()
    metadata = json.loads((directory / "metadata.json").read_text())
    expected_groups = {
        r["group_id"] for r in folds if int(r["outer_fold"]) == fold
    }
    assert set(metadata["outer_eval_groups"]) == expected_groups
    assert not metadata["validation_test_evaluated"]
    for held in metadata["ridge_inner_eval_groups"]:
        assert set(held).isdisjoint(expected_groups)
    selected = metadata["selection"]
    assert set(selected["gru_residual_rmsd_pilot"]["inner_eval_groups"]).isdisjoint(expected_groups)
    selection.append({
        "fold": fold,
        "ridge_alpha": selected["ridge_summary"]["alpha"],
        "gru_epoch": selected["gru_residual_rmsd_pilot"]["epoch"]
    })
    for row in read(directory / "replica_predictions.csv"):
        model, sid = row["model"], row["sample_id"]
        assert model in models and sid in targets
        system = row["system_id"]
        assert system == targets[sid]["system_id"]
        assert row["group_id"] == fm[system]["group_id"]
        assert int(row["fold"]) == fold == int(fm[system]["outer_fold"])
        assert np.isclose(float(row["true"]),
                          float(targets[sid]["future_target_A"]),
                          rtol=0, atol=1e-9)
        assert (model, sid) not in replicas
        replicas[model, sid] = row

assert len(replicas) == len(models) * 288
for model in models:
    assert {sid for m, sid in replicas if m == model} == set(targets)

rows = read(run / "system_predictions.csv")
by_model = {m: {} for m in models}
samples = {
    s: [sid for sid, t in targets.items() if t["system_id"] == s]
    for s in fm
}
for row in rows:
    model, system = row["model"], row["system_id"]
    assert model in models and system in fm
    assert system not in by_model[model]
    assert row["group_id"] == fm[system]["group_id"]
    assert int(row["fold"]) == int(fm[system]["outer_fold"])
    rr = [replicas[model, sid] for sid in samples[system]]
    assert len(rr) == int(row["n_valid_replicas"]) == int(fm[system]["n_valid_replicas"])
    for key in ("true", "predicted"):
        assert np.isclose(float(row[key]),
                          np.mean([float(r[key]) for r in rr]),
                          rtol=0, atol=1e-9)
    by_model[model][system] = row
assert len(rows) == 850
assert all(set(v) == set(fm) for v in by_model.values())

systems = sorted(fm)
truth = np.array([float(by_model[models[0]][s]["true"]) for s in systems])
errors = {}
for model in models:
    assert np.allclose(
        truth, [float(by_model[model][s]["true"]) for s in systems],
        rtol=0, atol=1e-9
    )
    errors[model] = np.abs(
        np.array([float(by_model[model][s]["predicted"]) for s in systems]) - truth
    )
    assert np.isfinite(errors[model]).all()
for row in read(run / "system_metrics.csv"):
    assert np.isclose(float(row["mae"]), errors[row["model"]].mean(),
                      rtol=0, atol=1e-9)

# Cada extracción incluye todos los sistemas del grupo seleccionado.
# La métrica mantiene el mismo peso por sistema.
membership = np.array([
    [fm[s]["group_id"] == g for s in systems] for g in group_ids
])
sizes = membership.sum(axis=1)
rng = np.random.default_rng(20261016)
draw = rng.integers(0, len(group_ids), size=(10000, len(group_ids)))
denominator = sizes[draw].sum(axis=1)
boot = {
    m: (membership @ errors[m])[draw].sum(axis=1) / denominator
    for m in models
}
metrics = {
    m: {
        "mae": float(errors[m].mean()),
        "mae_cluster_bootstrap_95": np.quantile(boot[m], [.025, .975]).tolist()
    }
    for m in models
}
comparisons = []
for a, b in [
    ("gru_residual_rmsd_pilot", "ridge_summary"),
    ("gru_residual_rmsd_pilot", "last_value_persistence"),
    ("ridge_summary", "last_value_persistence")
]:
    delta = errors[a] - errors[b]
    comparisons.append({
        "contrast": f"{a} minus {b}",
        "delta_mae_A": float(delta.mean()),
        "paired_cluster_bootstrap_95_A":
            np.quantile(boot[a] - boot[b], [.025, .975]).tolist(),
        "system_wins_for_first": int(np.sum(delta < -1e-12)),
        "system_ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "group_mean_error_wins_for_first":
            int(np.sum((membership @ delta) / sizes < -1e-12))
    })

summary = {
    "systems": 170, "groups": 149, "replicas": 288,
    "bootstrap_draws": 10000, "seed": 20261016,
    "bootstrap_unit": "whole sequence group; equal weight per system",
    "interval_scope": "exploratory percentile; fitted OOF predictions fixed; no retraining",
    "metrics": metrics, "comparisons": comparisons, "selection": selection,
    "input_predictions_sha256": sha(run / "system_predictions.csv"),
    "audit_code_sha256": sha(Path(__file__))
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("ATLAS222_PAIRED_GROUP_AUDIT=PASS")

# Inventario de espacio; todavía no verifica recuperabilidad ni borra.
base = run.parent
inventory = base / "atlas222_source_inventory_20261015_125033/source_inventory.csv"
assert sha(inventory) == "8877f3fcb12e53af21ae609d1b1a1176965f2584fd75ad03a2db21b6ab5ca3ce"
new = [r["system"] for r in read(inventory) if r["scope"] == "new_unassigned"]
assert len(set(new)) == len(new) == 142 and "3txs_C" not in new
raw = base.parent.parent / "raw/atlas/paper2_wave3_v1"
storage = []
for system in sorted(new):
    for path in sorted((raw / system / "unpacked").rglob("*.xtc")):
        assert path.is_file() and not path.is_symlink(), path
        storage.append({
            "system": system, "path": str(path), "bytes": path.stat().st_size
        })
if storage:
    with (out / "unpacked_xtc_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "path", "bytes"])
        writer.writeheader()
        writer.writerows(storage)
print("STORAGE_INVENTORY=" + json.dumps({
    "systems_with_unpacked_xtc": len({r["system"] for r in storage}),
    "xtc_files": len(storage),
    "logical_GiB": sum(r["bytes"] for r in storage) / 2**30,
    "archive_recoverability_checked": False,
    "files_deleted": 0
}))
print(f"OUTPUT={out}")
