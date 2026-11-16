from pathlib import Path
from collections import defaultdict
import csv
import json
import hashlib

base = Path("/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/results/paper2_hito0")
assignment = base / "atlas222_grouped_development_v1/system_assignment.csv"
targets = base / "atlas222_future_rmsd_train_3239158/train_future_rmsd_rows.csv"
out = base / "atlas222_future_rmsd_folds_v1"
seed = 20261016

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def flag(value):
    assert value.lower() in ("true", "false", "1", "0"), value
    return value.lower() in ("true", "1")

assert sha(assignment) == "79b983deb8c1f857fb7a22ce05dcdd107dbb1ff4fd37ee2971c046e7564e2149"
assert sha(targets) == "3caa09b9171ea5d12dccc3d9b9f44b0e2eff622083c849e31f623c876a53c1d9"
assigned = {r["system_id"]: r for r in read(assignment)}
rows = read(targets)
assert len(rows) == 612
assert len({r["sample_id"] for r in rows}) == 612

valid_by_system = defaultdict(list)
for row in rows:
    system = row["system_id"]
    assert assigned[system]["split"] == "train"
    assert row["group_id"] == assigned[system]["group_id"]
    if flag(row["future_valid"]):
        valid_by_system[system].append(row["sample_id"])

members = defaultdict(list)
for system in sorted(valid_by_system):
    members[assigned[system]["group_id"]].append(system)

assert len(valid_by_system) == 170
assert len(members) == 149
assert sum(map(len, valid_by_system.values())) == 288

# Equilibrar el número de sistemas manteniendo intactos los grupos.
# Los valores de los targets no intervienen en la asignación.
ordered = sorted(members, key=lambda g: (
    -len(members[g]),
    hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(),
    g
))
fold_groups = [[] for _ in range(5)]
loads = [0] * 5
for group in ordered:
    fold = min(range(5), key=lambda i: (loads[i], len(fold_groups[i]), i))
    fold_groups[fold].append(group)
    loads[fold] += len(members[group])

records = []
summary_folds = []
for fold, groups in enumerate(fold_groups):
    systems = sorted(s for g in groups for s in members[g])
    for system in systems:
        records.append({
            "system_id": system,
            "group_id": assigned[system]["group_id"],
            "outer_fold": fold,
            "n_valid_replicas": len(valid_by_system[system]),
        })
    summary_folds.append({
        "fold": fold,
        "eval_systems": len(systems),
        "eval_groups": len(groups),
        "eval_replicas": sum(len(valid_by_system[s]) for s in systems)
    })

assert len(records) == 170
assert len({r["system_id"] for r in records}) == 170
assert all(
    len({r["outer_fold"] for r in records if r["group_id"] == g}) == 1
    for g in members
)

out.mkdir(parents=True, exist_ok=False)
path = out / "outer_folds.csv"
with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "system_id", "group_id", "outer_fold", "n_valid_replicas"
    ])
    writer.writeheader()
    writer.writerows(sorted(records, key=lambda r: r["system_id"]))

summary = {
    "seed": seed,
    "systems": 170,
    "groups": 149,
    "valid_replicas": 288,
    "folds": summary_folds,
    "group_overlap_between_folds": False,
    "target_values_used_for_assignment": False,
    "inner_selection_must_also_respect_groups": True,
    "assignment_sha256": sha(assignment),
    "targets_sha256": sha(targets),
    "folds_sha256": sha(path),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("ATLAS222_OUTER_FOLDS=PASS")
print(f"OUTPUT={out}")
