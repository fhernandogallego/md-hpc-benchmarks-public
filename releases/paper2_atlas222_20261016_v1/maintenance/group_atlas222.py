from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import hashlib

work = Path("/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks")
base = work / "results/paper2_hito0"
screen = base / "atlas222_similarity_3239118"
seq = base / "atlas222_sequences_20261016_121738"
split = base / "atlas80_split_v1/protein_split.csv"
out = base / "atlas222_grouped_development_v1"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path, delimiter=","):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))

expected = "d31ef5c7d5fcea49e4f6414ea8bb70254ee5c63008b8c02707ab462c31e24bee"
assert sha(split) == expected, "El split histórico ha cambiado"
historic = {r["system_id"]: r["split"] for r in read(split)}
sources = read(screen / "source_inventory.csv")
systems = sorted(r["system"] for r in sources)
assert len(systems) == len(set(systems)) == 222
assert len(historic) == 80 and set(historic) <= set(systems)

parent = {s: s for s in systems}
edges = []

def root(s):
    while parent[s] != s:
        parent[s] = parent[parent[s]]
        s = parent[s]
    return s

def join(a, b, reason):
    assert a in parent and b in parent
    ra, rb = root(a), root(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)
    edges.append({"a": a, "b": b, "reason": reason})

for row in read(screen / "candidate_pairs.tsv", "\t"):
    join(row["system_a"], row["system_b"], "blast_30pct_80pct_both")

sequence_summary = json.loads((seq / "summary.json").read_text())
assert not sequence_summary["issues"]
for group in sequence_summary["exact_duplicate_groups"]:
    members = group["systems"]
    for s in members[1:]:
        join(members[0], s, "identical_observed_sequence")

by_pdb = defaultdict(list)
for s in systems:
    by_pdb[s.split("_", 1)[0]].append(s)
for members in by_pdb.values():
    for s in members[1:]:
        join(members[0], s, "same_pdb")

groups = defaultdict(list)
for s in systems:
    groups[root(s)].append(s)

assignment = {s: historic.get(s, "train") for s in systems}
conflicts = []
for members in groups.values():
    labels = {assignment[s] for s in members}
    if len(labels) > 1:
        conflicts.append({"systems": members, "splits": sorted(labels)})
if conflicts:
    print(json.dumps({"GROUP_CONFLICTS": conflicts}, indent=2))
    raise SystemExit("No se escribe ninguna asignación: hay conflictos")

# Se escribe en un directorio nuevo; nunca sobre el split histórico.
out.mkdir(parents=True, exist_ok=False)
rows = []
for members in sorted(groups.values(), key=lambda m: m[0]):
    group_id = "grp_" + members[0]
    for s in members:
        rows.append({
            "system_id": s,
            "split": assignment[s],
            "group_id": group_id,
            "origin": "atlas80" if s in historic else "wave3",
        })

with (out / "system_assignment.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["system_id", "split", "group_id", "origin"]
    )
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: r["system_id"]))

(out / "group_edges.json").write_text(json.dumps(edges, indent=2) + "\n")
summary = {
    "systems": len(systems),
    "systems_by_split": dict(Counter(assignment.values())),
    "groups_total": len(groups),
    "groups_by_split": dict(Counter(
        assignment[members[0]] for members in groups.values()
    )),
    "non_singleton_groups": [
        members for members in groups.values() if len(members) > 1
    ],
    "group_conflicts": conflicts,
    "new_system_assignment": "train",
    "atlas80_split_sha256": sha(split),
    "blast_pairs_sha256": sha(screen / "candidate_pairs.tsv"),
    "sequence_fasta_sha256": sha(seq / "sequences.fasta"),
    "assignment_sha256": sha(out / "system_assignment.csv"),
    "cv_folds_created": False,
    "targets": "future_RMSD_pending",
}
assert sha(split) == expected
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("ATLAS222_GROUPED_ASSIGNMENT=PASS")
print(f"OUTPUT={out}")
