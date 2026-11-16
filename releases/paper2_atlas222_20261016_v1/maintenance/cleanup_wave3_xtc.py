from pathlib import Path
import csv, json, hashlib, os, zipfile, subprocess

work = Path("/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks").resolve(strict=True)
base = work / "results/paper2_hito0"
raw = (work / "raw/atlas/paper2_wave3_v1").resolve(strict=True)
run = base / "atlas222_grouped_cv_20261016_130332"
audit = run / "paired_group_audit_v1"
out = base / "storage_audits" / ("wave3_cleanup_" + os.environ["SLURM_JOB_ID"])
out.mkdir(parents=True, exist_ok=False)

def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()

def sha(path):
    with path.open("rb") as f:
        return digest(f)

def stamp(path):
    st = path.stat(follow_symlinks=False)
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]

inventory = base / "atlas222_source_inventory_20261015_125033/source_inventory.csv"
assert sha(inventory) == "8877f3fcb12e53af21ae609d1b1a1176965f2584fd75ad03a2db21b6ab5ca3ce"
new = {r["system"]: r for r in read(inventory) if r["scope"] == "new_unassigned"}
assert len(new) == 142 and "3txs_C" not in new

report = json.loads((audit / "summary.json").read_text())
assert sha(run / "system_predictions.csv") == report["input_predictions_sha256"]
assert report["systems"] == 170 and report["groups"] == 149
future = base / "atlas222_future_rmsd_train_3239158"
assert sha(future / "train_future_rmsd_rows.csv") == "3caa09b9171ea5d12dccc3d9b9f44b0e2eff622083c849e31f623c876a53c1d9"
source_hashes = json.loads((future / "source_hashes.json").read_text())
metadata = json.loads((
    base / "atlas222_assembly_20261015_v1/dataset/multisystem_dataset_metadata.json"
).read_text())
packages = {s["system"]: s for s in metadata["sources"]}

queued = subprocess.run(
    ["squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%j"],
    check=True, capture_output=True, text=True
).stdout.splitlines()
assert not any(
    line.split("|", 1)[-1].strip().endswith("_" + system)
    for line in queued for system in new
), "Hay pipelines activos de esta tanda"

inventory_rows = read(audit / "unpacked_xtc_inventory.csv")
assert len(inventory_rows) == 852
expected_paths = {}
for row in inventory_rows:
    p = Path(row["path"]).resolve(strict=True)
    assert row["system"] in new and p not in expected_paths
    p.relative_to(raw / row["system"] / "unpacked")
    expected_paths[p] = row

plan, archives = [], []
for system in sorted(new):
    directory = raw / system
    assert not directory.is_symlink()
    assert not (directory / "unpacked").is_symlink()

    source = source_hashes[system]
    assert sha(Path(source["master_csv"])) == source["master_sha256"]
    for value in source["replicas"].values():
        assert sha(Path(value["path"])) == value["sha256"]

    release = Path(new[system]["release"]).resolve(strict=True)
    package = release / "prefix_sequences_raw.npz"
    assert sha(package) == packages[system]["npz_sha256"]
    pipeline = next(
        p for p in release.parents if (p / "PIPELINE_STATUS.txt").is_file()
    )
    status = dict(
        line.split("=", 1)
        for line in (pipeline / "PIPELINE_STATUS.txt").read_text().splitlines()
        if "=" in line
    )
    assert status["pipeline_status"] == "PASS" and status["system"] == system
    assert status["dataset_group"] == "paper2_wave3_v1"
    assert all(status[k] == "true" for k in ("package_audit", "cross_file_audit"))

    files = sorted((directory / "unpacked").rglob("*.xtc"))
    allowed = (
        {f"{system}_R{r}.xtc" for r in (1, 2, 3)}
        | {f"{system}_prod_R{r}_fit.xtc" for r in (1, 2, 3)}
    )
    assert len(files) == 6 and {p.name for p in files} == allowed
    by_name = {}
    for p in files:
        assert not p.is_symlink() and p.resolve(strict=True) == p
        assert p in expected_paths
        assert p.stat().st_size == int(expected_paths[p]["bytes"])
        by_name[p.name] = p

    recovered = {}
    for kind in ("analysis", "protein"):
        archive = directory / "archives" / f"{system}_{kind}.zip"
        assert not archive.is_symlink() and archive.resolve(strict=True) == archive
        before = stamp(archive)
        archive_sha = sha(archive)
        with zipfile.ZipFile(archive) as z:
            names = [info.filename for info in z.infolist()]
            assert len(names) == len(set(names)), "Miembros ZIP duplicados"
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                with z.open(info) as stream:
                    member_sha = digest(stream)  # Lectura completa comprueba CRC.
                if name in by_name:
                    assert name not in recovered, f"Miembro ambiguo: {name}"
                    p = by_name[name]
                    file_before = stamp(p)
                    file_sha = sha(p)
                    assert stamp(p) == file_before
                    assert info.file_size == p.stat().st_size
                    assert member_sha == file_sha
                    recovered[name] = {
                        "system": system, "path": str(p), "bytes": info.file_size,
                        "sha256": file_sha, "stamp": file_before,
                        "archive": str(archive), "member": info.filename
                    }
        assert stamp(archive) == before
        archives.append({
            "path": str(archive), "sha256": archive_sha, "stamp": before
        })
    assert set(recovered) == allowed
    plan.extend(recovered.values())
    print(f"RECOVERABILITY_PASS {system}", flush=True)

assert len(plan) == len(expected_paths) == 852
assert {Path(r["path"]) for r in plan} == set(expected_paths)
(out / "verified_plan.json").write_text(
    json.dumps({"files": plan, "archives": archives}, indent=2) + "\n"
)

# Ningún borrado ocurre antes de verificar toda la tanda.
for row in archives + plan:
    path = Path(row["path"])
    assert not path.is_symlink() and stamp(path) == row["stamp"], path
print("ALL_PRE_DELETE_CHECKS=PASS", flush=True)

with (out / "deletion.jsonl").open("x") as log:
    for row in plan:
        path = Path(row["path"])
        assert not path.is_symlink() and stamp(path) == row["stamp"]
        log.write(json.dumps({"event": "before_unlink", **row}) + "\n")
        log.flush()
        os.fsync(log.fileno())
        path.unlink()
        log.write(json.dumps({
            "event": "deleted", "path": str(path), "bytes": row["bytes"]
        }) + "\n")
        log.flush()
        os.fsync(log.fileno())

summary = {
    "systems": 142,
    "deleted_xtc_files": len(plan),
    "freed_logical_GiB": sum(r["bytes"] for r in plan) / 2**30,
    "zip_files_retained": len(archives),
    "excluded_system": "3txs_C"
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("WAVE3_XTC_CLEANUP=PASS")
print(f"OUTPUT={out}")
