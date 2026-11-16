#!/usr/bin/env python3
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

WORK = Path("/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks")
HOME = Path("/home/uva_dma_2/uva_dma_2_1")

RUN = WORK / "results/paper2_automation/auto_20261030_091302"
PILOT = WORK / "results/pid_atlas_poc"

OUT = WORK / "maintenance/earlystop" / (
    "pendientes_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)
OUT.mkdir(parents=True, exist_ok=False)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(name, data, columns):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(data)


def files_limited(root, depth=4):
    if not root.is_dir():
        return

    excluded = {
        ".git", "__pycache__", "series",
        "resultados_atlas_05", "resultados_atlas_10",
        "resultados_atlas_20", "test", "tests",
        "predictions", "full_train", "venv", ".venv"
    }

    for base, dirs, filenames in os.walk(root):
        level = len(Path(base).relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if d not in excluded)

        if level >= depth:
            dirs[:] = []

        for name in sorted(filenames):
            p = Path(base) / name
            if not p.is_symlink():
                yield p


def check_pilot():
    print("\n========== 1. PILOTO 500 ==========")

    roots = [
        PILOT / "full_v2_3262025",
        PILOT / "audit_v2_500_3268243",
        PILOT / "job_3258863"
    ]

    pattern = re.compile(
        r"rmsd|radius.of.gyration|select_atoms|"
        r"name CA|backbone|align|superpos|"
        r"weight|reference|atom.selection|"
        r"gyration",
        re.I
    )

    for root in roots:
        print("\nDIRECTORIO:", root)
        if not root.is_dir():
            print("NO DISPONIBLE")
            continue

        candidates = [
            p for p in files_limited(root)
            if p.suffix.lower() in (
                ".py", ".sh", ".sbatch",
                ".json", ".yaml", ".yml", ".md"
            )
        ]

        print("ARCHIVOS DE CODIGO/METADATOS:", len(candidates))

        for p in candidates[:100]:
            print("ARCHIVO:", p.relative_to(root), "bytes=", p.stat().st_size)

        for p in candidates:
            if p.suffix.lower() not in (".py", ".sh", ".sbatch"):
                continue
            if p.stat().st_size > 1000000:
                continue

            text = p.read_text(errors="replace").splitlines()
            hits = [i for i, line in enumerate(text) if pattern.search(line)]

            if not hits:
                continue

            print("\nCODIGO_RELEVANTE:", p)
            print("SHA256:", sha(p))

            indices = sorted({
                j
                for i in hits[:25]
                for j in range(max(0, i - 1), min(len(text), i + 3))
            })

            for j in indices[:100]:
                print("%04d: %s" % (j + 1, text[j][:190]))

        for name in ("cache_manifest.json", "input_manifest.json"):
            p = root / name

            if not p.is_file():
                continue

            try:
                obj = json.loads(p.read_text())
                print("\nMANIFIESTO:", p)
                print("SHA256:", sha(p))

                if isinstance(obj, dict):
                    print("CLAVES:", sorted(obj.keys()))

                    # Solo procedencia y configuracion.
                    allowed = re.compile(
                        r"code|script|selection|weight|"
                        r"align|reference|commit|source|method",
                        re.I
                    )

                    for k, v in obj.items():
                        if allowed.search(k) and isinstance(
                            v, (str, int, float, bool)
                        ):
                            print(k, "=", str(v)[:250])

            except (ValueError, OSError) as e:
                print("ERROR_MANIFIESTO:", str(e))


def check_core():
    print("\n========== 2. CORE RMSD Q30 ==========")

    cohort = RUN / "cohort"

    assignment_file = cohort / "system_assignment.csv"
    inventory_file = cohort / "source_inventory.csv"
    future_file = cohort / "train_future_rmsd_rows.csv"

    for p in (assignment_file, inventory_file, future_file):
        if not p.is_file():
            raise RuntimeError("Falta archivo necesario: " + str(p))

    assignments = rows(assignment_file)

    train = {
        r["system_id"]
        for r in assignments
        if r["split"].lower() == "train"
    }

    inventory = {
        r["system"]: r
        for r in rows(inventory_file)
    }

    future = rows(future_file)

    labeled = {
        r["system_id"]
        for r in future
        if (
            r["system_id"] in train
            and r["future_valid"].strip().lower() == "true"
        )
    }

    print("TRAIN_SISTEMAS:", len(train))
    print("RMSD_FUTURO_SISTEMAS:", len(labeled))

    records = []
    sample_schema = None
    sample_metadata = None
    errors = []

    for system in sorted(train):
        source = inventory.get(system, {})
        master_text = source.get("master_targets_csv", "")

        if not master_text:
            errors.append((system, "SIN_RUTA_MASTER"))
            continue

        master = Path(master_text)
        system_root = master.parent.parent

        core = system_root / "core_rmsd" / "core_rmsd_summary.csv"
        metadata = system_root / "core_rmsd" / "core_rmsd.json"

        record = {
            "system": system,
            "scope_757": system in labeled,
            "core_summary_exists": core.is_file(),
            "q30_replica_prefix_replicas": 0,
            "q30_series_files_exist": 0,
            "core_summary_path": str(core)
        }

        if core.is_file():
            try:
                with core.open(
                    newline="", encoding="utf-8-sig"
                ) as f:
                    reader = csv.DictReader(f)

                    if sample_schema is None:
                        sample_schema = reader.fieldnames

                    replicas = set()
                    series_ok = set()

                    for row in reader:
                        try:
                            quantile = float(row.get("quantile", "nan"))
                        except ValueError:
                            continue

                        if abs(quantile - 0.30) > 1e-8:
                            continue

                        if row.get("variant") != "replica_prefix":
                            continue

                        replica = row.get("replica", "")
                        replicas.add(replica)

                        series_path = row.get("timeseries_csv", "")

                        if series_path and Path(series_path).is_file():
                            series_ok.add(replica)

                    record["q30_replica_prefix_replicas"] = len(replicas)
                    record["q30_series_files_exist"] = len(series_ok)

                if metadata.is_file() and sample_metadata is None:
                    sample_metadata = (
                        str(metadata),
                        sha(metadata),
                        json.loads(metadata.read_text())
                    )

            except (OSError, ValueError, csv.Error) as e:
                errors.append((system, str(e)[:150]))

        records.append(record)

    columns = [
        "system", "scope_757", "core_summary_exists",
        "q30_replica_prefix_replicas",
        "q30_series_files_exist", "core_summary_path"
    ]

    write_csv("core_q30_disponibilidad.csv", records, columns)

    for label, subset in (
        ("TRAIN_924", [r for r in records]),
        ("RMSD_757", [r for r in records if r["scope_757"]])
    ):
        print("\nUNIVERSO:", label)
        print("SISTEMAS:", len(subset))
        print("CON_RESUMEN_CORE:", sum(
            r["core_summary_exists"] for r in subset
        ))
        print("SISTEMAS_CON_Q30:", sum(
            r["q30_replica_prefix_replicas"] > 0 for r in subset
        ))
        print("SISTEMAS_Q30_TRES_REPLICAS:", sum(
            r["q30_replica_prefix_replicas"] == 3 for r in subset
        ))
        print("REPLICAS_Q30_CON_RESUMEN:", sum(
            r["q30_replica_prefix_replicas"] for r in subset
        ))
        print("REPLICAS_Q30_CON_SERIE:", sum(
            r["q30_series_files_exist"] for r in subset
        ))

    print("\nCOLUMNAS_CORE_SUMMARY:", sample_schema)

    if sample_metadata:
        path, digest, obj = sample_metadata
        print("EJEMPLO_METADATA_CORE:", path)
        print("SHA256:", digest)

        for k in (
            "selection_atom", "rmsd_atoms",
            "alignment_atoms", "reference",
            "weights", "leakage_policy"
        ):
            if k in obj:
                print(k, "=", obj[k])

    print("\nINCIDENCIAS_CORE:", len(errors))

    write_csv(
        "incidencias_core.csv",
        [{"system": s, "motivo": e} for s, e in errors],
        ["system", "motivo"]
    )

    print(
        "NOTA: disponibilidad de una serie q30 NO demuestra "
        "validez estadistica del objetivo."
    )

    # Comprobar relacion entre validaciones RMSD originales
    # y futuras, sin abrir predicciones ni resultados de test.

    flags = Counter(
        (
            r.get("original_valid", "").lower(),
            r.get("future_valid", "").lower(),
            r.get("common_valid", "").lower()
        )
        for r in future
        if r["system_id"] in train
    )

    print("\nCRUCE_VALIDEZ_RMSD_ORIGINAL_FUTURO:")

    for state, n in sorted(flags.items()):
        print(state, n)


def check_provenance():
    print("\n========== 3. PARTICIONES Y COMMITS ==========")

    p = RUN / "cohort/system_assignment.csv"

    print("MANIFIESTO_ATLAS942:", p)
    print("SHA256:", sha(p))

    assignments = rows(p)
    counts = Counter(r["split"] for r in assignments)

    for split, n in sorted(counts.items()):
        groups = {
            r["group_id"]
            for r in assignments if r["split"] == split
        }
        print(
            "PARTICION:", split,
            "SISTEMAS:", n,
            "GRUPOS:", len(groups)
        )

    memberships = {}
    for r in assignments:
        memberships.setdefault(r["group_id"], set()).add(r["split"])

    conflicts = sum(len(v) > 1 for v in memberships.values())
    print("GRUPOS_CRUZADOS:", conflicts)

    if conflicts:
        raise RuntimeError("Hay grupos compartidos entre particiones")

    for p in (
        RUN / "tool_manifest.json",
        RUN / "pipeline_manifest.json",
        RUN / "config.json"
    ):
        if p.is_file():
            print("PROCEDENCIA:", p, "SHA256:", sha(p))

            obj = json.loads(p.read_text())

            if isinstance(obj, dict):
                print("CLAVES:", list(obj)[:30])

    print("\n========== DIAGNOSTICO GIT ==========")
    print("GIT_BINARIO:", shutil.which("git"))

    for root in (
        HOME / "paper2-auto-v3",
        HOME / "paper2-atlas942-publication"
    ):
        print("\nREPOSITORIO:", root)
        print("EXISTE:", root.exists())
        print("GIT_METADATA_EXISTE:", (root / ".git").exists())

        if not shutil.which("git") or not root.exists():
            continue

        for args in (
            ["rev-parse", "--show-toplevel"],
            ["rev-parse", "HEAD"],
            ["branch", "--show-current"]
        ):
            try:
                result = subprocess.run(
                    ["git", "-C", str(root)] + args,
                    capture_output=True,
                    text=True,
                    timeout=15
                )

                print("COMANDO:", " ".join(args))
                print("EXIT:", result.returncode)
                print("STDOUT:", result.stdout.strip()[:300])
                print("STDERR:", result.stderr.strip()[:500])

            except (OSError, subprocess.TimeoutExpired) as e:
                print("ERROR_GIT:", e)

    print("\n========== CANDIDATOS CALIBRACION ==========")

    roots = [
        WORK / "maintenance",
        RUN / "cohort",
        PILOT,
        HOME / "paper2-auto-v3",
        HOME / "paper2-atlas942-publication"
    ]

    candidates = []
    pattern = re.compile(
        r"calibr|development|partition_manifest|"
        r"split_manifest|holdout_manifest",
        re.I
    )

    for root in roots:
        for p in files_limited(root, depth=6):
            if pattern.search(p.name):
                candidates.append(str(p))

    for path in sorted(set(candidates))[:100]:
        print("CANDIDATO:", path)

    if not candidates:
        print("NO_ENCONTRADO_EN_RUTAS_INSPECCIONADAS")

    print(
        "NOTA: no interpretar validation como calibration "
        "sin un manifiesto o protocolo que lo acredite."
    )


def main():
    print("AUDITORIA_EARLYSTOP_PENDIENTES")
    print("SALIDA:", OUT)
    print("MODO: SOLO LECTURA DE FUENTES")
    print("TEST: NO EVALUADO")

    check_pilot()
    check_core()
    check_provenance()

    print("\nAUDITORIA_COMPLETADA")
    print("DIRECTORIO_RESULTADOS=" + str(OUT))


if __name__ == "__main__":
    main()
