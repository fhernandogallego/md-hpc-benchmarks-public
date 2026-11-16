"""Small, dependency-free I/O and integrity helpers for the SCAYLE controller."""
from __future__ import annotations

import contextlib
import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import uuid
import zipfile

SYSTEM = re.compile(r"^[0-9A-Za-z]{4}_[0-9A-Za-z]+$")
GROUP = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_.-]*$")
FAILED = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
          "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED", "SPECIAL_EXIT"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def sha(path):
    with Path(path).open("rb") as f:
        return digest(f)


def stamp(path):
    s = Path(path).stat(follow_symlinks=False)
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def read_csv(path, delimiter=","):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path, rows, fields=None):
    require(bool(rows) or fields is not None, f"Empty CSV without columns: {path}")
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    with temp.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def execute(command, log=None, cwd=None, env=None):
    if log:
        with Path(log).open("a") as f:
            f.write("RUN " + shlex.join(map(str, command)) + "\n")
            f.flush()
            subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        return ""
    try:
        return subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                              capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        # Preserve the actual scheduler error in the private controller log.
        if error.stderr:
            print(error.stderr.rstrip(), file=sys.stderr, flush=True)
        raise


def checked_members(z):
    names = set()
    for info in z.infolist():
        path = Path(info.filename)
        require(info.filename not in names, f"Duplicate ZIP member: {info.filename}")
        names.add(info.filename)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename,
                f"Unsafe ZIP member: {info.filename}")
        mode = info.external_attr >> 16
        require(not stat.S_ISLNK(mode), f"ZIP symlink forbidden: {info.filename}")
        require(not info.flag_bits & 1, f"Encrypted ZIP member: {info.filename}")
        yield info


def inspect_archive(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing/non-regular archive: {path}")
    before = stamp(path)
    with zipfile.ZipFile(path) as z:
        members = list(checked_members(z))
        for info in members:
            if not info.is_dir():
                with z.open(info) as f:
                    digest(f)  # Complete read checks CRC, too.
    result = {"path": str(path.resolve()), "sha256": sha(path), "stamp": before,
              "unpacked_bytes": sum(i.file_size for i in members),
              "members": len(members)}
    require(stamp(path) == before, f"Archive changed while verifying: {path}")
    return result


def unpack_archive(path, destination, verified):
    path, destination = Path(path), Path(destination)
    require(stamp(path) == verified["stamp"], f"Changed archive: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        members = list(checked_members(z))
        if destination.exists():
            require(not destination.is_symlink(), f"Symlink destination: {destination}")
            for info in members:
                if not info.is_dir():
                    p = destination / info.filename
                    require(p.is_file() and not p.is_symlink(), f"Missing existing member: {p}")
                    require(p.resolve().is_relative_to(destination.resolve()), f"Unsafe destination: {p}")
                    with z.open(info) as f:
                        require(digest(f) == sha(p), f"Existing unpacked file differs: {p}")
            return
        require(shutil.disk_usage(destination.parent).free >= verified["unpacked_bytes"] + 2**30,
                f"Insufficient free space to unpack {path}")
        temp = destination.with_name(destination.name + ".partial_" + uuid.uuid4().hex)
        temp.mkdir()
        try:
            for info in members:
                target = temp / info.filename
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, 4 * 1024 * 1024)
            require(stamp(path) == verified["stamp"], f"Archive changed during extraction: {path}")
            temp.rename(destination)
        except BaseException:
            # Keep partial evidence; never replace an existing destination.
            raise


def status_dict(path):
    return dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)


def job_ids(path):
    return {k: v for k, v in status_dict(path).items() if re.fullmatch(r"\d+", v)}


def slurm_states(ids):
    ids = list(map(str, ids))
    if not ids:
        return {}, set()
    # A finished ID may make `squeue -j ID` fail before sacct is queried.
    # Query our active jobs and filter locally. Scheduler/connectivity errors
    # still propagate; an absent active ID is never evidence of completion.
    requested = {identifier.split("_", 1)[0] for identifier in ids}
    queued = execute(["squeue", "-h", "-u", os.environ.get("USER") or getpass.getuser(), "-o", "%i|%T"])
    active = {line.split("|", 1)[0].split("_", 1)[0] for line in queued.splitlines()
              if line.strip() and line.split("|", 1)[0].split("_", 1)[0] in requested}
    text = execute(["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
                    "--format=JobID%64,State%40,ExitCode"])
    result = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3 and parts[0] and "." not in parts[0]:
            result[parts[0].strip()] = (parts[1].split()[0].rstrip("+"), parts[2].strip())
    return result, active


def jobs_outcome(ids, states, active):
    if any(str(i) in active for i in ids):
        return "ACTIVE"
    terminal = []
    for identifier in map(str, ids):
        matches = [v for k, v in states.items() if k == identifier or k.startswith(identifier + "_")]
        if not matches:
            return "UNKNOWN"  # Accounting lag: never infer completion or resubmit.
        terminal.extend(matches)
    if any(state in FAILED or (state == "COMPLETED" and code != "0:0")
           for state, code in terminal):
        return "FAILED"
    return "PASS" if all(s == "COMPLETED" and c == "0:0" for s, c in terminal) else "UNKNOWN"


def fresh_directory(path):
    path = Path(path)
    if path.exists():
        backup = path.parent / "attempts" / (path.name + "_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        backup.parent.mkdir(parents=True, exist_ok=True)
        path.rename(backup)
    path.mkdir(parents=True)
    return path
