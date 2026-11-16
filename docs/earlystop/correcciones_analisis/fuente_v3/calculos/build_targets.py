#!/usr/bin/env python3
"""Calculation 3 (part A): future-only targets for several observables.

Generalises future_builder.py (RMSD only) to every observable the pipeline extracts.
For each replica and observable, the target is the mean over frames with t > 20 ns and
t >= t0 (t0 = observable-specific equilibration; core is detected on its own series). It is
valid when that window lasts at least 20 ns and has N_eff >= 10 (pymbar statistical
inefficiency), exactly as for RMSD in the paper.

STEP 1 - inspect the inventory (does not write anything):
  python c3_build_targets.py --inventory inventory.csv --inspect
  It prints the inventory columns, the target names in the master table and the columns of
  each per-replica series file for the first system. Use it to fill targets_config.json.

STEP 2 - build:
  python c3_build_targets.py --inventory inventory.csv --config targets_config.json \
      --out resultados/c3_targets.csv

inventory.csv: one row per system, with a column "system" (or "system_id"), a column
"master_targets_csv" and one column per series file and replica, e.g. basic_R1_csv,
basic_R2_csv, basic_R3_csv (and others such as sasa_R1_csv if SASA/DSSP live elsewhere).
It is the same kind of file that future_builder.py read for ATLAS222.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import read_csv, write_csv, write_json, sha

PREFIX_NS = 20.0


def system_key(row: dict) -> str:
    return row.get("system") or row.get("system_id")


def future_target(times, values, t0_ns, inefficiency):
    ok = np.isfinite(times) & np.isfinite(values)
    times, values = times[ok], values[ok]
    start = int(np.searchsorted(times, max(PREFIX_NS + 1e-8, t0_ns - 1e-7), side="left"))
    while start < len(times) and times[start] <= PREFIX_NS + 1e-8:
        start += 1
    if start >= len(times) - 1:
        return {"duration_ns": 0.0, "n_frames": 0, "g": math.nan, "neff": 0.0, "valid": False, "value": math.nan}
    seg = values[start:]
    duration = float(times[-1] - times[start])
    if np.ptp(seg) < 1e-12:
        # Constant window: PyMBAR's sample covariance is zero and g is undefined. g = 1 is a
        # computational convention (applied on 10/11/2026), not evidence of independence.
        g = 1.0
    else:
        g = max(float(inefficiency(seg, fast=False)), 1.0)
    neff = len(seg) / g
    valid = duration >= 20.0 - 1e-8 and neff >= 10.0 - 1e-8
    return {"duration_ns": duration, "n_frames": len(seg), "g": g, "neff": neff, "valid": bool(valid),
            "value": float(seg.mean()) if valid else math.nan}


def inspect(inventory: Path) -> None:
    rows = read_csv(inventory)
    print("inventory rows:", len(rows))
    print("inventory columns:", list(rows[0]))
    first = rows[0]
    master = Path(first["master_targets_csv"])
    m = pd.read_csv(master)
    print("\nmaster table:", master)
    print("  columns:", list(m.columns))
    if "target_name" in m.columns:
        print("  target_name values:", sorted(m["target_name"].astype(str).unique()))
    for col, path in first.items():
        if col.endswith("_csv") and col != "master_targets_csv" and path and Path(path).is_file():
            head = pd.read_csv(path, nrows=3)
            print(f"\n{col}: {path}")
            print("  columns:", list(head.columns))
            print("  first time values:", head.iloc[:, 0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--constant-policy", choices=["exclude", "include"], default="exclude")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.inspect:
        inspect(args.inventory)
        return
    if not (args.config and args.out):
        ap.error("--config and --out are required to build")
    if args.out.exists():
        raise FileExistsError(args.out)
    import pymbar
    from pymbar import timeseries

    config = json.loads(args.config.read_text())
    inventory = read_csv(args.inventory)
    records, problems = [], []
    for n, row in enumerate(inventory):
        system = system_key(row)
        master = pd.read_csv(row["master_targets_csv"])
        master = master[master["system"].astype(str) == system] if "system" in master.columns else master
        cache = {}
        for target, spec in config["targets"].items():
            for replica in (1, 2, 3):
                path = row.get(spec["series_file"].format(r=replica))
                if not path:
                    problems.append(f"{system} {target} R{replica}: no column {spec['series_file'].format(r=replica)}")
                    continue
                if path not in cache:
                    cache[path] = pd.read_csv(path)
                frame = cache[path]
                if spec["column"] not in frame.columns:
                    problems.append(f"{system} {target}: column {spec['column']} missing in {path}")
                    continue
                raw_values = frame[spec["column"]].to_numpy(float)
                raw_times = frame[spec.get("time_column", "time_ns")].to_numpy(float)
                if not (np.isfinite(raw_values).all() and np.isfinite(raw_times).all()):
                    raise ValueError(f"Nonfinite series: {system} {target} R{replica}")
                if not np.all(np.diff(raw_times) > 0):
                    raise ValueError(f"Non-increasing times: {system} {target} R{replica}")
                t0 = None
                name = spec.get("master_target_name")
                if name is not None:
                    sel = master[(master["target_name"].astype(str) == name) & (master["replica"].astype(int) == replica)]
                    if len(sel) == 1 and np.isfinite(float(sel["t0_ns"].iloc[0])):
                        t0 = float(sel["t0_ns"].iloc[0])
                if spec.get("equilibration") == "detect":
                    if np.ptp(raw_values) < 1e-12:
                        t0 = float(raw_times[0])
                    else:
                        first, _, _ = timeseries.detect_equilibration(raw_values, fast=False)
                        t0 = float(raw_times[int(first)])
                if t0 is None:
                    raise ValueError(f"Missing equilibration metadata: {system} {target} R{replica}")
                res = future_target(frame[spec.get("time_column", "time_ns")].to_numpy(float),
                                    frame[spec["column"]].to_numpy(float), t0, timeseries.statistical_inefficiency)
                window = raw_values[(raw_times > PREFIX_NS + 1e-8) & (raw_times >= t0 - 1e-7)]
                constant = len(window) > 1 and np.ptp(window) < 1e-12
                if constant and args.constant_policy == "exclude":
                    res["valid"] = False
                    res["value"] = math.nan
                records.append({"constant_window": bool(constant), "constant_policy": args.constant_policy,"sample_id": f"{system}_R{replica}", "system_id": system, "replica": replica,
                                "target": target, "t0_ns": t0, "future_valid": res["valid"],
                                "future_target": res["value"], "future_duration_ns": res["duration_ns"],
                                "future_neff": res["neff"], "future_g": res["g"]})
        if n < 3 or (n + 1) % 100 == 0:
            print(f"[{n + 1}/{len(inventory)}] {system}", flush=True)
    if not records:
        raise SystemExit("No targets built:\n" + "\n".join(problems[:20]))
    write_csv(args.out, records)
    summary = {}
    for target in config["targets"]:
        rows = [r for r in records if r["target"] == target]
        valid = [r for r in rows if r["future_valid"]]
        summary[target] = {"replicas": len(rows), "valid_replicas": len(valid),
                           "valid_systems": len({r["system_id"] for r in valid})}
        print(f"{target:18s} valid replicas {len(valid)}/{len(rows)}  systems {summary[target]['valid_systems']}")
    write_json(args.out.with_suffix(".summary.json"), {
        "calculation": "revised_build_targets", "constant_policy": args.constant_policy, "inventory_sha256": sha(args.inventory),
        "config": config, "config_sha256": sha(args.config),
        "script_sha256": sha(Path(__file__)), "targets_sha256": sha(args.out),
        "numpy_version": np.__version__, "pymbar_version": getattr(pymbar, "__version__", "unknown"),
        "per_target": summary, "problems": problems[:200], "n_problems": len(problems)})
    if problems:
        print(f"WARNING {len(problems)} problems; first ones:")
        for p in problems[:10]:
            print("  ", p)
        raise SystemExit('C3_TARGETS=PARTIAL: inspect missing series/columns before fitting')
    print("C3_TARGETS=PASS")


if __name__ == "__main__":
    main()
