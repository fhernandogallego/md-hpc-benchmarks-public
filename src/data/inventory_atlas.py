#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TOPOLOGY_EXTENSIONS = {
    ".pdb",
    ".gro",
    ".tpr",
    ".psf",
    ".prmtop",
    ".top",
}

TRAJECTORY_EXTENSIONS = {
    ".xtc",
    ".trr",
    ".dcd",
    ".nc",
    ".netcdf",
    ".lh5",
}

ANALYSIS_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".dat",
    ".xvg",
    ".json",
    ".npy",
    ".npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventaría el contenido de un paquete de ATLAS."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_json = args.output_json.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {input_dir}")

    files = sorted(path for path in input_dir.rglob("*") if path.is_file())

    extension_counts: Counter[str] = Counter()
    records: list[dict[str, object]] = []

    topologies: list[str] = []
    trajectories: list[str] = []
    analyses: list[str] = []

    total_bytes = 0

    for path in files:
        relative_path = path.relative_to(input_dir)
        suffix = path.suffix.lower()
        size = path.stat().st_size

        extension_counts[suffix or "<sin_extension>"] += 1
        total_bytes += size

        record = {
            "path": str(relative_path),
            "extension": suffix,
            "size_bytes": size,
        }
        records.append(record)

        if suffix in TOPOLOGY_EXTENSIONS:
            topologies.append(str(relative_path))

        if suffix in TRAJECTORY_EXTENSIONS:
            trajectories.append(str(relative_path))

        if suffix in ANALYSIS_EXTENSIONS:
            analyses.append(str(relative_path))

    result = {
        "input_directory": str(input_dir),
        "file_count": len(files),
        "total_size_bytes": total_bytes,
        "extension_counts": dict(sorted(extension_counts.items())),
        "topology_candidates": topologies,
        "trajectory_candidates": trajectories,
        "analysis_candidates": analyses,
        "files": records,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Directorio: {input_dir}")
    print(f"Archivos: {len(files)}")
    print(f"Tamaño total: {total_bytes / (1024 ** 2):.2f} MiB")

    print("\nExtensiones:")
    for extension, count in sorted(extension_counts.items()):
        print(f"  {extension:<16} {count}")

    print("\nTopologías candidatas:")
    for path in topologies:
        print(f"  {path}")

    print("\nTrayectorias candidatas:")
    for path in trajectories:
        print(f"  {path}")

    print("\nArchivos de análisis:")
    for path in analyses[:50]:
        print(f"  {path}")

    if len(analyses) > 50:
        print(f"  ... y {len(analyses) - 50} archivos más")

    print(f"\nInventario guardado en: {output_json}")


if __name__ == "__main__":
    main()
