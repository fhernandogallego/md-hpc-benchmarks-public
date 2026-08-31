#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


LEGACY_SYSTEM = "1k5n_A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el comparador oficial heredado con cualquier "
            "sistema ATLAS mediante alias temporales reproducibles."
        )
    )

    parser.add_argument(
        "--system",
        required=True,
    )
    parser.add_argument(
        "--official-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--calculated-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--legacy-script",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def require_directory(path: Path) -> Path:
    path = path.resolve()

    if not path.is_dir():
        raise FileNotFoundError(path)

    return path


def require_file(path: Path) -> Path:
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Archivo vacío: {path}"
        )

    return path


def create_alias(
    source: Path,
    destination: Path,
) -> None:
    source = require_file(source)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    destination.symlink_to(source)


def alias_system_files(
    source_dir: Path,
    destination_dir: Path,
    system: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    matching_files = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.name.startswith(system)
    )

    if not matching_files:
        raise RuntimeError(
            f"No se encontraron archivos que comiencen por "
            f"{system!r} en {source_dir}."
        )

    for source in matching_files:
        alias_name = source.name.replace(
            system,
            LEGACY_SYSTEM,
            1,
        )

        destination = (
            destination_dir
            / alias_name
        )

        create_alias(
            source,
            destination,
        )

        records.append(
            {
                "source": str(
                    source.resolve()
                ),
                "alias": str(
                    destination
                ),
            }
        )

    return records


def main() -> int:
    args = parse_args()

    official_dir = require_directory(
        args.official_dir
    )
    calculated_dir = require_directory(
        args.calculated_dir
    )
    legacy_script = require_file(
        args.legacy_script
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    compatibility_dir = (
        output_dir
        / "_legacy_compatibility"
    )
    official_alias_dir = (
        compatibility_dir
        / "official"
    )
    calculated_alias_dir = (
        compatibility_dir
        / "calculated"
    )

    official_alias_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    calculated_alias_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    official_aliases = alias_system_files(
        source_dir=official_dir,
        destination_dir=official_alias_dir,
        system=args.system,
    )

    calculated_aliases = alias_system_files(
        source_dir=calculated_dir,
        destination_dir=calculated_alias_dir,
        system=args.system,
    )

    required_aliases = [
        official_alias_dir
        / f"{LEGACY_SYSTEM}_RMSD.tsv",
        official_alias_dir
        / f"{LEGACY_SYSTEM}_gyrate.tsv",
    ]

    for replica in (1, 2, 3):
        required_aliases.append(
            calculated_alias_dir
            / (
                f"{LEGACY_SYSTEM}_R{replica}_"
                "basic_observables_100ps.csv"
            )
        )

    for filename in required_aliases:
        if not filename.is_file():
            raise FileNotFoundError(
                f"Alias requerido no creado: {filename}"
            )

    command = [
        sys.executable,
        str(legacy_script),
        "--official-dir",
        str(official_alias_dir),
        "--calculated-dir",
        str(calculated_alias_dir),
        "--output-dir",
        str(output_dir),
    ]

    print("=" * 100)
    print("COMPARADOR ATLAS GENÉRICO")
    print("=" * 100)
    print(f"Sistema real:       {args.system}")
    print(f"Alias heredado:     {LEGACY_SYSTEM}")
    print(f"Official real:      {official_dir}")
    print(f"Calculated real:    {calculated_dir}")
    print(f"Output:             {output_dir}")
    print(f"Comparador legado:  {legacy_script}")
    print()
    print("Comando:")
    print(" ".join(command))
    print()

    completed = subprocess.run(
        command,
        check=False,
    )

    metadata = {
        "system": args.system,
        "legacy_system_alias": LEGACY_SYSTEM,
        "legacy_script": str(
            legacy_script
        ),
        "official_directory": str(
            official_dir
        ),
        "calculated_directory": str(
            calculated_dir
        ),
        "output_directory": str(
            output_dir
        ),
        "return_code": int(
            completed.returncode
        ),
        "official_aliases": (
            official_aliases
        ),
        "calculated_aliases": (
            calculated_aliases
        ),
    }

    metadata_file = (
        output_dir
        / "generic_comparison_metadata.json"
    )

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Metadatos del adaptador: "
        f"{metadata_file}"
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "El comparador heredado terminó con código "
            f"{completed.returncode}."
        )

    print("COMPARADOR GENÉRICO: COMPLETED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
