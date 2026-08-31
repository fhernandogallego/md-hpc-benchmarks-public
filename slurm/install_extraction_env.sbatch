#!/bin/bash
#SBATCH --job-name=install_md_env
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"
WHEELHOUSE="$WORK/wheels/extraction"
REQUIREMENTS="$PROJECT/configs/requirements-extraction.in"

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "$VENV/bin/activate"

export TMPDIR="$WORK/tmp/pip_${SLURM_JOB_ID}"
export PIP_CACHE_DIR="$WORK/tmp/pip-cache"

mkdir -p "$TMPDIR"
mkdir -p "$PIP_CACHE_DIR"

echo "========================================"
echo "INSTALACIÓN DEL ENTORNO DE EXTRACCIÓN"
echo "========================================"
echo "Fecha:       $(date)"
echo "Nodo:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID"
echo "Python:      $(which python)"
echo "Wheelhouse:  $WHEELHOUSE"
echo

if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "ERROR: no existe el wheelhouse: $WHEELHOUSE" >&2
    exit 1
fi

if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "ERROR: no existe el archivo de requisitos: $REQUIREMENTS" >&2
    exit 1
fi

echo "Actualizando herramientas de instalación desde wheels locales..."
python -m pip install \
  --no-index \
  --find-links="$WHEELHOUSE" \
  --upgrade pip setuptools wheel

echo
echo "Instalando dependencias..."
python -m pip install \
  --no-index \
  --find-links="$WHEELHOUSE" \
  -r "$REQUIREMENTS"

echo
echo "Comprobando dependencias..."
python -m pip check

echo
echo "Versiones instaladas:"
python - <<'PY'
import Bio
import MDAnalysis
import matplotlib
import mdtraj
import numpy
import pandas
import pymbar
import scipy
import yaml

packages = {
    "Python": __import__("sys").version.split()[0],
    "NumPy": numpy.__version__,
    "SciPy": scipy.__version__,
    "pandas": pandas.__version__,
    "Matplotlib": matplotlib.__version__,
    "PyYAML": yaml.__version__,
    "Biopython": Bio.__version__,
    "MDAnalysis": MDAnalysis.__version__,
    "MDTraj": mdtraj.__version__,
    "PyMBAR": pymbar.__version__,
}

for name, version in packages.items():
    print(f"{name:<15} {version}")
PY

echo
echo "Guardando lock del entorno..."
python -m pip freeze | sort > "$PROJECT/configs/requirements-extraction.lock.txt"

echo
echo "Lock creado en:"
ls -lh "$PROJECT/configs/requirements-extraction.lock.txt"

rm -rf "$TMPDIR"

echo
echo "Instalación terminada correctamente."
