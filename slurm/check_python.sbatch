#!/bin/bash
#SBATCH --job-name=check_python
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "$HOME/venvs/md-poc-py311/bin/activate"

echo "========================================"
echo "COMPROBACIÓN DEL ENTORNO PYTHON"
echo "========================================"
echo "Fecha:       $(date)"
echo "Nodo:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID"
echo "Python:      $(which python)"
echo "Versión:     $(python --version 2>&1)"
echo "Pip:         $(python -m pip --version)"
echo

python - <<'PY'
import os
import platform
import sys

print("sys.executable:", sys.executable)
print("sys.version:", sys.version.replace("\n", " "))
print("platform:", platform.platform())
print("CPU asignadas:", os.environ.get("SLURM_CPUS_PER_TASK"))
print("Python funcional dentro del nodo.")
PY

deactivate

echo
echo "Comprobación terminada correctamente."
