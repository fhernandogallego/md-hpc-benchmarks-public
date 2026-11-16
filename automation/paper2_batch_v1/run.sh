#!/bin/bash
# Single public entry point; loading the native Python module is mandatory.
set -euo pipefail
tool_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/venvs/md-poc-py311/bin/activate"
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
exec python -u "$tool_root/controller.py" "$@"
