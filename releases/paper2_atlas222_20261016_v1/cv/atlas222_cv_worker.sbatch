#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --account=uva_dma_2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00

set -euo pipefail
module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/venvs/md-poc-py311/bin/activate"
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
run="$1"
cd "$run/code"
export PYTHONPATH="$run/code"
python -u train_atlas222_grouped.py "$run" "${SLURM_ARRAY_TASK_ID:-aggregate}"
