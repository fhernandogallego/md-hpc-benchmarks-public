#!/bin/bash
#SBATCH --job-name=atlas_eqsens
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

INPUT_DIR="$WORK/processed/atlas/smoke/1k5n_A/basic"
OUTPUT_DIR="$WORK/results/equilibration_sensitivity/1k5n_A"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

python "$PROJECT/src/diagnostics/equilibration_sensitivity.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

deactivate
