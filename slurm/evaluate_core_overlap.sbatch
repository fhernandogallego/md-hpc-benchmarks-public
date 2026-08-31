#!/bin/bash
#SBATCH --job-name=atlas_core
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

INPUT_CSV="$WORK/processed/atlas/smoke/1k5n_A/rmsf/rmsf_prefix_full_long.csv"
OUTPUT_DIR="$WORK/results/core_overlap/1k5n_A"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "EVALUACIÓN DEL NÚCLEO RÍGIDO"
echo "========================================"
echo "Fecha:   $(date)"
echo "Nodo:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo

python "$PROJECT/src/core/evaluate_core_overlap.py" \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --quantiles "0.30,0.40,0.50" \
  --min-votes 2 \
  --min-segment-length 3

echo
echo "Archivos generados:"
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
