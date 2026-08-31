#!/bin/bash
#SBATCH --job-name=atlas_rmsf
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

ANALYSIS_DIR="$WORK/raw/atlas/smoke/1k5n_A/unpacked/1k5n_A_analysis"
OUTPUT_DIR="$WORK/results/atlas_rmsf_validation/1k5n_A"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "VALIDACIÓN DEL RMSF DE ATLAS"
echo "========================================"
echo "Fecha:   $(date)"
echo "Nodo:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo

python "$PROJECT/src/validation/validate_atlas_rmsf.py" \
  --analysis-dir "$ANALYSIS_DIR" \
  --output-dir "$OUTPUT_DIR"

echo
echo "Archivos generados:"
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
