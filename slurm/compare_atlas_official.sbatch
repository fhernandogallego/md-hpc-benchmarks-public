#!/bin/bash
#SBATCH --job-name=atlas_compare
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

OFFICIAL_DIR="$WORK/raw/atlas/smoke/1k5n_A/unpacked/1k5n_A_analysis"
CALCULATED_DIR="$WORK/processed/atlas/smoke/1k5n_A/basic"
OUTPUT_DIR="$WORK/results/atlas_comparison/1k5n_A"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "COMPARACIÓN CON ATLAS"
echo "========================================"
echo "Fecha:      $(date)"
echo "Nodo:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Oficial:    $OFFICIAL_DIR"
echo "Calculado:  $CALCULATED_DIR"
echo "Salida:     $OUTPUT_DIR"
echo

python "$PROJECT/src/validation/compare_atlas_official.py" \
  --official-dir "$OFFICIAL_DIR" \
  --calculated-dir "$CALCULATED_DIR" \
  --output-dir "$OUTPUT_DIR"

deactivate
