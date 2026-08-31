#!/bin/bash
#SBATCH --job-name=atlas_vtargets
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

INPUT_DIR="$WORK/processed/atlas/smoke/1k5n_A/basic"
SENSITIVITY_CSV="$WORK/results/equilibration_sensitivity/1k5n_A/equilibration_sensitivity.csv"
OUTPUT_DIR="$WORK/processed/atlas/smoke/1k5n_A/targets"

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
echo "TARGETS VALIDADOS POR OBSERVABLE"
echo "========================================"
echo "Fecha:   $(date)"
echo "Nodo:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo

python "$PROJECT/src/targets/build_validated_targets.py" \
  --input-dir "$INPUT_DIR" \
  --sensitivity-csv "$SENSITIVITY_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --nskip 10 \
  --min-duration-ns 20 \
  --min-neff 10 \
  --tail-start-ns 80

echo
echo "Archivos generados:"
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
