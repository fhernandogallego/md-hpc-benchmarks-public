#!/bin/bash
#SBATCH --job-name=atlas_master_targets
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

: "${BASIC_LONG:?Falta BASIC_LONG}"
: "${STRUCTURAL_LONG:?Falta STRUCTURAL_LONG}"
: "${COMPOSITION_CSV:?Falta COMPOSITION_CSV}"
: "${OUTPUT_DIR:?Falta OUTPUT_DIR}"

PROJECT="$HOME/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "========================================"
echo "ATLAS MASTER TARGETS"
echo "========================================"
echo "Fecha:  $(date)"
echo "Nodo:   $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo
echo "Basic:       $BASIC_LONG"
echo "Structural:  $STRUCTURAL_LONG"
echo "Composition: $COMPOSITION_CSV"
echo "Output:      $OUTPUT_DIR"
echo

python "$PROJECT/src/dataset/assemble_master_targets.py" \
  --basic-long "$BASIC_LONG" \
  --structural-long "$STRUCTURAL_LONG" \
  --composition-csv "$COMPOSITION_CSV" \
  --output-dir "$OUTPUT_DIR"

deactivate
