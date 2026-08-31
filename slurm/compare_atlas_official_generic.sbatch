#!/bin/bash
#SBATCH --job-name=atlas_compare
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

set -euo pipefail

: "${SYSTEM:?Falta SYSTEM}"
: "${OFFICIAL_DIR:?Falta OFFICIAL_DIR}"
: "${CALCULATED_DIR:?Falta CALCULATED_DIR}"
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

echo "============================================"
echo "COMPARACIÓN CONTRA ATLAS OFICIAL"
echo "============================================"
echo "Fecha:       $(date)"
echo "Nodo:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID"
echo "Sistema:     $SYSTEM"
echo "Official:    $OFFICIAL_DIR"
echo "Calculated:  $CALCULATED_DIR"
echo "Output:      $OUTPUT_DIR"
echo

python "$PROJECT/src/pipeline/compare_atlas_official_generic.py" \
  --system "$SYSTEM" \
  --official-dir "$OFFICIAL_DIR" \
  --calculated-dir "$CALCULATED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --legacy-script \
    "$PROJECT/src/validation/compare_atlas_official.py"

deactivate
