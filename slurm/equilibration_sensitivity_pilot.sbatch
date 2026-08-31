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

: "${SYSTEM:?Falta SYSTEM}"
: "${INPUT_DIR:?Falta INPUT_DIR}"
: "${OUTPUT_DIR:?Falta OUTPUT_DIR}"

REPLICAS="${REPLICAS:-1,2,3}"

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

echo "================================================"
echo "SENSIBILIDAD DE EQUILIBRIO ATLAS"
echo "================================================"
echo "Fecha:       $(date)"
echo "Nodo:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID"
echo "Sistema:     $SYSTEM"
echo "Réplicas:    $REPLICAS"
echo "Input:       $INPUT_DIR"
echo "Output:      $OUTPUT_DIR"
echo

mkdir -p "$OUTPUT_DIR"

python "$PROJECT/src/diagnostics/equilibration_sensitivity.py" \
  --system "$SYSTEM" \
  --replicas "$REPLICAS" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

echo
echo "Archivos generados:"

find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
