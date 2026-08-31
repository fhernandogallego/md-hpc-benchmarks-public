#!/bin/bash
#SBATCH --job-name=atlas_preflight
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

: "${SYSTEM:?Falta SYSTEM}"
: "${UNPACKED_DIR:?Falta UNPACKED_DIR}"
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

echo "========================================"
echo "ATLAS SYSTEM PREFLIGHT"
echo "========================================"
echo "Fecha:      $(date)"
echo "Nodo:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Sistema:    $SYSTEM"
echo "Réplicas:   $REPLICAS"
echo "Unpacked:   $UNPACKED_DIR"
echo "Output:     $OUTPUT_DIR"
echo

python "$PROJECT/src/pipeline/preflight_atlas_system.py" \
  --system "$SYSTEM" \
  --replicas "$REPLICAS" \
  --unpacked-dir "$UNPACKED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --analysis-step-ps 100 \
  --protein-step-ps 10 \
  --expected-end-ns 100 \
  --time-tolerance-ps 1e-3

deactivate
