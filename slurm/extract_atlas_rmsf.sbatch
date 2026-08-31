#!/bin/bash
#SBATCH --job-name=atlas_rmsf
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
: "${DATASET_GROUP:?Falta DATASET_GROUP}"
: "${RAW_ROOT:?Falta RAW_ROOT}"
: "${OUTPUT_ROOT:?Falta OUTPUT_ROOT}"

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
echo "EXTRACCIÓN RMSF ATLAS"
echo "================================================"
echo "Fecha:          $(date)"
echo "Nodo:           $(hostname)"
echo "Job ID:         $SLURM_JOB_ID"
echo "Sistema:        $SYSTEM"
echo "Grupo:          $DATASET_GROUP"
echo "Réplicas:       $REPLICAS"
echo "Raw root:       $RAW_ROOT"
echo "Output root:    $OUTPUT_ROOT"
echo

python "$PROJECT/src/pipeline/extract_atlas_rmsf.py" \
  --system "$SYSTEM" \
  --dataset-group "$DATASET_GROUP" \
  --raw-root "$RAW_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --replicas "$REPLICAS" \
  --start-ns 0.1 \
  --prefix-end-ns 20 \
  --expected-native-step-ps 10 \
  --expected-duration-ns 100

deactivate
