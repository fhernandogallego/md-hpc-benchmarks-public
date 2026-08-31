#!/bin/bash
#SBATCH --job-name=atlas_basic
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G

set -euo pipefail

: "${SYSTEM:?Falta SYSTEM}"
: "${DATASET_GROUP:?Falta DATASET_GROUP}"
: "${RAW_ROOT:?Falta RAW_ROOT}"
: "${OUTPUT_ROOT:?Falta OUTPUT_ROOT}"
: "${REPLICA:?Falta REPLICA}"

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
echo "EXTRACCIÓN BÁSICA ATLAS"
echo "================================================"
echo "Fecha:          $(date)"
echo "Nodo:           $(hostname)"
echo "Job ID:         $SLURM_JOB_ID"
echo "Sistema:        $SYSTEM"
echo "Grupo:          $DATASET_GROUP"
echo "Réplica:        $REPLICA"
echo "Raw root:       $RAW_ROOT"
echo "Output root:    $OUTPUT_ROOT"
echo

python "$PROJECT/src/pipeline/extract_atlas_basic.py" \
  --system "$SYSTEM" \
  --dataset-group "$DATASET_GROUP" \
  --replica "$REPLICA" \
  --raw-root "$RAW_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --prefix-end-ns 20 \
  --coarse-step-ps 100 \
  --expected-native-step-ps 10 \
  --expected-duration-ns 100

deactivate
