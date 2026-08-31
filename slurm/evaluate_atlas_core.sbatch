#!/bin/bash
#SBATCH --job-name=atlas_core
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

set -euo pipefail

: "${SYSTEM:?Falta SYSTEM}"
: "${DATASET_GROUP:?Falta DATASET_GROUP}"
: "${RMSF_LONG:?Falta RMSF_LONG}"
: "${OUTPUT_ROOT:?Falta OUTPUT_ROOT}"

PROJECT="$HOME/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

QUANTILES="${QUANTILES:-0.30,0.40,0.50}"
MIN_VOTES="${MIN_VOTES:-2}"
MIN_SEGMENT_LENGTH="${MIN_SEGMENT_LENGTH:-3}"

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
echo "EVALUACIÓN DE NÚCLEOS ATLAS"
echo "================================================"
echo "Fecha:                 $(date)"
echo "Nodo:                  $(hostname)"
echo "Job ID:                $SLURM_JOB_ID"
echo "Sistema:               $SYSTEM"
echo "Grupo:                 $DATASET_GROUP"
echo "RMSF:                  $RMSF_LONG"
echo "Output root:           $OUTPUT_ROOT"
echo "Cuantiles:             $QUANTILES"
echo "Votos mínimos:         $MIN_VOTES"
echo "Longitud mínima:       $MIN_SEGMENT_LENGTH"
echo

python "$PROJECT/src/pipeline/evaluate_atlas_core.py" \
  --system "$SYSTEM" \
  --dataset-group "$DATASET_GROUP" \
  --rmsf-long "$RMSF_LONG" \
  --output-root "$OUTPUT_ROOT" \
  --quantiles "$QUANTILES" \
  --min-votes "$MIN_VOTES" \
  --min-segment-length "$MIN_SEGMENT_LENGTH"

deactivate
