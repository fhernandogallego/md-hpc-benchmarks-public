#!/bin/bash
#SBATCH --job-name=atlas_sasa
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

: "${SYSTEM:?Falta SYSTEM}"
: "${DATASET_GROUP:?Falta DATASET_GROUP}"
: "${RAW_ROOT:?Falta RAW_ROOT}"
: "${PROCESSED_ROOT:?Falta PROCESSED_ROOT}"

REPLICAS="${REPLICAS:-1,2,3}"
STRIDE="${STRIDE:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-25}"
PREFIX_END_NS="${PREFIX_END_NS:-20}"

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
echo "EXTRACCIÓN SASA/DSSP ATLAS"
echo "================================================"
echo "Fecha:             $(date)"
echo "Nodo:              $(hostname)"
echo "Job ID:            $SLURM_JOB_ID"
echo "Sistema:           $SYSTEM"
echo "Grupo:             $DATASET_GROUP"
echo "Réplicas:          $REPLICAS"
echo "Stride:            $STRIDE"
echo "Paso de salida:    100 ps"
echo "Chunk size:        $CHUNK_SIZE"
echo "Prefijo:           0–${PREFIX_END_NS} ns"
echo "Raw root:          $RAW_ROOT"
echo "Processed root:    $PROCESSED_ROOT"
echo

python "$PROJECT/src/pipeline/extract_atlas_sasa_dssp.py" \
  --system "$SYSTEM" \
  --dataset-group "$DATASET_GROUP" \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --replicas "$REPLICAS" \
  --stride "$STRIDE" \
  --chunk-size "$CHUNK_SIZE" \
  --prefix-end-ns "$PREFIX_END_NS" \
  --expected-native-step-ps 10 \
  --expected-duration-ns 100

deactivate
