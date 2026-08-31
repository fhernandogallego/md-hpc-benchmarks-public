#!/bin/bash
#SBATCH --job-name=atlas_basic_generic
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --array=1-3

set -euo pipefail

: "${SYSTEM:?Falta exportar SYSTEM}"
: "${DATASET_GROUP:?Falta exportar DATASET_GROUP}"
: "${OUTPUT_ROOT:?Falta exportar OUTPUT_ROOT}"

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

RAW_ROOT="$WORK/raw/atlas"
REPLICA="$SLURM_ARRAY_TASK_ID"

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
echo "ATLAS BASIC GENERIC"
echo "========================================"
echo "Fecha:    $(date)"
echo "Nodo:     $(hostname)"
echo "Job ID:   $SLURM_ARRAY_JOB_ID"
echo "Tarea:    $SLURM_ARRAY_TASK_ID"
echo "Sistema:  $SYSTEM"
echo "Grupo:    $DATASET_GROUP"
echo "Réplica:  R$REPLICA"
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
