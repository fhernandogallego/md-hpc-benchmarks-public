#!/bin/bash
#SBATCH --job-name=atlas_sasa_generic
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

: "${SYSTEM:?Falta exportar SYSTEM}"
: "${DATASET_GROUP:?Falta exportar DATASET_GROUP}"
: "${PROCESSED_ROOT:?Falta exportar PROCESSED_ROOT}"

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

RAW_ROOT="$WORK/raw/atlas"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "========================================"
echo "ATLAS SASA/DSSP GENERIC"
echo "========================================"
echo "Fecha:    $(date)"
echo "Nodo:     $(hostname)"
echo "Job ID:   $SLURM_JOB_ID"
echo "Sistema:  $SYSTEM"
echo "Grupo:    $DATASET_GROUP"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo

python "$PROJECT/src/pipeline/extract_atlas_sasa_dssp.py" \
  --system "$SYSTEM" \
  --dataset-group "$DATASET_GROUP" \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --replicas "1,2,3" \
  --stride 10 \
  --chunk-size 25 \
  --sphere-points 960 \
  --probe-radius-nm 0.14 \
  --prefix-end-ns 20 \
  --expected-native-step-ps 10 \
  --expected-duration-ns 100

deactivate
