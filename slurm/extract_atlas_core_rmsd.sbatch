#!/bin/bash
#SBATCH --job-name=atlas_core_rmsd_generic
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "========================================"
echo "ATLAS CORE-RMSD GENERIC"
echo "========================================"
echo "Fecha:    $(date)"
echo "Nodo:     $(hostname)"
echo "Job ID:   $SLURM_JOB_ID"
echo "Sistema:  $SYSTEM"
echo "Grupo:    $DATASET_GROUP"
echo

python "$PROJECT/src/pipeline/extract_atlas_core_rmsd.py" \
  --system "$SYSTEM" \
  --dataset-group "$DATASET_GROUP" \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --replicas "1,2,3" \
  --quantiles "0.30,0.40,0.50" \
  --prefix-end-ns 20 \
  --tail-start-ns 80 \
  --expected-native-step-ps 10 \
  --expected-duration-ns 100

deactivate
