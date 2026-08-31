#!/bin/bash
#SBATCH --job-name=atlas_manifest
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

MANIFEST="$PROJECT/manifests/atlas_systems.csv"
RAW_ROOT="$WORK/raw/atlas"
OUTPUT_CSV="$WORK/results/atlas_manifest/atlas_manifest_validation.csv"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$PROJECT/src/dataset/validate_atlas_manifest.py" \
  --manifest "$MANIFEST" \
  --raw-root "$RAW_ROOT" \
  --output-csv "$OUTPUT_CSV"

deactivate
