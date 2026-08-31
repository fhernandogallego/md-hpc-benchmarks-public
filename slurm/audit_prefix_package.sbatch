#!/bin/bash
#SBATCH --job-name=atlas_prefix_audit
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

PACKAGE_DIR="$WORK/processed/atlas/smoke/1k5n_A/prefix_sequences"
OUTPUT_DIR="$PACKAGE_DIR/audit"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

python "$PROJECT/src/dataset/audit_prefix_package.py" \
  --npz "$PACKAGE_DIR/prefix_sequences_raw.npz" \
  --manifest "$PACKAGE_DIR/prefix_sequence_manifest.csv" \
  --metadata "$PACKAGE_DIR/prefix_sequences_metadata.json" \
  --output-dir "$OUTPUT_DIR" \
  --expected-steps 201 \
  --expected-step-ps 100 \
  --maximum-time-ns 20

echo
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
