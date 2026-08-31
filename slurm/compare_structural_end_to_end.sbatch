#!/bin/bash
#SBATCH --job-name=atlas_struct_e2e_reg
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

SYSTEM="1k5n_A"

CANONICAL_DIR="$WORK/processed/atlas/smoke/$SYSTEM/structural_targets_compositional"
CANDIDATE_DIR="$WORK/processed/atlas_parameterized_test/smoke/$SYSTEM/structural_targets_end_to_end"
OUTPUT_DIR="$WORK/results/parameterization_regression/$SYSTEM/structural_targets_end_to_end"

FILENAMES="structural_targets_long.csv,structural_model_targets_wide.csv,secondary_structure_composition.csv"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$PROJECT/src/pipeline/compare_csv_regression.py" \
  --canonical-dir "$CANONICAL_DIR" \
  --candidate-dir "$CANDIDATE_DIR" \
  --filenames "$FILENAMES" \
  --output-dir "$OUTPUT_DIR" \
  --tolerance 1e-10

deactivate
