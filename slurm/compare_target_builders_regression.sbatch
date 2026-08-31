#!/bin/bash
#SBATCH --job-name=atlas_target_reg
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

CANONICAL_ROOT="$WORK/processed/atlas/smoke/$SYSTEM"
CANDIDATE_ROOT="$WORK/processed/atlas_parameterized_test/smoke/$SYSTEM"
RESULT_ROOT="$WORK/results/parameterization_regression/$SYSTEM"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$PROJECT/src/pipeline/compare_csv_regression.py" \
  --canonical-dir "$CANONICAL_ROOT/targets" \
  --candidate-dir "$CANDIDATE_ROOT/targets" \
  --filenames "validated_targets_long.csv,model_targets_wide.csv" \
  --output-dir "$RESULT_ROOT/basic_targets" \
  --tolerance 1e-10

python "$PROJECT/src/pipeline/compare_csv_regression.py" \
  --canonical-dir "$CANONICAL_ROOT/structural_targets_compositional" \
  --candidate-dir "$CANDIDATE_ROOT/structural_targets_compositional" \
  --filenames "structural_targets_long.csv,structural_model_targets_wide.csv,secondary_structure_composition.csv" \
  --output-dir "$RESULT_ROOT/structural_targets" \
  --tolerance 1e-10

deactivate
