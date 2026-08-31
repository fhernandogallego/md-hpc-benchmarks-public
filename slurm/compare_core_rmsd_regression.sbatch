#!/bin/bash
#SBATCH --job-name=atlas_core_rmsd_reg
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

SYSTEM="1k5n_A"

CANONICAL_DIR="$WORK/processed/atlas/smoke/$SYSTEM/core_rmsd"
CANDIDATE_DIR="$WORK/processed/atlas_parameterized_test/smoke/$SYSTEM/core_rmsd"
OUTPUT_DIR="$WORK/results/parameterization_regression/$SYSTEM/core_rmsd"

FILENAMES="core_rmsd_summary.csv,core_rmsd_comparisons.csv,core_rmsd_selections.csv"

for REPLICA in 1 2 3
do
    for QUANTILE in 30 40 50
    do
        FILENAMES="${FILENAMES},${SYSTEM}_R${REPLICA}_q${QUANTILE}_core_rmsd.csv"
    done
done

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python "$PROJECT/src/pipeline/compare_csv_regression.py" \
  --canonical-dir "$CANONICAL_DIR" \
  --candidate-dir "$CANDIDATE_DIR" \
  --filenames "$FILENAMES" \
  --output-dir "$OUTPUT_DIR" \
  --path-columns "timeseries_csv" \
  --tolerance 1e-10

deactivate
