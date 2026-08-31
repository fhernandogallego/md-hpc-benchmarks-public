#!/bin/bash
#SBATCH --job-name=atlas_core_rmsd
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

INPUT_DIR="$WORK/raw/atlas/smoke/1k5n_A/unpacked"
BASIC_DIR="$WORK/processed/atlas/smoke/1k5n_A/basic"

CORE_DIR="$WORK/results/core_overlap/1k5n_A"
REPLICA_CORE_CSV="$CORE_DIR/core_replica_residues.csv"
CONSENSUS_CORE_CSV="$CORE_DIR/core_consensus_residues.csv"

OUTPUT_DIR="$WORK/processed/atlas/smoke/1k5n_A/core_rmsd"

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

echo "========================================"
echo "EXTRACCIÓN DE CORE-RMSD"
echo "========================================"
echo "Fecha:   $(date)"
echo "Nodo:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo

python "$PROJECT/src/features/extract_core_rmsd.py" \
  --input-dir "$INPUT_DIR" \
  --basic-dir "$BASIC_DIR" \
  --replica-core-csv "$REPLICA_CORE_CSV" \
  --consensus-core-csv "$CONSENSUS_CORE_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --quantiles "0.30,0.40,0.50" \
  --prefix-end-ns 20 \
  --tail-start-ns 80

echo
echo "Archivos generados:"
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
