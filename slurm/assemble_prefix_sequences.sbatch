#!/bin/bash
#SBATCH --job-name=atlas_prefix_seq
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

SYSTEM="1k5n_A"

ROOT="$WORK/processed/atlas/smoke/$SYSTEM"
BASIC_DIR="$ROOT/basic"
CORE_RMSD_DIR="$ROOT/core_rmsd"
SASA_DSSP_DIR="$ROOT/sasa_dssp"
MASTER_TARGETS="$ROOT/master_targets/master_targets_wide.csv"

OUTPUT_DIR="$ROOT/prefix_sequences"

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
echo "ENSAMBLAJE DE SECUENCIAS DEL PREFIJO"
echo "========================================"
echo "Fecha:   $(date)"
echo "Nodo:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Sistema: $SYSTEM"
echo

python "$PROJECT/src/dataset/assemble_prefix_sequences.py" \
  --system "$SYSTEM" \
  --basic-dir "$BASIC_DIR" \
  --core-rmsd-dir "$CORE_RMSD_DIR" \
  --sasa-dssp-dir "$SASA_DSSP_DIR" \
  --master-targets "$MASTER_TARGETS" \
  --output-dir "$OUTPUT_DIR" \
  --prefix-end-ns 20 \
  --expected-step-ps 100

echo
echo "Archivos generados:"
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
