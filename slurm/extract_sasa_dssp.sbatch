#!/bin/bash
#SBATCH --job-name=atlas_sasa_ss
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

INPUT_DIR="$WORK/raw/atlas/smoke/1k5n_A/unpacked"
OUTPUT_DIR="$WORK/processed/atlas/smoke/1k5n_A/sasa_dssp"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "EXTRACCIÓN DE SASA Y DSSP"
echo "========================================"
echo "Fecha:   $(date)"
echo "Nodo:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo

python "$PROJECT/src/features/extract_sasa_dssp.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --stride 10 \
  --chunk-size 25 \
  --sphere-points 960 \
  --prefix-end-ns 20

echo
echo "Archivos generados:"
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
