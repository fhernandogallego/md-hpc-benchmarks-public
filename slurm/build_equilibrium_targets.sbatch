#!/bin/bash
#SBATCH --job-name=atlas_targets
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

INPUT_DIR="$WORK/processed/atlas/smoke/1k5n_A/basic"
OUTPUT_DIR="$WORK/processed/atlas/smoke/1k5n_A/targets"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "CONSTRUCCIÓN DE TARGETS DE EQUILIBRIO"
echo "========================================"
echo "Fecha:   $(date)"
echo "Nodo:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Entrada: $INPUT_DIR"
echo "Salida:  $OUTPUT_DIR"
echo

python "$PROJECT/src/targets/build_equilibrium_targets.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --nskip 10

echo
echo "Archivos generados:"
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
