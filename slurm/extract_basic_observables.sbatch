#!/bin/bash
#SBATCH --job-name=atlas_basic
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

INPUT_DIR="$WORK/raw/atlas/smoke/1k5n_A/unpacked"
OUTPUT_DIR="$WORK/processed/atlas/smoke/1k5n_A/basic"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="$WORK/tmp/matplotlib_${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MPLCONFIGDIR"

echo "========================================"
echo "EXTRACCIÓN DE RMSD Y RADIO DE GIRO"
echo "========================================"
echo "Fecha:   $(date)"
echo "Nodo:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Entrada: $INPUT_DIR"
echo "Salida:  $OUTPUT_DIR"
echo

python "$PROJECT/src/features/extract_basic_observables.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prefix-ns 20

echo
echo "Archivos generados:"
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%k KB\n' \
  | sort

deactivate
