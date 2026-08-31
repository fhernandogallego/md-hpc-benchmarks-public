#!/bin/bash
#SBATCH --job-name=atlas_master_targets
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

BASIC_LONG="$WORK/processed/atlas/smoke/1k5n_A/targets/validated_targets_long.csv"

STRUCTURAL_DIR="$WORK/processed/atlas/smoke/1k5n_A/structural_targets_compositional"
STRUCTURAL_LONG="$STRUCTURAL_DIR/structural_targets_long.csv"
COMPOSITION_CSV="$STRUCTURAL_DIR/secondary_structure_composition.csv"

OUTPUT_DIR="$WORK/processed/atlas/smoke/1k5n_A/master_targets"

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
echo "ENSAMBLAJE DE TARGETS MAESTROS"
echo "========================================"
echo "Fecha:  $(date)"
echo "Nodo:   $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo

python "$PROJECT/src/dataset/assemble_master_targets.py" \
  --basic-long "$BASIC_LONG" \
  --structural-long "$STRUCTURAL_LONG" \
  --composition-csv "$COMPOSITION_CSV" \
  --output-dir "$OUTPUT_DIR"

echo
echo "Archivos generados:"
find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\t%s bytes\n' \
  | sort

deactivate
