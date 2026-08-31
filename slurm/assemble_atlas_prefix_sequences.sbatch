#!/bin/bash
#SBATCH --job-name=atlas_prefix_seq
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

: "${SYSTEM:?Falta SYSTEM}"
: "${BASIC_DIR:?Falta BASIC_DIR}"
: "${CORE_RMSD_DIR:?Falta CORE_RMSD_DIR}"
: "${SASA_DSSP_DIR:?Falta SASA_DSSP_DIR}"
: "${MASTER_TARGETS:?Falta MASTER_TARGETS}"
: "${OUTPUT_DIR:?Falta OUTPUT_DIR}"

REPLICAS="${REPLICAS:-1,2,3}"

PROJECT="$HOME/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "========================================"
echo "ATLAS PREFIX SEQUENCES"
echo "========================================"
echo "Fecha:      $(date)"
echo "Nodo:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Sistema:    $SYSTEM"
echo "Réplicas:   $REPLICAS"
echo "Basic:      $BASIC_DIR"
echo "Core RMSD:  $CORE_RMSD_DIR"
echo "SASA/DSSP:  $SASA_DSSP_DIR"
echo "Targets:    $MASTER_TARGETS"
echo "Output:     $OUTPUT_DIR"
echo

python "$PROJECT/src/dataset/assemble_prefix_sequences.py" \
  --system "$SYSTEM" \
  --replicas "$REPLICAS" \
  --basic-dir "$BASIC_DIR" \
  --core-rmsd-dir "$CORE_RMSD_DIR" \
  --sasa-dssp-dir "$SASA_DSSP_DIR" \
  --master-targets "$MASTER_TARGETS" \
  --output-dir "$OUTPUT_DIR" \
  --prefix-end-ns 20 \
  --expected-step-ps 100

deactivate
