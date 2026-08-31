#!/bin/bash
#SBATCH --job-name=atlas_inventory
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"
VENV="$HOME/venvs/md-poc-py311"

INPUT_DIR="$WORK/raw/atlas/smoke/1k5n_A/unpacked"
OUTPUT_JSON="$WORK/results/atlas_inventory/1k5n_A_inventory.json"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1

python "$PROJECT/src/data/inventory_atlas.py" \
    "$INPUT_DIR" \
    "$OUTPUT_JSON"

deactivate
