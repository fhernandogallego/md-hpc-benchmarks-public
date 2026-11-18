#!/usr/bin/env bash
# CILS v3, bloque A: consolidación de los 76 fragmentos x 2 políticas (bloquea si falta o sobra algo).
# Uso: sbatch --dependency=afterok:<JOB_DEL_ARRAY> cils_v3_consolidar.sbatch
#SBATCH --job-name=cils_v3_merge
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/merge_%j.out
#SBATCH --error=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/merge_%j.err
set -euo pipefail
module purge
module load Python/3.11.5-GCCcore-13.2.0
source /home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PROG=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/earlystop/cils_v3_shards.py
python3 "$PROG" verify
python3 "$PROG" consolidate
