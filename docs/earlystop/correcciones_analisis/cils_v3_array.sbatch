#!/usr/bin/env bash
# CILS v3, bloque A: reconstrucción de objetivos de los fragmentos 001-075 (el 000 ya está validado por el piloto 3387582).
# Una tarea = un fragmento y una política. 150 tareas: 1..150 -> fragmento 1 + (id-1)/2; include en las impares, exclude en las pares.
# Recursos de la medición del piloto: 48,4 min y 0,13 GiB por tarea de 10 sistemas (ver PILOTO_3387582_RESULTADO.md).
# Lanzar SOLO tras confirmar los límites de la cuenta: sbatch --array=1-150%10 cils_v3_array.sbatch
#SBATCH --job-name=cils_v3_targets
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --array=1-150%10
#SBATCH --output=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/array_%A_%a.out
#SBATCH --error=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/array_%A_%a.err
set -euo pipefail
module purge
module load Python/3.11.5-GCCcore-13.2.0
source /home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PROG=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/earlystop/cils_v3_shards.py
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SHARD=$((1 + IDX / 2))
if [ $((IDX % 2)) -eq 0 ]; then POLICY=include; else POLICY=exclude; fi
printf 'ARRAY_TASK job=%s task=%s shard=%03d policy=%s time=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SHARD" "$POLICY" "$(date -Is)"
python3 "$PROG" run --shard "$SHARD" --policy "$POLICY"
printf 'ARRAY_TASK_PASS job=%s task=%s shard=%03d policy=%s time=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SHARD" "$POLICY" "$(date -Is)"
