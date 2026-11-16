#!/usr/bin/env bash
#SBATCH --job-name=cils_v3_pilot_000
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/pilot_%j.out
#SBATCH --error=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/pilot_%j.err
set -euo pipefail
module purge
module load Python/3.11.5-GCCcore-13.2.0
source /home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/activate
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PROG=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/earlystop/cils_v3_shards.py
STATE=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1
printf 'PILOT_START job=%s time=%s\n' "$SLURM_JOB_ID" "$(date -Is)"
python3 "$PROG" run --shard 0 --policy include
python3 "$PROG" run --shard 0 --policy exclude
python3 - "$STATE" <<'PY'
import json, pathlib, sys
state=pathlib.Path(sys.argv[1]); parts=state/'parts'
for policy in ('include','exclude'):
  cp=json.loads((parts/f'{policy}_000'/'complete.json').read_text())
  print('PILOT_CHECKPOINT',policy,'systems',cp['n_systems'],'rows',cp['n_rows'],
        'wall_seconds',cp['elapsed_seconds'],'maxrss_kib',cp['child_maxrss_kib'],
        'cpu_seconds',cp['child_cpu_seconds'])
PY
printf 'PILOT_PASS job=%s time=%s\n' "$SLURM_JOB_ID" "$(date -Is)"
