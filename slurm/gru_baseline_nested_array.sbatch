#!/bin/bash
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?Falta SLURM_ARRAY_TASK_ID}"

if [[ "$#" -ne 4 ]]; then
    echo "ERROR: uso:"
    echo "$0 DATA_ROOT OUTPUT_ROOT TRAINER_PATH EXPECTED_TRAINER_SHA256"
    exit 2
fi

DATA_ROOT="$1"
OUTPUT_ROOT="$2"
TRAINER="$3"
EXPECTED_TRAINER_SHA256="$4"

TASK_ID="$SLURM_ARRAY_TASK_ID"

if (( TASK_ID < 0 || TASK_ID > 54 )); then
    echo "ERROR: task_id fuera de rango: $TASK_ID"
    exit 2
fi

OUTER=$(( TASK_ID / 5 + 1 ))
INNER=$(( TASK_ID % 5 + 1 ))

FOLD_ID="$(
    printf 'fold_%02d_inner_%02d' \
        "$OUTER" \
        "$INNER"
)"

FOLD_NPZ="$DATA_ROOT/folds/${FOLD_ID}.npz"
RUN_DIR="$OUTPUT_ROOT/folds/$FOLD_ID"

STATUS_DIR="$OUTPUT_ROOT/status"
PASS_FILE="$STATUS_DIR/${FOLD_ID}.PASS"
FAIL_FILE="$STATUS_DIR/${FOLD_ID}.FAIL"

mkdir -p "$STATUS_DIR"

rm -f \
    "$PASS_FILE" \
    "$FAIL_FILE"

on_exit() {
    RC=$?

    if [[ "$RC" -eq 0 ]]; then
        printf 'job_id=%s\ntask_id=%s\nfold_id=%s\n' \
            "${SLURM_JOB_ID:-NA}" \
            "$TASK_ID" \
            "$FOLD_ID" \
            > "$PASS_FILE"
    else
        printf 'exit_code=%s\njob_id=%s\ntask_id=%s\nfold_id=%s\n' \
            "$RC" \
            "${SLURM_JOB_ID:-NA}" \
            "$TASK_ID" \
            "$FOLD_ID" \
            > "$FAIL_FILE"
    fi
}

trap on_exit EXIT

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONNOUSERSITE=1

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "$HOME/venvs/md-poc-py311/bin/activate"

echo "================================================================================"
echo "GRU BASELINE NESTED ARRAY"
echo "================================================================================"
echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-NA}"
echo "task_id=$TASK_ID"
echo "outer=$OUTER"
echo "inner=$INNER"
echo "fold_id=$FOLD_ID"
echo "fold_npz=$FOLD_NPZ"
echo "run_dir=$RUN_DIR"
echo "trainer=$TRAINER"
echo

if [[ ! -f "$FOLD_NPZ" ]]; then
    echo "ERROR: fold inexistente: $FOLD_NPZ"
    exit 3
fi

if [[ ! -f "$TRAINER" ]]; then
    echo "ERROR: trainer inexistente: $TRAINER"
    exit 4
fi

ACTUAL_TRAINER_SHA256="$(
    sha256sum "$TRAINER" \
        | awk '{print $1}'
)"

echo "expected_trainer_sha256=$EXPECTED_TRAINER_SHA256"
echo "actual_trainer_sha256=$ACTUAL_TRAINER_SHA256"

if [[ "$ACTUAL_TRAINER_SHA256" != "$EXPECTED_TRAINER_SHA256" ]]; then
    echo "ERROR: trainer SHA256 mismatch"
    exit 5
fi

if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: output ya existe: $RUN_DIR"
    exit 6
fi

echo
echo "===== RUNTIME ====="

python --version

python - <<'PY'
import numpy
import torch

print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_num_threads:", torch.get_num_threads())
PY

echo
echo "===== TRAIN ====="

python \
    "$TRAINER" \
    --fold-npz "$FOLD_NPZ" \
    --output-dir "$RUN_DIR" \
    --seed 20260811 \
    --hidden-size 32 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --max-epochs 2000 \
    --patience 50 \
    --min-delta 0.00001

echo
echo "===== VERIFY OUTPUT ====="

(
    cd "$RUN_DIR"
    sha256sum -c SHA256SUMS.txt
)

echo
echo "FOLD_TRAINING: PASS"
echo "OUTER_TEST_USED: NO"
echo "fold_id=$FOLD_ID"
