#!/bin/bash

set -euo pipefail

MODE="${1:---dry-run}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
    echo "Uso: $0 [--dry-run|--submit]"
    exit 2
fi

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"

DATA_ROOT="$WORK/results/preprocessing/atlas11_nested_training_v1_2026-08-11"

RUN_ID="atlas11_gru_baseline_nested_v1_2026-08-11"

OUTPUT_ROOT="$WORK/results/modeling/$RUN_ID"

TRAINER="$PROJECT/src/modeling/train_gru_baseline.py"
SBATCH_SOURCE="$PROJECT/slurm/gru_baseline_nested_array.sbatch"
SUBMITTER_SOURCE="$PROJECT/slurm/submit_gru_baseline_nested.sh"

EXPECTED_TRAINER_COMMIT="3c274fd730ca55bf6da34e11944e64568ab74613"

ARRAY_SPEC="0-54%11"

echo "================================================================================"
echo "GRU BASELINE NESTED SUBMITTER"
echo "================================================================================"
echo "mode=$MODE"
echo "project=$PROJECT"
echo "data_root=$DATA_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "array_spec=$ARRAY_SPEC"
echo

if [[ ! -f "$TRAINER" ]]; then
    echo "ERROR: trainer inexistente"
    exit 3
fi

if [[ ! -f "$SBATCH_SOURCE" ]]; then
    echo "ERROR: sbatch worker inexistente"
    exit 4
fi

TRAINER_LAST_COMMIT="$(
    git -C "$PROJECT" \
        log -1 \
        --format='%H' \
        -- \
        src/modeling/train_gru_baseline.py
)"

TRAINER_SHA256="$(
    sha256sum "$TRAINER" \
        | awk '{print $1}'
)"

CURRENT_HEAD="$(
    git -C "$PROJECT" \
        rev-parse HEAD
)"

echo "===== TRAINER PROVENANCE ====="
echo "current_head=$CURRENT_HEAD"
echo "trainer_last_commit=$TRAINER_LAST_COMMIT"
echo "expected_trainer_commit=$EXPECTED_TRAINER_COMMIT"
echo "trainer_sha256=$TRAINER_SHA256"

if [[ "$TRAINER_LAST_COMMIT" != "$EXPECTED_TRAINER_COMMIT" ]]; then
    echo "TRAINER_COMMIT: FAIL"
    exit 5
fi

echo "TRAINER_COMMIT: PASS"

if ! git -C "$PROJECT" diff --quiet -- \
    src/modeling/train_gru_baseline.py
then
    echo "TRAINER_WORKTREE_CLEAN: FAIL"
    exit 6
fi

if ! git -C "$PROJECT" diff --cached --quiet -- \
    src/modeling/train_gru_baseline.py
then
    echo "TRAINER_INDEX_CLEAN: FAIL"
    exit 7
fi

echo "TRAINER_WORKTREE_CLEAN: PASS"
echo "TRAINER_INDEX_CLEAN: PASS"

echo
echo "===== FOLD PREFLIGHT ====="

N_FOUND=0
N_MISSING=0

for TASK_ID in $(seq 0 54); do

    OUTER=$(( TASK_ID / 5 + 1 ))
    INNER=$(( TASK_ID % 5 + 1 ))

    FOLD_ID="$(
        printf 'fold_%02d_inner_%02d' \
            "$OUTER" \
            "$INNER"
    )"

    FOLD="$DATA_ROOT/folds/${FOLD_ID}.npz"

    if [[ -f "$FOLD" ]]; then
        N_FOUND=$(( N_FOUND + 1 ))
    else
        echo "MISSING: task=$TASK_ID fold=$FOLD_ID"
        N_MISSING=$(( N_MISSING + 1 ))
    fi
done

echo "folds_found=$N_FOUND"
echo "folds_missing=$N_MISSING"

if [[ "$N_FOUND" -ne 55 || "$N_MISSING" -ne 0 ]]; then
    echo "FOLD_PREFLIGHT: FAIL"
    exit 8
fi

echo "FOLD_PREFLIGHT: PASS"

echo
echo "===== ARRAY MAPPING ====="

for TASK_ID in 0 1 2 3 4 5 49 50 51 52 53 54; do

    OUTER=$(( TASK_ID / 5 + 1 ))
    INNER=$(( TASK_ID % 5 + 1 ))

    FOLD_ID="$(
        printf 'fold_%02d_inner_%02d' \
            "$OUTER" \
            "$INNER"
    )"

    echo \
        "task=$TASK_ID outer=$OUTER inner=$INNER fold=$FOLD_ID"
done

echo
echo "mapping_total=55"
echo "parallel_limit=11"

if [[ "$MODE" == "--dry-run" ]]; then
    echo
    echo "OUTPUT_CREATED: NO"
    echo "JOBS_SUBMITTED: NO"
    echo "DRY_RUN: PASS"
    exit 0
fi

echo
echo "===== OUTPUT PRECONDITION ====="

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: output root ya existe:"
    echo "$OUTPUT_ROOT"
    exit 9
fi

mkdir -p \
    "$OUTPUT_ROOT/folds" \
    "$OUTPUT_ROOT/logs" \
    "$OUTPUT_ROOT/status" \
    "$OUTPUT_ROOT/provenance"

PROVENANCE="$OUTPUT_ROOT/provenance"

cp \
    "$TRAINER" \
    "$PROVENANCE/train_gru_baseline.py"

cp \
    "$SBATCH_SOURCE" \
    "$PROVENANCE/gru_baseline_nested_array.sbatch"

cp \
    "$SUBMITTER_SOURCE" \
    "$PROVENANCE/submit_gru_baseline_nested.sh"

chmod 444 \
    "$PROVENANCE/train_gru_baseline.py" \
    "$PROVENANCE/gru_baseline_nested_array.sbatch" \
    "$PROVENANCE/submit_gru_baseline_nested.sh"

FROZEN_TRAINER="$PROVENANCE/train_gru_baseline.py"
FROZEN_SBATCH="$PROVENANCE/gru_baseline_nested_array.sbatch"

FROZEN_TRAINER_SHA256="$(
    sha256sum "$FROZEN_TRAINER" \
        | awk '{print $1}'
)"

if [[ "$FROZEN_TRAINER_SHA256" != "$TRAINER_SHA256" ]]; then
    echo "ERROR: frozen trainer SHA mismatch"
    exit 10
fi

{
    echo "run_id=$RUN_ID"
    echo "created_utc=$(date -u --iso-8601=seconds)"
    echo "git_head=$CURRENT_HEAD"
    echo "trainer_last_commit=$TRAINER_LAST_COMMIT"
    echo "trainer_sha256=$TRAINER_SHA256"
    echo "data_root=$DATA_ROOT"
    echo "array_spec=$ARRAY_SPEC"
    echo "seed=20260811"
    echo "hidden_size=32"
    echo "learning_rate=0.001"
    echo "weight_decay=0.0001"
    echo "max_epochs=2000"
    echo "patience=50"
    echo "min_delta=0.00001"
    echo "outer_test_used=false"
} > "$PROVENANCE/run_config.txt"

printf \
    'task_id\touter_fold\tinner_fold\tfold_id\tfold_npz\tfold_sha256\n' \
    > "$PROVENANCE/fold_manifest.tsv"

for TASK_ID in $(seq 0 54); do

    OUTER=$(( TASK_ID / 5 + 1 ))
    INNER=$(( TASK_ID % 5 + 1 ))

    FOLD_ID="$(
        printf 'fold_%02d_inner_%02d' \
            "$OUTER" \
            "$INNER"
    )"

    FOLD="$DATA_ROOT/folds/${FOLD_ID}.npz"

    FOLD_SHA="$(
        sha256sum "$FOLD" \
            | awk '{print $1}'
    )"

    printf \
        '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$TASK_ID" \
        "$OUTER" \
        "$INNER" \
        "$FOLD_ID" \
        "$FOLD" \
        "$FOLD_SHA" \
        >> "$PROVENANCE/fold_manifest.tsv"
done

(
    cd "$PROVENANCE"

    sha256sum \
        train_gru_baseline.py \
        gru_baseline_nested_array.sbatch \
        submit_gru_baseline_nested.sh \
        run_config.txt \
        fold_manifest.tsv \
        > SHA256SUMS_PRE_SUBMISSION.txt
)

JOB_ID="$(
    sbatch \
        --parsable \
        --array="$ARRAY_SPEC" \
        --output="$OUTPUT_ROOT/logs/%A_%a.out" \
        --error="$OUTPUT_ROOT/logs/%A_%a.err" \
        "$FROZEN_SBATCH" \
        "$DATA_ROOT" \
        "$OUTPUT_ROOT" \
        "$FROZEN_TRAINER" \
        "$FROZEN_TRAINER_SHA256"
)"

printf '%s\n' \
    "$JOB_ID" \
    > "$PROVENANCE/submission_job_id.txt"

echo
echo "===== SUBMISSION ====="
echo "job_id=$JOB_ID"
echo "output_root=$OUTPUT_ROOT"
echo "SUBMISSION: PASS"
