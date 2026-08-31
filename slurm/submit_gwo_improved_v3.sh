#!/bin/bash

set -euo pipefail

RUN_ROOT="${1:?Usage: submit_gwo_improved_v3.sh RUN_ROOT}"

PROJECT="$HOME/md-hpc-benchmarks"

SBATCH="$PROJECT/slurm/gwo_improved_v3_array.sbatch"

if [[ -e "$RUN_ROOT" ]]; then
    echo "RUN_ROOT already exists: $RUN_ROOT"
    exit 1
fi

mkdir -p \
    "$RUN_ROOT/logs" \
    "$RUN_ROOT/provenance"


MANIFEST="$RUN_ROOT/provenance/task_manifest.tsv"

printf \
    'task_id\touter_fold\tmethod\toptimizer_seed\n' \
    > "$MANIFEST"

SEEDS=(
    20260811
    20260812
    20260813
)

for TASK_ID in $(seq 0 32); do

    SEED_INDEX=$((TASK_ID % 3))
    OUTER_INDEX=$((TASK_ID / 3))

    OUTER_FOLD=$((OUTER_INDEX + 1))

    SEED="${SEEDS[$SEED_INDEX]}"

    printf \
        '%d\t%d\t%s\t%d\n' \
        "$TASK_ID" \
        "$OUTER_FOLD" \
        "improved_gwo" \
        "$SEED" \
        >> "$MANIFEST"

done


cp \
    "$SBATCH" \
    "$RUN_ROOT/provenance/"

cp \
    "$PROJECT/configs/gwo_search_space_v3.json" \
    "$RUN_ROOT/provenance/"

cp \
    "$PROJECT/configs/gwo_algorithm_v1.json" \
    "$RUN_ROOT/provenance/"


cat > "$RUN_ROOT/provenance/run_config.txt" <<EOF
analysis=final_improved_gwo_v3_final_space
git_commit=$(git -C "$PROJECT" rev-parse HEAD)

method=improved_gwo
outer_contexts=11
optimizer_seeds=20260811,20260812,20260813
n_searches=33

fitness_evaluations_per_search=30
inner_folds_per_candidate=5
maximum_fold_trainings_without_cache=4950

array=0-32%11
partition=genoa
cpus_per_task=5
workers_per_search=5
memory=8G
walltime=12:00:00

outer_test_evaluated=false
EOF


(
    cd "$RUN_ROOT/provenance"

    sha256sum \
        gwo_improved_v3_array.sbatch \
        gwo_search_space_v3.json \
        gwo_algorithm_v1.json \
        task_manifest.tsv \
        run_config.txt \
        > SHA256SUMS.txt

    sha256sum -c SHA256SUMS.txt
)


SUBMIT="$(
    sbatch \
        --parsable \
        --array=0-32%11 \
        --output="$RUN_ROOT/logs/slurm-%A_%a.out" \
        --error="$RUN_ROOT/logs/slurm-%A_%a.err" \
        "$SBATCH" \
        "$RUN_ROOT"
)"

JOB_ID="${SUBMIT%%;*}"

cat > "$RUN_ROOT/provenance/submission.txt" <<EOF
job_id=$JOB_ID
sbatch_result=$SUBMIT
array=0-32%11
partition=genoa
run_root=$RUN_ROOT
outer_test_evaluated=false
EOF

echo "sbatch_result=$SUBMIT"
echo "job_id=$JOB_ID"
echo "run_root=$RUN_ROOT"
