#!/bin/bash

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"

RUN_ID="atlas11_gwo_benchmark_v1_2026-08-12"
RUN_ROOT="$WORK/results/optimization/$RUN_ID"

SBATCH_FILE="$PROJECT/slurm/gwo_benchmark_v1_array.sbatch"

TRAINER="$PROJECT/src/modeling/train_gru_tunable.py"
SPACE="$PROJECT/configs/gwo_search_space_v1.json"
DECODER="$PROJECT/src/optimization/gwo_space.py"
ALGO="$PROJECT/configs/gwo_algorithm_v1.json"
ENGINE="$PROJECT/src/optimization/run_gwo_search.py"
SUBMITTER="$PROJECT/slurm/submit_gwo_benchmark_v1.sh"

EXPECTED_TRAINER_SHA="1201be34061d2374d6490700ad7f36967e516e98df049fc25179e71847a137a4"
EXPECTED_SPACE_SHA="18338c44d297a3f116998c06877e5f5ecfbb3aa2a04a44c5745e796f0f0473df"
EXPECTED_DECODER_SHA="007e1e9d55c1f25a311f2487ac0a55e2778a38ed1ce65446d172d50dccf5936d"
EXPECTED_ALGO_SHA="3bdae410576c2dfbdeffdd4cb683790c83e7ec8d5747678a33aeaf5460346076"
EXPECTED_ENGINE_SHA="8851e5ef4ae61b5dcb571d53170b2cef5b928188cbfaee7b14fac899e349fd9c"

cd "$PROJECT"

if [[ -e "$RUN_ROOT" ]]; then
    echo "RUN_ROOT already exists: $RUN_ROOT"
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Repository must be clean before submission."
    git status --short
    exit 1
fi

check_sha() {
    local expected="$1"
    local path="$2"
    local actual

    actual="$(
        sha256sum "$path" |
        awk '{print $1}'
    )"

    if [[ "$actual" != "$expected" ]]; then
        echo "SHA mismatch: $path"
        echo "expected=$expected"
        echo "actual=$actual"
        exit 1
    fi
}

check_sha "$EXPECTED_TRAINER_SHA" "$TRAINER"
check_sha "$EXPECTED_SPACE_SHA" "$SPACE"
check_sha "$EXPECTED_DECODER_SHA" "$DECODER"
check_sha "$EXPECTED_ALGO_SHA" "$ALGO"
check_sha "$EXPECTED_ENGINE_SHA" "$ENGINE"

mkdir -p \
    "$RUN_ROOT/logs" \
    "$RUN_ROOT/searches" \
    "$RUN_ROOT/provenance"

cp "$TRAINER" \
    "$RUN_ROOT/provenance/train_gru_tunable.py"

cp "$SPACE" \
    "$RUN_ROOT/provenance/gwo_search_space_v1.json"

cp "$DECODER" \
    "$RUN_ROOT/provenance/gwo_space.py"

cp "$ALGO" \
    "$RUN_ROOT/provenance/gwo_algorithm_v1.json"

cp "$ENGINE" \
    "$RUN_ROOT/provenance/run_gwo_search.py"

cp "$SBATCH_FILE" \
    "$RUN_ROOT/provenance/gwo_benchmark_v1_array.sbatch"

cp "$SUBMITTER" \
    "$RUN_ROOT/provenance/submit_gwo_benchmark_v1.sh"

MANIFEST="$RUN_ROOT/provenance/task_manifest.tsv"

printf \
    'task_id\touter_fold\tmethod\toptimizer_seed\n' \
    > "$MANIFEST"

METHODS=(
    standard_gwo
    improved_gwo
    improved_no_m1
    improved_no_m2
    improved_no_m3
)

SEEDS=(
    20260811
    20260812
    20260813
)

for OUTER in $(seq 1 11); do

    for METHOD_INDEX in $(seq 0 4); do

        METHOD="${METHODS[$METHOD_INDEX]}"

        for SEED_INDEX in $(seq 0 2); do

            SEED="${SEEDS[$SEED_INDEX]}"

            TASK_ID=$((
                (OUTER - 1) * 15
                + METHOD_INDEX * 3
                + SEED_INDEX
            ))

            printf \
                '%d\t%d\t%s\t%d\n' \
                "$TASK_ID" \
                "$OUTER" \
                "$METHOD" \
                "$SEED" \
                >> "$MANIFEST"

        done

    done

done

export MANIFEST

python3 - <<'PY'
from pathlib import Path
import csv
import os
import sys

path = Path(
    os.environ["MANIFEST"]
)

rows = list(
    csv.DictReader(
        path.open(
            encoding="utf-8"
        ),
        delimiter="\t",
    )
)

errors = []


def check(condition, label):
    print(
        label + ":",
        "PASS" if condition else "FAIL",
    )

    if not condition:
        errors.append(label)


check(
    len(rows) == 165,
    "MANIFEST_ROWS_165",
)

ids = [
    int(row["task_id"])
    for row in rows
]

check(
    sorted(ids)
    == list(range(165)),
    "TASK_IDS_0_164",
)

tuples = {
    (
        int(row["outer_fold"]),
        row["method"],
        int(row["optimizer_seed"]),
    )
    for row in rows
}

check(
    len(tuples) == 165,
    "UNIQUE_SEARCH_COMBINATIONS_165",
)

for row in rows:

    task = int(
        row["task_id"]
    )

    seed_index = task % 3
    tmp = task // 3
    method_index = tmp % 5
    outer_index = tmp // 5

    expected_outer = (
        outer_index + 1
    )

    if (
        expected_outer
        != int(
            row["outer_fold"]
        )
    ):
        errors.append(
            "reverse outer mapping"
        )
        break

check(
    "reverse outer mapping"
    not in errors,
    "REVERSE_TASK_MAPPING",
)

outer_counts = {}

method_counts = {}

seed_counts = {}

for row in rows:

    outer = int(
        row["outer_fold"]
    )

    method = row["method"]

    seed = int(
        row["optimizer_seed"]
    )

    outer_counts[outer] = (
        outer_counts.get(
            outer,
            0,
        )
        + 1
    )

    method_counts[method] = (
        method_counts.get(
            method,
            0,
        )
        + 1
    )

    seed_counts[seed] = (
        seed_counts.get(
            seed,
            0,
        )
        + 1
    )

check(
    set(
        outer_counts.values()
    ) == {15},
    "FIFTEEN_SEARCHES_PER_OUTER",
)

check(
    set(
        method_counts.values()
    ) == {33},
    "THIRTY_THREE_SEARCHES_PER_METHOD",
)

check(
    set(
        seed_counts.values()
    ) == {55},
    "FIFTY_FIVE_SEARCHES_PER_SEED",
)

print(
    "outer_counts:",
    outer_counts,
)

print(
    "method_counts:",
    method_counts,
)

print(
    "seed_counts:",
    seed_counts,
)

if errors:
    print(
        "TASK_MANIFEST_AUDIT: FAIL"
    )
    sys.exit(1)

print(
    "TASK_MANIFEST_AUDIT: PASS"
)
PY

GIT_HEAD="$(
    git rev-parse HEAD
)"

ENGINE_COMMIT="$(
    git log -1 \
        --format='%H' \
        -- "$ENGINE"
)"

cat > "$RUN_ROOT/provenance/run_config.txt" <<EOF
run_id=$RUN_ID
git_head=$GIT_HEAD
engine_commit=$ENGINE_COMMIT

array=0-164%11
n_array_tasks=165
max_concurrent_searches=11

partition=sapphire
account=uva_dma_2
qos=normal

cpus_per_task=5
workers_per_search=5
memory=8G
walltime=12:00:00

n_outer_contexts=11
n_methods=5
n_optimizer_seeds=3

fitness_evaluations_per_search=30
inner_folds_per_candidate=5
maximum_theoretical_fold_trainings=24750

outer_test_evaluated=false

trainer_sha256=$EXPECTED_TRAINER_SHA
search_space_sha256=$EXPECTED_SPACE_SHA
decoder_sha256=$EXPECTED_DECODER_SHA
algorithm_sha256=$EXPECTED_ALGO_SHA
engine_sha256=$EXPECTED_ENGINE_SHA
EOF

(
    cd "$RUN_ROOT/provenance"

    sha256sum \
        train_gru_tunable.py \
        gwo_search_space_v1.json \
        gwo_space.py \
        gwo_algorithm_v1.json \
        run_gwo_search.py \
        gwo_benchmark_v1_array.sbatch \
        submit_gwo_benchmark_v1.sh \
        task_manifest.tsv \
        run_config.txt \
        > SHA256SUMS.txt

    sha256sum -c SHA256SUMS.txt
)

SUBMIT_OUTPUT="$(
    sbatch \
        --parsable \
        --array=0-164%11 \
        --output="$RUN_ROOT/logs/slurm-%A_%a.out" \
        --error="$RUN_ROOT/logs/slurm-%A_%a.err" \
        "$SBATCH_FILE" \
        "$RUN_ROOT"
)"

JOB_ID="${SUBMIT_OUTPUT%%;*}"

cat > "$RUN_ROOT/provenance/submission.txt" <<EOF
job_id=$JOB_ID
sbatch_result=$SUBMIT_OUTPUT
array=0-164%11
run_root=$RUN_ROOT
EOF

echo "========================================"
echo "GWO BENCHMARK SUBMITTED"
echo "========================================"
echo "run_id=$RUN_ID"
echo "run_root=$RUN_ROOT"
echo "job_id=$JOB_ID"
echo "array=0-164%11"
echo "tasks=165"
echo "max_concurrent=11"
echo "outer_test_evaluated=false"
echo "SUBMISSION: PASS"
