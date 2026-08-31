#!/bin/bash
#SBATCH --job-name=atlas11-gwo-v2
#SBATCH --account=uva_dma_2
#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=12:00:00

set -euo pipefail

RUN_ROOT="${1:?RUN_ROOT required}"

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"

DATA_ROOT="$WORK/results/preprocessing/atlas11_nested_training_v1_2026-08-11"

TRAINER="$PROJECT/src/modeling/train_gru_tunable.py"
SPACE="$PROJECT/configs/gwo_search_space_v2.json"
DECODER="$PROJECT/src/optimization/gwo_space.py"
ALGO="$PROJECT/configs/gwo_algorithm_v1.json"
ENGINE="$PROJECT/src/optimization/run_gwo_search.py"

EXPECTED_TRAINER_SHA="1201be34061d2374d6490700ad7f36967e516e98df049fc25179e71847a137a4"
EXPECTED_SPACE_SHA="6699764b80a8bddaae0a071cfedcd65a4a072521ba95bcb10140cf26512717e1"
EXPECTED_DECODER_SHA="007e1e9d55c1f25a311f2487ac0a55e2778a38ed1ce65446d172d50dccf5936d"
EXPECTED_ALGO_SHA="3bdae410576c2dfbdeffdd4cb683790c83e7ec8d5747678a33aeaf5460346076"
EXPECTED_ENGINE_SHA="8851e5ef4ae61b5dcb571d53170b2cef5b928188cbfaee7b14fac899e349fd9c"

EXPECTED_ENGINE_COMMIT="5629fe5fdc672e9949478028aff22ab57c3d947a"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONNOUSERSITE=1

cd "$PROJECT"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/venvs/md-poc-py311/bin/activate"

check_sha() {
    local expected="$1"
    local path="$2"
    local actual

    actual="$(
        sha256sum "$path" |
        awk '{print $1}'
    )"

    if [[ "$actual" != "$expected" ]]; then
        echo "SHA MISMATCH: $path"
        echo "expected=$expected"
        echo "actual=$actual"
        return 1
    fi
}

check_sha \
    "$EXPECTED_TRAINER_SHA" \
    "$TRAINER"

check_sha \
    "$EXPECTED_SPACE_SHA" \
    "$SPACE"

check_sha \
    "$EXPECTED_DECODER_SHA" \
    "$DECODER"

check_sha \
    "$EXPECTED_ALGO_SHA" \
    "$ALGO"

check_sha \
    "$EXPECTED_ENGINE_SHA" \
    "$ENGINE"

ENGINE_COMMIT="$(
    git log -1 \
        --format='%H' \
        -- "$ENGINE"
)"

if [[ "$ENGINE_COMMIT" != "$EXPECTED_ENGINE_COMMIT" ]]; then
    echo "ENGINE COMMIT MISMATCH"
    echo "expected=$EXPECTED_ENGINE_COMMIT"
    echo "actual=$ENGINE_COMMIT"
    exit 1
fi


# --------------------------------------------------
# 33 tasks:
#
# 11 outer contexts x 3 optimizer seeds
#
# task 0  = outer01 seed20260811
# task 1  = outer01 seed20260812
# task 2  = outer01 seed20260813
# task 3  = outer02 seed20260811
# ...
# task 32 = outer11 seed20260813
# --------------------------------------------------

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"

if (( TASK_ID < 0 || TASK_ID > 32 )); then
    echo "Invalid task id: $TASK_ID"
    exit 1
fi

SEEDS=(
    20260811
    20260812
    20260813
)

SEED_INDEX=$((TASK_ID % 3))
OUTER_INDEX=$((TASK_ID / 3))

OUTER_FOLD=$((OUTER_INDEX + 1))
OPTIMIZER_SEED="${SEEDS[$SEED_INDEX]}"

METHOD="improved_gwo"

printf -v OUTER_PAD "%02d" "$OUTER_FOLD"

SEARCH_PARENT="$RUN_ROOT/searches/fold_${OUTER_PAD}/${METHOD}"
SEARCH_OUT="$SEARCH_PARENT/seed_${OPTIMIZER_SEED}"

mkdir -p "$SEARCH_PARENT"

echo "================================================================================"
echo "FINAL IMPROVED GWO V2"
echo "================================================================================"
echo "job_id=$SLURM_JOB_ID"
echo "array_task_id=$TASK_ID"
echo "hostname=$(hostname)"
echo "outer_fold=$OUTER_FOLD"
echo "method=$METHOD"
echo "optimizer_seed=$OPTIMIZER_SEED"
echo "search_out=$SEARCH_OUT"
echo "engine_commit=$ENGINE_COMMIT"

python --version

if [[ -d "$SEARCH_OUT" ]]; then

    if [[ -f "$SEARCH_OUT/SHA256SUMS.txt" ]]; then

        (
            cd "$SEARCH_OUT"
            sha256sum -c SHA256SUMS.txt
        )

        echo "SEARCH_ALREADY_COMPLETE: PASS"
        exit 0
    fi

    echo "INCOMPLETE_OUTPUT_EXISTS: $SEARCH_OUT"
    exit 1
fi

if [[ -d "${SEARCH_OUT}.work" ]]; then
    echo "STALE_WORK_DIRECTORY_EXISTS: ${SEARCH_OUT}.work"
    exit 1
fi


python \
    "$ENGINE" \
    --spec "$SPACE" \
    --algorithm-config "$ALGO" \
    --outer-fold "$OUTER_FOLD" \
    --method "$METHOD" \
    --optimizer-seed "$OPTIMIZER_SEED" \
    --data-root "$DATA_ROOT" \
    --trainer "$TRAINER" \
    --output-dir "$SEARCH_OUT" \
    --workers 5


(
    cd "$SEARCH_OUT"
    sha256sum -c SHA256SUMS.txt
)


export SEARCH_OUT
export OUTER_FOLD
export OPTIMIZER_SEED
export EXPECTED_SPACE_SHA


python - <<'PY'
from pathlib import Path
import json
import os
import sys

root = Path(
    os.environ[
        "SEARCH_OUT"
    ]
)

summary = json.loads(
    (
        root
        / "search_summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

config = json.loads(
    (
        root
        / "run_config.json"
    ).read_text(
        encoding="utf-8"
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
    summary[
        "evaluation_count"
    ] == 30,
    "EVALUATION_COUNT_30",
)

check(
    summary[
        "evaluation_budget"
    ] == 30,
    "EVALUATION_BUDGET_30",
)

check(
    summary[
        "method"
    ] == "improved_gwo",
    "METHOD_IMPROVED_GWO",
)

check(
    summary[
        "optimizer_seed"
    ] == int(
        os.environ[
            "OPTIMIZER_SEED"
        ]
    ),
    "SEED_MATCH",
)

check(
    config[
        "outer_fold"
    ] == int(
        os.environ[
            "OUTER_FOLD"
        ]
    ),
    "OUTER_FOLD_MATCH",
)

check(
    config[
        "spec_sha256"
    ] == os.environ[
        "EXPECTED_SPACE_SHA"
    ],
    "V2_SPACE_HASH_MATCH",
)

check(
    config[
        "outer_test_evaluated"
    ] is False,
    "OUTER_TEST_EVALUATED_NO",
)

check(
    len(
        summary[
            "best"
        ][
            "fold_losses"
        ]
    ) == 5,
    "BEST_HAS_FIVE_INNER_FOLDS",
)

if errors:
    print(
        "FINAL_GWO_TASK_AUDIT: FAIL"
    )
    sys.exit(1)

print(
    "FINAL_GWO_TASK_AUDIT: PASS"
)
PY

echo "FINAL_IMPROVED_GWO_TASK: PASS"
