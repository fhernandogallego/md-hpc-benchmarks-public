#!/bin/bash
#SBATCH --job-name=gwo-v3-o08-s12-retry
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

SPACE="$PROJECT/configs/gwo_search_space_v3.json"
ALGO="$PROJECT/configs/gwo_algorithm_v1.json"
TRAINER="$PROJECT/src/modeling/train_gru_tunable.py"

BASE_ENGINE="$PROJECT/src/optimization/run_gwo_search.py"
ENGINE="$PROJECT/src/optimization/run_gwo_search_robust.py"
POLICY="$PROJECT/configs/gwo_numerical_failure_policy_v1.json"

EXPECTED_SPACE_SHA="ef1ab63caffaed81615a4232f643748096b76bfa3301351463fd97f5e714e3d0"
EXPECTED_ALGO_SHA="3bdae410576c2dfbdeffdd4cb683790c83e7ec8d5747678a33aeaf5460346076"
EXPECTED_TRAINER_SHA="1201be34061d2374d6490700ad7f36967e516e98df049fc25179e71847a137a4"
EXPECTED_BASE_ENGINE_SHA="8851e5ef4ae61b5dcb571d53170b2cef5b928188cbfaee7b14fac899e349fd9c"
EXPECTED_ROBUST_ENGINE_SHA="fe2e2b85e40c2b9eea9c676b3ec3f90e9d300501feede224acb3c776b345f9cb"
EXPECTED_POLICY_SHA="21a7888bcfe9b6b01c9673bf4cf5c6234fb2c3c9a11079e0509396fa73deb8cd"

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
        exit 1
    fi
}

check_sha "$EXPECTED_SPACE_SHA" "$SPACE"
check_sha "$EXPECTED_ALGO_SHA" "$ALGO"
check_sha "$EXPECTED_TRAINER_SHA" "$TRAINER"
check_sha "$EXPECTED_BASE_ENGINE_SHA" "$BASE_ENGINE"
check_sha "$EXPECTED_ROBUST_ENGINE_SHA" "$ENGINE"
check_sha "$EXPECTED_POLICY_SHA" "$POLICY"

OUTER_FOLD=8
OPTIMIZER_SEED=20260812
METHOD="improved_gwo"

SEARCH_OUT="$RUN_ROOT/searches/fold_08/improved_gwo/seed_20260812"

if [[ -e "$SEARCH_OUT" ]]; then
    echo "FINAL SEARCH ALREADY EXISTS"
    exit 1
fi

if [[ -e "${SEARCH_OUT}.work" ]]; then
    echo "WORK SEARCH ALREADY EXISTS"
    exit 1
fi

echo "================================================================================"
echo "GWO V3 ROBUST RETRY"
echo "================================================================================"
echo "outer_fold=$OUTER_FOLD"
echo "optimizer_seed=$OPTIMIZER_SEED"
echo "method=$METHOD"
echo "numerical_failure_fold_penalty=1000000.0"
echo "outer_test_evaluated=NO"

python     "$ENGINE"     --spec "$SPACE"     --algorithm-config "$ALGO"     --outer-fold "$OUTER_FOLD"     --method "$METHOD"     --optimizer-seed "$OPTIMIZER_SEED"     --data-root "$DATA_ROOT"     --trainer "$TRAINER"     --output-dir "$SEARCH_OUT"     --workers 5

(
    cd "$SEARCH_OUT"
    sha256sum -c SHA256SUMS.txt
)

export SEARCH_OUT

python3 - <<'PYAUDIT'
from pathlib import Path
import json
import os
import sys

root = Path(
    os.environ["SEARCH_OUT"]
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
    summary["evaluation_count"] == 30,
    "EVALUATION_COUNT_30",
)

check(
    summary["evaluation_budget"] == 30,
    "EVALUATION_BUDGET_30",
)

check(
    summary["method"] == "improved_gwo",
    "METHOD_MATCH",
)

check(
    summary["optimizer_seed"] == 20260812,
    "SEED_MATCH",
)

check(
    config["outer_fold"] == 8,
    "OUTER_MATCH",
)

check(
    config["outer_test_evaluated"] is False,
    "OUTER_TEST_EVALUATED_NO",
)

check(
    len(
        summary["best"]["fold_losses"]
    ) == 5,
    "BEST_HAS_FIVE_INNER_FOLDS",
)

if errors:
    print(
        "ROBUST_RETRY_AUDIT: FAIL"
    )
    sys.exit(1)

print(
    "best_fitness:",
    summary["best"]["fitness"],
)

print(
    "best_candidate:",
    summary["best"]["candidate"],
)

print(
    "ROBUST_RETRY_AUDIT: PASS"
)
PYAUDIT

echo "GWO_V3_ROBUST_RETRY: PASS"
