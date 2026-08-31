#!/bin/bash
#SBATCH --job-name=atlas11-final-epochs
#SBATCH --account=uva_dma_2
#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"

DATA_ROOT="$WORK/results/preprocessing/atlas11_nested_training_v1_2026-08-11"

SELECTED="$WORK/results/analysis/atlas11_final_gwo_v3_selection_v1_2026-08-17/final_selected_models.csv"

OUTPUT_ROOT="$WORK/results/analysis/atlas11_final_epoch_derivation_v1_2026-08-17"

SPEC="$PROJECT/configs/gwo_search_space_v3.json"
TRAINER="$PROJECT/src/modeling/train_gru_tunable.py"
RUNNER="$PROJECT/src/analysis/derive_final_epoch_plan.py"

EXPECTED_SELECTED_SHA="b2f1c163bc9fb6af2159509728ac2b4cabac6c2323db9bbff0f742c7375fe0c7"
EXPECTED_RUNNER_SHA="583883c92ee6440696dbacf9c65efa5bbe728869bde396de509c204394da82b2"
EXPECTED_SPEC_SHA="ef1ab63caffaed81615a4232f643748096b76bfa3301351463fd97f5e714e3d0"
EXPECTED_TRAINER_SHA="1201be34061d2374d6490700ad7f36967e516e98df049fc25179e71847a137a4"

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

check_sha "$EXPECTED_SELECTED_SHA" "$SELECTED"
check_sha "$EXPECTED_RUNNER_SHA" "$RUNNER"
check_sha "$EXPECTED_SPEC_SHA" "$SPEC"
check_sha "$EXPECTED_TRAINER_SHA" "$TRAINER"

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"

if (( TASK_ID < 0 || TASK_ID > 10 )); then
    echo "Invalid task: $TASK_ID"
    exit 1
fi

OUTER=$((TASK_ID + 1))

printf -v OUTER_PAD "%02d" "$OUTER"

OUTPUT="$OUTPUT_ROOT/outer_${OUTER_PAD}"

if [[ -e "$OUTPUT" || -e "${OUTPUT}.work" ]]; then
    echo "OUTPUT ALREADY EXISTS: $OUTPUT"
    exit 1
fi

python     "$RUNNER"     --project "$PROJECT"     --data-root "$DATA_ROOT"     --selected-csv "$SELECTED"     --spec "$SPEC"     --trainer "$TRAINER"     --outer-fold "$OUTER"     --output-dir "$OUTPUT"     --workers 5

(
    cd "$OUTPUT"
    sha256sum -c SHA256SUMS.txt
)

echo "FINAL_EPOCH_ARRAY_TASK: PASS"
