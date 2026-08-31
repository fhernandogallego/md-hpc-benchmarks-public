#!/bin/bash
#SBATCH --job-name=atlas11-base-final
#SBATCH --account=uva_dma_2
#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"

PREPROC="$WORK/results/preprocessing/atlas11_outer_training_v1_2026-08-17"

PLAN="$WORK/results/analysis/atlas11_gru_baseline_final_epoch_plan_v1_2026-08-17/final_epoch_plan.csv"

OUTPUT_ROOT="$WORK/results/modeling/atlas11_gru_baseline_final_outer_v1_2026-08-17"

BASELINE_TRAINER="$PROJECT/src/modeling/train_gru_baseline.py"

TRAINER="$PROJECT/src/modeling/train_gru_baseline_final_outer.py"

EXPECTED_BASELINE_SHA="b680e82822c39745c0bfbe3734f82129e83332fc9eb6004e4a236415965a045f"
EXPECTED_FINAL_TRAINER_SHA="d02b49d1eeb963e2f4ef1e1c7239a62947b08115545945bea41b9dde357c65be"

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
    local file="$2"
    local actual

    actual="$(
        sha256sum "$file" |
        awk '{print $1}'
    )"

    if [[ "$actual" != "$expected" ]]; then
        echo "SHA MISMATCH: $file"
        echo "expected=$expected"
        echo "actual=$actual"
        return 1
    fi
}

check_sha     "$EXPECTED_BASELINE_SHA"     "$BASELINE_TRAINER"

check_sha     "$EXPECTED_FINAL_TRAINER_SHA"     "$TRAINER"

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"

if (( TASK_ID < 0 || TASK_ID > 10 )); then
    echo "INVALID TASK: $TASK_ID"
    false
fi

OUTER=$((TASK_ID + 1))

printf -v PAD "%02d" "$OUTER"

PACKAGE="$PREPROC/folds/fold_${PAD}.npz"

OUTPUT="$OUTPUT_ROOT/outer_${PAD}"

python     "$TRAINER"     --outer-fold "$OUTER"     --package "$PACKAGE"     --epoch-plan "$PLAN"     --output-dir "$OUTPUT"

(
    cd "$OUTPUT"
    sha256sum -c SHA256SUMS.txt
)

echo "FINAL_BASELINE_ARRAY_TASK: PASS"
