#!/bin/bash
#SBATCH --job-name=atlas11-final-gru
#SBATCH --account=uva_dma_2
#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"

PREPROC="$WORK/results/preprocessing/atlas11_outer_training_v1_2026-08-17"

EPOCH_PLAN="$WORK/results/analysis/atlas11_final_epoch_derivation_v1_2026-08-17/final_epoch_plan.csv"

SELECTED="$WORK/results/analysis/atlas11_final_gwo_v3_selection_v1_2026-08-17/final_selected_models.csv"

TUNABLE="$PROJECT/src/modeling/train_gru_tunable.py"

TRAINER="$PROJECT/src/modeling/train_gru_final_outer.py"

OUTPUT_ROOT="$WORK/results/modeling/atlas11_gru_final_outer_v1_2026-08-17"

EXPECTED_TUNABLE_SHA="1201be34061d2374d6490700ad7f36967e516e98df049fc25179e71847a137a4"

EXPECTED_TRAINER_SHA="a2c724a79fd0774ea602ed2bd5536c18e778c00c4667658e26f42bbda2c171b6"

EXPECTED_EPOCH_SHA="ef1e6ec3201b54b59928092fa6c68f5b3971984385b8e45b5a88acb125b88a11"

EXPECTED_SELECTED_SHA="b2f1c163bc9fb6af2159509728ac2b4cabac6c2323db9bbff0f742c7375fe0c7"

EXPECTED_MANIFEST_SHA="3f9ed24195fd97c060ca478d3ee326cc2306f2a8b8db7772151c745cbbe45576"

EXPECTED_SCALERS_SHA="3981aad4c13e7ac821c09c2961aa6720f588da7296023cf84c44766aee3d2ecd"

EXPECTED_PREPROC_META_SHA="0e0b53ec20856bcf8063d2dad9b1705d3c947fd6a78e1147233cfb9075756877"

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

check_sha     "$EXPECTED_TUNABLE_SHA"     "$TUNABLE"

check_sha     "$EXPECTED_TRAINER_SHA"     "$TRAINER"

check_sha     "$EXPECTED_EPOCH_SHA"     "$EPOCH_PLAN"

check_sha     "$EXPECTED_SELECTED_SHA"     "$SELECTED"

check_sha     "$EXPECTED_MANIFEST_SHA"     "$PREPROC/outer_training_manifest.csv"

check_sha     "$EXPECTED_SCALERS_SHA"     "$PREPROC/outer_scalers.json"

check_sha     "$EXPECTED_PREPROC_META_SHA"     "$PREPROC/metadata.json"

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"

if (( TASK_ID < 0 || TASK_ID > 10 )); then
    echo "INVALID TASK ID: $TASK_ID"
    false
fi

OUTER=$((TASK_ID + 1))

printf -v OUTER_PAD "%02d" "$OUTER"

PACKAGE="$PREPROC/folds/fold_${OUTER_PAD}.npz"

OUTPUT="$OUTPUT_ROOT/outer_${OUTER_PAD}"

if [[ -e "$OUTPUT"       || -e "${OUTPUT}.work" ]]; then

    echo "OUTPUT ALREADY EXISTS: $OUTPUT"
    false
fi

echo "================================================================================"
echo "FINAL GRU OUTER $OUTER_PAD"
echo "================================================================================"

python     "$TRAINER"     --outer-fold "$OUTER"     --package "$PACKAGE"     --epoch-plan "$EPOCH_PLAN"     --selected-models "$SELECTED"     --output-dir "$OUTPUT"

(
    cd "$OUTPUT"
    sha256sum -c SHA256SUMS.txt
)

export OUTPUT

python3 - <<'PYAUDIT'
from pathlib import Path
import json
import os

root = Path(
    os.environ["OUTPUT"]
)

metadata = json.loads(
    (
        root
        / "run_metadata.json"
    ).read_text(
        encoding="utf-8"
    )
)

assert (
    metadata["training_samples"]
    == 30
)

assert (
    metadata["training_systems"]
    == 10
)

assert (
    metadata["epochs_completed"]
    == metadata["epochs_planned"]
)

assert (
    metadata["validation_used"]
    is False
)

assert (
    metadata["early_stopping_used"]
    is False
)

assert (
    metadata["outer_test_targets_used"]
    is False
)

assert (
    metadata["outer_test_evaluated"]
    is False
)

assert (
    root
    / "final_model.pt"
).is_file()

print(
    "EXACT_EPOCH_COUNT: PASS"
)

print(
    "NO_VALIDATION: PASS"
)

print(
    "NO_EARLY_STOPPING: PASS"
)

print(
    "OUTER_TEST_EVALUATED: NO"
)

print(
    "FINAL_MODEL_CHECKPOINT: PASS"
)
PYAUDIT

echo "FINAL_ARRAY_TASK: PASS"
