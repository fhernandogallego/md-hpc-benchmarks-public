#!/bin/bash
#SBATCH --job-name=hidden-boundary
#SBATCH --account=uva_dma_2
#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

RUN_ROOT="${1:?RUN_ROOT required}"

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"

DATA_ROOT="$WORK/results/preprocessing/atlas11_nested_training_v1_2026-08-11"

SELECTED="$WORK/results/analysis/atlas11_gwo_benchmark_inner_v2_2026-08-13/improved_selected_by_outer.csv"

SPEC="$PROJECT/configs/gwo_search_space_v1.json"
TRAINER="$PROJECT/src/modeling/train_gru_tunable.py"
RUNNER="$PROJECT/src/analysis/run_hidden_boundary_sensitivity.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONNOUSERSITE=1

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "$HOME/venvs/md-poc-py311/bin/activate"

OUTER_FOLD=$((SLURM_ARRAY_TASK_ID + 1))

printf -v OUTER_PAD "%02d" "$OUTER_FOLD"

OUT="$RUN_ROOT/outer_${OUTER_PAD}"

echo "========================================"
echo "HIDDEN BOUNDARY ARRAY TASK"
echo "========================================"
echo "job_id=$SLURM_JOB_ID"
echo "task_id=$SLURM_ARRAY_TASK_ID"
echo "outer_fold=$OUTER_FOLD"
echo "hostname=$(hostname)"
echo "output=$OUT"

if [[ -d "$OUT" ]]; then

    if [[ -f "$OUT/SHA256SUMS.txt" ]]; then
        (
            cd "$OUT"
            sha256sum -c SHA256SUMS.txt
        )

        echo "OUTER_ALREADY_COMPLETE: PASS"
        exit 0
    fi

    echo "INCOMPLETE_OUTPUT_EXISTS: $OUT"
    exit 1
fi

if [[ -d "${OUT}.work" ]]; then
    echo "STALE_WORK_EXISTS: ${OUT}.work"
    exit 1
fi

python \
    "$RUNNER" \
    --project "$PROJECT" \
    --data-root "$DATA_ROOT" \
    --selected-csv "$SELECTED" \
    --spec "$SPEC" \
    --trainer "$TRAINER" \
    --outer-fold "$OUTER_FOLD" \
    --output-dir "$OUT" \
    --workers 5

(
    cd "$OUT"
    sha256sum -c SHA256SUMS.txt
)

python - "$OUT" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])

summary = json.loads(
    (
        root / "summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

assert (
    summary[
        "outer_test_evaluated"
    ]
    is False
)

assert (
    summary[
        "hidden_sizes_evaluated"
    ]
    == [96, 128, 160]
)

assert len(
    summary[
        "candidates"
    ]
) == 3

for candidate in summary[
    "candidates"
]:
    assert len(
        candidate[
            "fold_losses"
        ]
    ) == 5

    assert all(
        record[
            "outer_test_used"
        ]
        is False
        for record
        in candidate[
            "fold_records"
        ]
    )

print(
    "OUTER_SENSITIVITY_AUDIT: PASS"
)
PY

echo "HIDDEN_BOUNDARY_TASK: PASS"
