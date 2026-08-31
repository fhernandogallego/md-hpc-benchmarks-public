#!/usr/bin/env bash

set -uo pipefail

PROJECT="${PROJECT:-$HOME/md-hpc-benchmarks}"

if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERROR: SCRATCH no está definido." >&2
    exit 1
fi

WORK="${WORK:-$SCRATCH/md-hpc-benchmarks}"

MANIFEST="$PROJECT/manifests/atlas_systems.csv"
ORCHESTRATOR="$PROJECT/slurm/submit_atlas_system_pipeline.sh"

DATASET_GROUP=""
RUN_ID=""
DRY_RUN=0

usage() {
    cat <<'EOF'
Uso:
  submit_atlas_batch.sh \
    --dataset-group GROUP \
    --run-id RUN_ID \
    [--dry-run]

Selecciona del manifiesto los sistemas:
  enabled=true
  dataset_group=GROUP
  status=staged

y ejecuta submit_atlas_system_pipeline.sh para cada uno.
EOF
}

while [[ "$#" -gt 0 ]]
do
    case "$1" in
        --dataset-group)
            DATASET_GROUP="${2:-}"
            shift 2
            ;;
        --run-id)
            RUN_ID="${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: argumento desconocido: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$DATASET_GROUP" ]]; then
    echo "ERROR: falta --dataset-group." >&2
    exit 1
fi

if [[ -z "$RUN_ID" ]]; then
    echo "ERROR: falta --run-id." >&2
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: no existe el manifiesto: $MANIFEST" >&2
    exit 1
fi

if [[ ! -x "$ORCHESTRATOR" ]]; then
    echo "ERROR: no existe o no es ejecutable: $ORCHESTRATOR" >&2
    exit 1
fi

mapfile -t SYSTEMS < <(
    awk -F',' -v grp="$DATASET_GROUP" '
        NR > 1 &&
        $2 == "true" &&
        $4 == grp &&
        $5 == "staged" {
            print $1
        }
    ' "$MANIFEST"
)

if [[ "${#SYSTEMS[@]}" -eq 0 ]]; then
    echo "ERROR: no hay sistemas staged para $DATASET_GROUP." >&2
    exit 1
fi

SUBMISSION_ROOT="$WORK/results/atlas_batch_submissions/$DATASET_GROUP/$RUN_ID"

mkdir -p "$SUBMISSION_ROOT/logs"

cp -a \
  "$MANIFEST" \
  "$SUBMISSION_ROOT/manifest_at_submission.csv"

printf '%s\n' "${SYSTEMS[@]}" \
  > "$SUBMISSION_ROOT/systems.txt"

sha256sum \
  "$ORCHESTRATOR" \
  "$PROJECT/slurm/atlas_pipeline_stage.sbatch" \
  > "$SUBMISSION_ROOT/orchestrator_code.sha256"

SUMMARY="$SUBMISSION_ROOT/submission_summary.csv"

echo "system,status,run_root,log" > "$SUMMARY"

echo "================================================================================"
echo "ATLAS BATCH SUBMISSION"
echo "================================================================================"
echo "Dataset group: $DATASET_GROUP"
echo "Run ID:        $RUN_ID"
echo "Dry run:       $DRY_RUN"
echo "Systems:       ${#SYSTEMS[@]}"
echo

GLOBAL_STATUS=0

for SYSTEM in "${SYSTEMS[@]}"
do
    LOG="$SUBMISSION_ROOT/logs/${SYSTEM}.log"

    RUN_ROOT="$WORK/results/atlas_pipeline_runs/$DATASET_GROUP/$SYSTEM/$RUN_ID"

    CMD=(
        bash
        "$ORCHESTRATOR"
        --system "$SYSTEM"
        --dataset-group "$DATASET_GROUP"
        --run-id "$RUN_ID"
    )

    if [[ "$DRY_RUN" -eq 1 ]]; then
        CMD+=(--dry-run)
    fi

    echo "--------------------------------------------------------------------------------"
    echo "SYSTEM: $SYSTEM"

    "${CMD[@]}" > "$LOG" 2>&1
    STATUS=$?

    if [[ "$STATUS" -eq 0 ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            LABEL="DRY_RUN_PASS"
        else
            LABEL="SUBMITTED"
        fi

        echo "RESULTADO: $LABEL"

        printf '%s,%s,%s,%s\n' \
          "$SYSTEM" \
          "$LABEL" \
          "$RUN_ROOT" \
          "$LOG" \
          >> "$SUMMARY"
    else
        echo "RESULTADO: FAIL"
        echo
        echo "Últimas líneas:"
        tail -n 40 "$LOG"

        printf '%s,%s,%s,%s\n' \
          "$SYSTEM" \
          "FAIL" \
          "$RUN_ROOT" \
          "$LOG" \
          >> "$SUMMARY"

        GLOBAL_STATUS=1
    fi
done

echo
echo "================================================================================"
echo "RESUMEN"
echo "================================================================================"

cat "$SUMMARY"

echo
if [[ "$GLOBAL_STATUS" -eq 0 ]]; then
    echo "ATLAS BATCH SUBMISSION: PASS"
else
    echo "ATLAS BATCH SUBMISSION: FAIL"
fi

exit "$GLOBAL_STATUS"
