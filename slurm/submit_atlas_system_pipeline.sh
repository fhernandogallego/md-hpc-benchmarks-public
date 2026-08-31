#!/bin/bash

set -euo pipefail

PROJECT="$HOME/md-hpc-benchmarks"
WORK="$SCRATCH/md-hpc-benchmarks"

WORKER="$PROJECT/slurm/atlas_pipeline_stage.sbatch"
MANIFEST="$PROJECT/manifests/atlas_systems.csv"

SYSTEM=""
DATASET_GROUP=""
RUN_ID=""
DRY_RUN=false

usage() {
    cat <<EOF
Uso:

  bash $0 \
    --system SYSTEM \
    --dataset-group DATASET_GROUP \
    [--run-id RUN_ID] \
    [--dry-run]

Ejemplo:

  bash $0 \
    --system 1bzy_A \
    --dataset-group pilot_batch_v1 \
    --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system)
            SYSTEM="$2"
            shift 2
            ;;

        --dataset-group)
            DATASET_GROUP="$2"
            shift 2
            ;;

        --run-id)
            RUN_ID="$2"
            shift 2
            ;;

        --dry-run)
            DRY_RUN=true
            shift
            ;;

        -h|--help)
            usage
            shift
            ;;

        *)
            echo "ERROR: argumento desconocido: $1" >&2
            usage
            false
            ;;
    esac
done

if [[ -z "$SYSTEM" ]]; then
    echo "ERROR: falta --system" >&2
    false
fi

if [[ -z "$DATASET_GROUP" ]]; then
    echo "ERROR: falta --dataset-group" >&2
    false
fi

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(date +"%Y%m%d_%H%M%S")"
fi

if [[ ! -f "$WORKER" ]]; then
    echo "ERROR: no existe worker:"
    echo "$WORKER"
    false
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: no existe manifiesto:"
    echo "$MANIFEST"
    false
fi

RAW_SYSTEM_ROOT="$WORK/raw/atlas/$DATASET_GROUP/$SYSTEM"
UNPACKED_DIR="$RAW_SYSTEM_ROOT/unpacked"
ANALYSIS_DIR="$UNPACKED_DIR/${SYSTEM}_analysis"
PROTEIN_DIR="$UNPACKED_DIR/${SYSTEM}_protein"

RUN_ROOT="$WORK/results/atlas_pipeline_runs/$DATASET_GROUP/$SYSTEM/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"

echo "================================================================================"
echo "ATLAS SYSTEM PIPELINE ORCHESTRATOR"
echo "================================================================================"
echo "System:         $SYSTEM"
echo "Dataset group:  $DATASET_GROUP"
echo "Run ID:         $RUN_ID"
echo "Run root:       $RUN_ROOT"
echo "Dry run:        $DRY_RUN"
echo

if [[ ! -d "$UNPACKED_DIR" ]]; then
    echo "ERROR: falta unpacked:"
    echo "$UNPACKED_DIR"
    false
fi

if [[ ! -d "$ANALYSIS_DIR" ]]; then
    echo "ERROR: falta analysis dir:"
    echo "$ANALYSIS_DIR"
    false
fi

if [[ ! -d "$PROTEIN_DIR" ]]; then
    echo "ERROR: falta protein dir:"
    echo "$PROTEIN_DIR"
    false
fi

REQUIRED_OFFICIAL=(
  "$ANALYSIS_DIR/${SYSTEM}_RMSD.tsv"
  "$ANALYSIS_DIR/${SYSTEM}_gyrate.tsv"
  "$ANALYSIS_DIR/${SYSTEM}_RMSF.tsv"
)

for FILE in "${REQUIRED_OFFICIAL[@]}"; do
    if [[ ! -f "$FILE" ]]; then
        echo "ERROR: falta fichero oficial ATLAS:"
        echo "$FILE"
        false
    fi
done

MANIFEST_ROWS="$(
    awk \
      -F',' \
      -v sys="$SYSTEM" \
      '$1 == sys {print}' \
      "$MANIFEST"
)"

if [[ -z "$MANIFEST_ROWS" ]]; then
    echo "ERROR: $SYSTEM no aparece en:"
    echo "$MANIFEST"
    false
fi

N_MANIFEST_ROWS="$(
    printf '%s\n' "$MANIFEST_ROWS" \
    | wc -l
)"

if [[ "$N_MANIFEST_ROWS" -ne 1 ]]; then
    echo "ERROR: $SYSTEM aparece $N_MANIFEST_ROWS veces en el manifiesto."
    false
fi

MANIFEST_GROUP="$(
    printf '%s\n' "$MANIFEST_ROWS" \
    | awk -F',' '{print $4}'
)"

MANIFEST_ENABLED="$(
    printf '%s\n' "$MANIFEST_ROWS" \
    | awk -F',' '{print $2}'
)"

MANIFEST_REPLICAS="$(
    printf '%s\n' "$MANIFEST_ROWS" \
    | awk -F',' '{print $3}'
)"

if [[ "$MANIFEST_GROUP" != "$DATASET_GROUP" ]]; then
    echo "ERROR: dataset_group no coincide con manifiesto."
    echo "Solicitado: $DATASET_GROUP"
    echo "Manifest:   $MANIFEST_GROUP"
    false
fi

if [[ "$MANIFEST_ENABLED" != "true" ]]; then
    echo "ERROR: sistema no habilitado en manifiesto."
    false
fi

if [[ "$MANIFEST_REPLICAS" != "3" ]]; then
    echo "ERROR: se esperaban 3 réplicas."
    echo "Manifest: $MANIFEST_REPLICAS"
    false
fi

echo "Pre-submit checks: PASS"
echo

if [[ "$DRY_RUN" == "true" ]]; then
    echo "================================================================================"
    echo "PLAN DE EJECUCIÓN"
    echo "================================================================================"
    cat <<EOF

preflight
   |
   +--> basic[1-3] ---------> basic_official
   |        |
   |        +--------------> sensitivity --> basic_targets
   |
   +--> rmsf --------------> rmsf_official
   |        |
   |        +--------------> core -----------+
   |                                           |
   +--> sasa_dssp --> structural_targets       |
                         |                     |
basic_targets -----------+                     |
          |                                     |
          +--> master_targets                   |
                    |                           |
                    +---------------------------+--> core_rmsd
                    |                               |
                    +-------------------------------+
                                                    |
                                              prefix
                                                    |
                                                  audit
                                                    |
                          basic_official ------------+
                          rmsf_official --------------+
                                                    |
                                                finalize

EOF

    echo "DRY RUN: PASS"
    echo "No se ha enviado ningún trabajo a Slurm."
else
    mkdir -p \
      "$LOG_DIR"

    export_common="ALL,SYSTEM=$SYSTEM,DATASET_GROUP=$DATASET_GROUP,RUN_ROOT=$RUN_ROOT"

    submit_job() {
        local stage="$1"
        local dependency="$2"
        local time_limit="$3"
        local memory="$4"
        local cpus="$5"
        local array_spec="${6:-}"

        local args=(
          --parsable
          --job-name="atlas_${stage}_${SYSTEM}"
          --time="$time_limit"
          --mem="$memory"
          --cpus-per-task="$cpus"
          --export="$export_common,STAGE=$stage"
          --output="$LOG_DIR/%x_%j.out"
          --error="$LOG_DIR/%x_%j.err"
        )

        if [[ -n "$dependency" ]]; then
            args+=(
              --dependency="afterok:$dependency"
            )
        fi

        if [[ -n "$array_spec" ]]; then
            args+=(
              --array="$array_spec"
            )
        fi

        local raw_id

        raw_id="$(
            sbatch \
              "${args[@]}" \
              "$WORKER"
        )"

        printf '%s\n' \
          "${raw_id%%;*}"
    }

    PREFLIGHT_ID="$(
        submit_job \
          preflight \
          "" \
          "00:45:00" \
          "4G" \
          "1"
    )"

    BASIC_ID="$(
        submit_job \
          basic \
          "$PREFLIGHT_ID" \
          "00:30:00" \
          "3G" \
          "1" \
          "1-3"
    )"

    BASIC_OFFICIAL_ID="$(
        submit_job \
          basic_official \
          "$BASIC_ID" \
          "00:30:00" \
          "3G" \
          "1"
    )"

    RMSF_ID="$(
        submit_job \
          rmsf \
          "$PREFLIGHT_ID" \
          "00:45:00" \
          "4G" \
          "1"
    )"

    RMSF_OFFICIAL_ID="$(
        submit_job \
          rmsf_official \
          "$RMSF_ID" \
          "00:20:00" \
          "2G" \
          "1"
    )"

    CORE_ID="$(
        submit_job \
          core \
          "$RMSF_ID" \
          "00:20:00" \
          "3G" \
          "1"
    )"

    SASA_ID="$(
        submit_job \
          sasa_dssp \
          "$PREFLIGHT_ID" \
          "04:00:00" \
          "8G" \
          "4"
    )"

    SENSITIVITY_ID="$(
        submit_job \
          sensitivity \
          "$BASIC_ID" \
          "00:30:00" \
          "2G" \
          "1"
    )"

    BASIC_TARGETS_ID="$(
        submit_job \
          basic_targets \
          "$SENSITIVITY_ID" \
          "00:20:00" \
          "2G" \
          "1"
    )"

    STRUCTURAL_TARGETS_ID="$(
        submit_job \
          structural_targets \
          "$SASA_ID" \
          "00:20:00" \
          "2G" \
          "1"
    )"

    MASTER_DEP="${BASIC_TARGETS_ID}:${STRUCTURAL_TARGETS_ID}"

    MASTER_ID="$(
        submit_job \
          master_targets \
          "$MASTER_DEP" \
          "00:10:00" \
          "2G" \
          "1"
    )"

    CORE_RMSD_DEP="${CORE_ID}:${BASIC_ID}"

    CORE_RMSD_ID="$(
        submit_job \
          core_rmsd \
          "$CORE_RMSD_DEP" \
          "01:00:00" \
          "3G" \
          "1"
    )"

    PREFIX_DEP="${MASTER_ID}:${CORE_RMSD_ID}:${SASA_ID}"

    PREFIX_ID="$(
        submit_job \
          prefix \
          "$PREFIX_DEP" \
          "00:15:00" \
          "2G" \
          "1"
    )"

    AUDIT_ID="$(
        submit_job \
          audit \
          "$PREFIX_ID" \
          "00:10:00" \
          "2G" \
          "1"
    )"

    FINAL_DEP="${AUDIT_ID}:${BASIC_OFFICIAL_ID}:${RMSF_OFFICIAL_ID}"

    FINAL_ID="$(
        submit_job \
          finalize \
          "$FINAL_DEP" \
          "00:10:00" \
          "2G" \
          "1"
    )"

    cat > "$RUN_ROOT/job_ids.txt" <<EOF
system=$SYSTEM
dataset_group=$DATASET_GROUP
run_id=$RUN_ID
run_root=$RUN_ROOT
preflight=$PREFLIGHT_ID
basic=$BASIC_ID
basic_official=$BASIC_OFFICIAL_ID
rmsf=$RMSF_ID
rmsf_official=$RMSF_OFFICIAL_ID
core=$CORE_ID
sasa_dssp=$SASA_ID
sensitivity=$SENSITIVITY_ID
basic_targets=$BASIC_TARGETS_ID
structural_targets=$STRUCTURAL_TARGETS_ID
master_targets=$MASTER_ID
core_rmsd=$CORE_RMSD_ID
prefix=$PREFIX_ID
audit=$AUDIT_ID
finalize=$FINAL_ID
EOF

    echo "================================================================================"
    echo "PIPELINE SUBMITTED"
    echo "================================================================================"
    printf '%-22s %s\n' "preflight" "$PREFLIGHT_ID"
    printf '%-22s %s\n' "basic array" "$BASIC_ID"
    printf '%-22s %s\n' "basic official" "$BASIC_OFFICIAL_ID"
    printf '%-22s %s\n' "rmsf" "$RMSF_ID"
    printf '%-22s %s\n' "rmsf official" "$RMSF_OFFICIAL_ID"
    printf '%-22s %s\n' "core" "$CORE_ID"
    printf '%-22s %s\n' "sasa/dssp" "$SASA_ID"
    printf '%-22s %s\n' "sensitivity" "$SENSITIVITY_ID"
    printf '%-22s %s\n' "basic targets" "$BASIC_TARGETS_ID"
    printf '%-22s %s\n' "structural targets" "$STRUCTURAL_TARGETS_ID"
    printf '%-22s %s\n' "master targets" "$MASTER_ID"
    printf '%-22s %s\n' "core rmsd" "$CORE_RMSD_ID"
    printf '%-22s %s\n' "prefix" "$PREFIX_ID"
    printf '%-22s %s\n' "audit" "$AUDIT_ID"
    printf '%-22s %s\n' "finalize" "$FINAL_ID"

    echo
    echo "Run root:"
    echo "$RUN_ROOT"
    echo
    echo "Jobs:"
    echo "$RUN_ROOT/job_ids.txt"
fi
