#!/bin/bash
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

: "${STAGE:?Falta STAGE}"
: "${SYSTEM:?Falta SYSTEM}"
: "${DATASET_GROUP:?Falta DATASET_GROUP}"
: "${RUN_ROOT:?Falta RUN_ROOT}"

PROJECT="${PROJECT:-$HOME/md-hpc-benchmarks}"
WORK="${WORK:-$SCRATCH/md-hpc-benchmarks}"
VENV="$HOME/venvs/md-poc-py311"

RAW_ROOT="$WORK/raw/atlas"
UNPACKED_DIR="$RAW_ROOT/$DATASET_GROUP/$SYSTEM/unpacked"
ANALYSIS_DIR="$UNPACKED_DIR/${SYSTEM}_analysis"

PROCESSED_ROOT="$RUN_ROOT/processed"
SYSTEM_ROOT="$PROCESSED_ROOT/$DATASET_GROUP/$SYSTEM"

VALIDATION_ROOT="$RUN_ROOT/validation/$SYSTEM"
SENSITIVITY_ROOT="$RUN_ROOT/equilibration_sensitivity/$SYSTEM"
AUDIT_ROOT="$RUN_ROOT/audit/$SYSTEM"

BASIC_DIR="$SYSTEM_ROOT/basic"
RMSF_DIR="$SYSTEM_ROOT/rmsf"
CORE_DIR="$SYSTEM_ROOT/core_overlap"
CORE_RMSD_DIR="$SYSTEM_ROOT/core_rmsd"
SASA_DSSP_DIR="$SYSTEM_ROOT/sasa_dssp"

BASIC_TARGETS_DIR="$SYSTEM_ROOT/targets"
STRUCTURAL_TARGETS_DIR="$SYSTEM_ROOT/structural_targets_compositional"
MASTER_TARGETS_DIR="$SYSTEM_ROOT/master_targets"
PREFIX_DIR="$SYSTEM_ROOT/prefix_sequences"

mkdir -p \
  "$PROCESSED_ROOT" \
  "$VALIDATION_ROOT" \
  "$SENSITIVITY_ROOT" \
  "$AUDIT_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1

# SASA/Shrake-Rupley escala con OpenMP. El stage sasa_dssp
# utiliza los CPUs reservados por Slurm; el resto del pipeline
# permanece limitado a un único hilo OpenMP.
if [[ "$STAGE" == "sasa_dssp" || "$STAGE" == "sasa_replica" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
else
    export OMP_NUM_THREADS=1
fi

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "================================================================================"
echo "ATLAS PIPELINE STAGE"
echo "================================================================================"
echo "Fecha:          $(date)"
echo "Nodo:           $(hostname)"
echo "Job ID:         ${SLURM_JOB_ID:-NA}"
echo "Stage:          $STAGE"
echo "System:         $SYSTEM"
echo "Dataset group:  $DATASET_GROUP"
echo "Run root:       $RUN_ROOT"
echo

case "$STAGE" in

    preflight)
        python \
          "$PROJECT/src/pipeline/preflight_atlas_system.py" \
          --system "$SYSTEM" \
          --replicas "1,2,3" \
          --unpacked-dir "$UNPACKED_DIR" \
          --output-dir "$RUN_ROOT/preflight/$SYSTEM" \
          --analysis-step-ps 100 \
          --protein-step-ps 10 \
          --expected-end-ns 100 \
          --time-tolerance-ps 1e-3
        ;;

    basic)
        : "${SLURM_ARRAY_TASK_ID:?basic requiere array 1-3}"

        REPLICA="$SLURM_ARRAY_TASK_ID"

        python \
          "$PROJECT/src/pipeline/extract_atlas_basic.py" \
          --system "$SYSTEM" \
          --dataset-group "$DATASET_GROUP" \
          --replica "$REPLICA" \
          --raw-root "$RAW_ROOT" \
          --output-root "$PROCESSED_ROOT" \
          --prefix-end-ns 20 \
          --coarse-step-ps 100 \
          --expected-native-step-ps 10 \
          --expected-duration-ns 100
        ;;

    basic_official)
        python \
          "$PROJECT/src/pipeline/compare_atlas_official_generic.py" \
          --system "$SYSTEM" \
          --official-dir "$ANALYSIS_DIR" \
          --calculated-dir "$BASIC_DIR" \
          --output-dir "$VALIDATION_ROOT/basic_official" \
          --legacy-script "$PROJECT/src/validation/compare_atlas_official.py"
        ;;

    rmsf)
        python \
          "$PROJECT/src/pipeline/extract_atlas_rmsf.py" \
          --system "$SYSTEM" \
          --dataset-group "$DATASET_GROUP" \
          --raw-root "$RAW_ROOT" \
          --output-root "$PROCESSED_ROOT" \
          --replicas "1,2,3" \
          --start-ns 0.1 \
          --prefix-end-ns 20 \
          --expected-native-step-ps 10 \
          --expected-duration-ns 100
        ;;

    rmsf_official)
        OFFICIAL_RMSF="$ANALYSIS_DIR/${SYSTEM}_RMSF.tsv"
        CALCULATED_RMSF="$RMSF_DIR/rmsf_prefix_full_long.csv"

        if [[ ! -f "$OFFICIAL_RMSF" ]]; then
            echo "ERROR: falta RMSF oficial:"
            echo "$OFFICIAL_RMSF"
            false
        fi

        if [[ ! -f "$CALCULATED_RMSF" ]]; then
            echo "ERROR: falta RMSF calculado:"
            echo "$CALCULATED_RMSF"
            false
        fi

        python \
          "$PROJECT/src/validation/compare_atlas_rmsf_official.py" \
          --system "$SYSTEM" \
          --official-rmsf "$OFFICIAL_RMSF" \
          --calculated-long "$CALCULATED_RMSF" \
          --output-dir "$VALIDATION_ROOT/rmsf_official" \
          --official-scale-to-A 1.0
        ;;

    core)
        python \
          "$PROJECT/src/pipeline/evaluate_atlas_core.py" \
          --system "$SYSTEM" \
          --dataset-group "$DATASET_GROUP" \
          --rmsf-long "$RMSF_DIR/rmsf_prefix_full_long.csv" \
          --output-root "$PROCESSED_ROOT" \
          --quantiles "0.3,0.4,0.5" \
          --min-votes 2 \
          --min-segment-length 3
        ;;

    core_rmsd)
        python \
          "$PROJECT/src/pipeline/extract_atlas_core_rmsd.py" \
          --system "$SYSTEM" \
          --dataset-group "$DATASET_GROUP" \
          --raw-root "$RAW_ROOT" \
          --processed-root "$PROCESSED_ROOT" \
          --replicas "1,2,3" \
          --quantiles "0.3,0.4,0.5" \
          --prefix-end-ns 20 \
          --tail-start-ns 80 \
          --expected-native-step-ps 10 \
          --expected-duration-ns 100
        ;;

    sasa_dssp)
        python \
          "$PROJECT/src/pipeline/extract_atlas_sasa_dssp.py" \
          --system "$SYSTEM" \
          --dataset-group "$DATASET_GROUP" \
          --raw-root "$RAW_ROOT" \
          --processed-root "$PROCESSED_ROOT" \
          --replicas "1,2,3" \
          --stride 10 \
          --prefix-end-ns 20 \
          --expected-native-step-ps 10 \
          --expected-duration-ns 100
        ;;

    sasa_replica)
        : "${SLURM_ARRAY_TASK_ID:?sasa_replica requiere array 1-3}"

        REPLICA="$SLURM_ARRAY_TASK_ID"
        REPLICA_ROOT="$RUN_ROOT/sasa_replica_outputs/R${REPLICA}"

        echo "SASA replica:   R${REPLICA}"
        echo "OMP threads:    ${OMP_NUM_THREADS}"
        echo "Replica root:   ${REPLICA_ROOT}"

        python \
          "$PROJECT/src/pipeline/extract_atlas_sasa_dssp.py" \
          --system "$SYSTEM" \
          --dataset-group "$DATASET_GROUP" \
          --raw-root "$RAW_ROOT" \
          --processed-root "$REPLICA_ROOT" \
          --replicas "$REPLICA" \
          --stride 10 \
          --chunk-size 25 \
          --sphere-points 960 \
          --probe-radius-nm 0.14 \
          --prefix-end-ns 20 \
          --expected-native-step-ps 10 \
          --expected-duration-ns 100
        ;;

    sasa_merge)
        mkdir -p "$SASA_DSSP_DIR"

        python \
          "$PROJECT/src/pipeline/merge_atlas_sasa_dssp.py" \
          --system "$SYSTEM" \
          --dataset-group "$DATASET_GROUP" \
          --replicas "1,2,3" \
          --input-parent "$RUN_ROOT/sasa_replica_outputs" \
          --output-dir "$SASA_DSSP_DIR"
        ;;

    sensitivity)
        python \
          "$PROJECT/src/diagnostics/equilibration_sensitivity.py" \
          --system "$SYSTEM" \
          --replicas "1,2,3" \
          --input-dir "$BASIC_DIR" \
          --output-dir "$SENSITIVITY_ROOT"
        ;;

    sensitivity_replica)
        : "${SLURM_ARRAY_TASK_ID:?sensitivity_replica requiere array 1-3}"

        REPLICA="$SLURM_ARRAY_TASK_ID"
        REPLICA_ROOT="$RUN_ROOT/sensitivity_replica_outputs/R${REPLICA}"

        echo "Sensitivity replica: R${REPLICA}"
        echo "Replica root:        ${REPLICA_ROOT}"

        python \
          "$PROJECT/src/diagnostics/equilibration_sensitivity.py" \
          --system "$SYSTEM" \
          --replicas "$REPLICA" \
          --input-dir "$BASIC_DIR" \
          --output-dir "$REPLICA_ROOT"
        ;;

    sensitivity_merge)
        mkdir -p "$SENSITIVITY_ROOT"

        python \
          "$PROJECT/src/pipeline/merge_atlas_equilibration_sensitivity.py" \
          --system "$SYSTEM" \
          --replicas "1,2,3" \
          --input-parent "$RUN_ROOT/sensitivity_replica_outputs" \
          --output-dir "$SENSITIVITY_ROOT"
        ;;

    basic_targets)
        python \
          "$PROJECT/src/targets/build_validated_targets.py" \
          --system "$SYSTEM" \
          --replicas "1,2,3" \
          --input-dir "$BASIC_DIR" \
          --sensitivity-csv "$SENSITIVITY_ROOT/equilibration_sensitivity.csv" \
          --output-dir "$BASIC_TARGETS_DIR" \
          --nskip 10 \
          --min-duration-ns 20 \
          --min-neff 10 \
          --tail-start-ns 80
        ;;

    structural_targets)
        python \
          "$PROJECT/src/targets/build_compositional_structural_targets.py" \
          --system "$SYSTEM" \
          --replicas "1,2,3" \
          --input-dir "$SASA_DSSP_DIR" \
          --output-dir "$STRUCTURAL_TARGETS_DIR" \
          --nskip 1 \
          --min-duration-ns 20 \
          --min-neff 10 \
          --tail-start-ns 80
        ;;

    master_targets)
        python \
          "$PROJECT/src/dataset/assemble_master_targets.py" \
          --basic-long "$BASIC_TARGETS_DIR/validated_targets_long.csv" \
          --structural-long "$STRUCTURAL_TARGETS_DIR/structural_targets_long.csv" \
          --composition-csv "$STRUCTURAL_TARGETS_DIR/secondary_structure_composition.csv" \
          --output-dir "$MASTER_TARGETS_DIR"
        ;;

    prefix)
        python \
          "$PROJECT/src/dataset/assemble_prefix_sequences.py" \
          --system "$SYSTEM" \
          --replicas "1,2,3" \
          --basic-dir "$BASIC_DIR" \
          --core-rmsd-dir "$CORE_RMSD_DIR" \
          --sasa-dssp-dir "$SASA_DSSP_DIR" \
          --master-targets "$MASTER_TARGETS_DIR/master_targets_wide.csv" \
          --output-dir "$PREFIX_DIR" \
          --prefix-end-ns 20 \
          --expected-step-ps 100
        ;;

    audit)
        PACKAGE_AUDIT="$AUDIT_ROOT/package"
        mkdir -p "$PACKAGE_AUDIT"

        python \
          "$PROJECT/src/dataset/audit_prefix_package.py" \
          --npz "$PREFIX_DIR/prefix_sequences_raw.npz" \
          --manifest "$PREFIX_DIR/prefix_sequence_manifest.csv" \
          --metadata "$PREFIX_DIR/prefix_sequences_metadata.json" \
          --output-dir "$PACKAGE_AUDIT" \
          --expected-steps 201 \
          --expected-step-ps 100 \
          --maximum-time-ns 20

        python \
          "$PROJECT/src/pipeline/audit_prefix_package.py" \
          --system "$SYSTEM" \
          --replicas "1,2,3" \
          --package-dir "$PREFIX_DIR" \
          --master-targets "$MASTER_TARGETS_DIR/master_targets_wide.csv" \
          --prefix-end-ns 20 \
          --step-ps 100 \
          --numeric-tolerance 1e-10 \
          | tee "$AUDIT_ROOT/pipeline_audit.txt"
        ;;

    finalize)
        REQUIRED_FILES=(
          "$PREFIX_DIR/prefix_sequences_raw.npz"
          "$PREFIX_DIR/prefix_sequence_manifest.csv"
          "$PREFIX_DIR/prefix_sequences_metadata.json"
          "$MASTER_TARGETS_DIR/master_targets_wide.csv"
          "$VALIDATION_ROOT/basic_official/comparison_summary.csv"
          "$AUDIT_ROOT/pipeline_audit.txt"
        )

        for FILE in "${REQUIRED_FILES[@]}"; do
            if [[ ! -f "$FILE" ]]; then
                echo "ERROR: falta archivo final:"
                echo "$FILE"
                false
            fi
        done

        if ! find "$VALIDATION_ROOT/rmsf_official" \
          -type f \
          -print -quit \
          | grep -q .; then
            echo "ERROR: no hay resultados de comparación RMSF oficial."
            false
        fi

        cat > "$RUN_ROOT/PIPELINE_STATUS.txt" <<EOF
pipeline_status=PASS
system=$SYSTEM
dataset_group=$DATASET_GROUP
n_replicas=3
prefix_end_ns=20
expected_steps=201
expected_step_ps=100
normalized=false
imputed=false
official_basic_comparison=true
official_rmsf_comparison=true
package_audit=true
cross_file_audit=true
completed_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

        (
            set -euo pipefail

            cd "$RUN_ROOT"

            {
                find \
                  preflight \
                  processed \
                  validation \
                  equilibration_sensitivity \
                  audit \
                  -type f \
                  -print0

                printf '%s\0' \
                  "PIPELINE_STATUS.txt"
            } \
            | sort -z \
            | xargs -0 sha256sum \
            > checksums.sha256

            sha256sum \
              --check \
              checksums.sha256
        )

        echo
        echo "================================================================================"
echo "ATLAS SYSTEM PIPELINE: PASS"
echo "================================================================================"
echo "System:   $SYSTEM"
echo "Group:    $DATASET_GROUP"
echo "Run root: $RUN_ROOT"
echo
        ;;

    *)
        echo "ERROR: STAGE desconocido: $STAGE"
        false
        ;;
esac

deactivate
