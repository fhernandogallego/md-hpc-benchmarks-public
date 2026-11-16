#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Uso: bash run_revision.sh RUN_ATLAS942 INVENTARIO OUTPUT_NUEVO" >&2
  exit 2
fi
RUN_PATH=$1
INVENTORY_PATH=$2
OUTPUT_PATH=$3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ -e "$OUTPUT_PATH" ]; then
  echo "La salida ya existe: $OUTPUT_PATH" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUTPUT_PATH"
python3 "$SCRIPT_DIR/build_targets.py" --inventory "$INVENTORY_PATH" \
  --config "$SCRIPT_DIR/targets_config.json" --constant-policy include \
  --out "$OUTPUT_PATH/targets_include_constants.csv"
python3 "$SCRIPT_DIR/verify_rebuilt_rmsd.py" --run "$RUN_PATH" \
  --rebuilt-targets "$OUTPUT_PATH/targets_include_constants.csv" \
  --out "$OUTPUT_PATH/rmsd_reference_check.json"
python3 "$SCRIPT_DIR/build_targets.py" --inventory "$INVENTORY_PATH" \
  --config "$SCRIPT_DIR/targets_config.json" --constant-policy exclude \
  --out "$OUTPUT_PATH/targets_exclude_constants.csv"
python3 "$SCRIPT_DIR/reanalyse.py" --run "$RUN_PATH" \
  --targets "$OUTPUT_PATH/targets_include_constants.csv" --only rmsd \
  --mode prefix --out "$OUTPUT_PATH/prefix_without_core"
python3 "$SCRIPT_DIR/temporal_controls.py" --run "$RUN_PATH" \
  --targets "$OUTPUT_PATH/targets_include_constants.csv" --only rmsd \
  --prefix-ns 5 10 20 --out "$OUTPUT_PATH/temporal_controls_rmsd"
for TARGET_NAME in core_rmsd_q30 rg sasa helix strand; do
  python3 "$SCRIPT_DIR/temporal_controls.py" --run "$RUN_PATH" \
    --targets "$OUTPUT_PATH/targets_include_constants.csv" --only "$TARGET_NAME" \
    --prefix-ns 20 --out "$OUTPUT_PATH/temporal_controls_$TARGET_NAME"
done
for TARGET_NAME in rmsd core_rmsd_q30 rg sasa helix strand; do
  python3 "$SCRIPT_DIR/reanalyse.py" --run "$RUN_PATH" \
    --targets "$OUTPUT_PATH/targets_include_constants.csv" --only "$TARGET_NAME" \
    --mode intervals --out "$OUTPUT_PATH/intervals_$TARGET_NAME"
done
for TARGET_NAME in helix strand; do
  python3 "$SCRIPT_DIR/temporal_controls.py" --run "$RUN_PATH" \
    --targets "$OUTPUT_PATH/targets_exclude_constants.csv" --only "$TARGET_NAME" \
    --prefix-ns 20 --out "$OUTPUT_PATH/temporal_controls_${TARGET_NAME}_exclude_constants"
  python3 "$SCRIPT_DIR/reanalyse.py" --run "$RUN_PATH" \
    --targets "$OUTPUT_PATH/targets_exclude_constants.csv" --only "$TARGET_NAME" \
    --mode intervals --out "$OUTPUT_PATH/intervals_${TARGET_NAME}_exclude_constants"
done
python3 "$SCRIPT_DIR/summarize_temporal.py" --results "$OUTPUT_PATH" \
  --out "$OUTPUT_PATH/RESUMEN_BASELINES_ABLACION.txt"
