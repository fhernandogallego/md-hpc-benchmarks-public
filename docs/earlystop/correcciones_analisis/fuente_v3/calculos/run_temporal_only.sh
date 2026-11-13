#!/usr/bin/env bash
# Add the v3 controls after a previous target reconstruction, without repeating it.
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Uso: bash run_temporal_only.sh RUN TARGETS_INCLUDE OUTPUT_NUEVO [TARGETS_EXCLUDE]" >&2
  exit 2
fi
RUN_PATH=$1
TARGETS_PATH=$2
OUTPUT_PATH=$3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ -e "$OUTPUT_PATH" ]; then
  echo "La salida ya existe: $OUTPUT_PATH" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUTPUT_PATH"
python3 "$SCRIPT_DIR/temporal_controls.py" --run "$RUN_PATH" --targets "$TARGETS_PATH" \
  --only rmsd --prefix-ns 5 10 20 --out "$OUTPUT_PATH/temporal_controls_rmsd"
for TARGET_NAME in core_rmsd_q30 rg sasa helix strand; do
  python3 "$SCRIPT_DIR/temporal_controls.py" --run "$RUN_PATH" --targets "$TARGETS_PATH" \
    --only "$TARGET_NAME" --prefix-ns 20 --out "$OUTPUT_PATH/temporal_controls_$TARGET_NAME"
done
if [ "$#" -eq 4 ]; then
  for TARGET_NAME in helix strand; do
    python3 "$SCRIPT_DIR/temporal_controls.py" --run "$RUN_PATH" --targets "$4" \
      --only "$TARGET_NAME" --prefix-ns 20 --out "$OUTPUT_PATH/temporal_controls_${TARGET_NAME}_exclude_constants"
  done
fi
python3 "$SCRIPT_DIR/summarize_temporal.py" --results "$OUTPUT_PATH" \
  --out "$OUTPUT_PATH/RESUMEN_BASELINES_ABLACION.txt"
