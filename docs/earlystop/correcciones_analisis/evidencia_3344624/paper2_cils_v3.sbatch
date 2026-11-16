#!/usr/bin/env bash
#SBATCH --job-name=paper2_cils_v3
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/activate

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUNLOG="$REV_LOGDIR/ejecucion_v3.log"

echo "JOB_ID=$SLURM_JOB_ID"
echo "MODO=$REV_MODE"
date -Is

python3 - <<'PY'
import sys
import numpy
import pandas
import pymbar
print("Python:", sys.executable, sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("PyMBAR:", pymbar.__version__)
PY

if [ "$REV_MODE" = temporal ]; then
  if ! bash "$REV_CODE/calculos/run_temporal_only.sh" \
    "$REV_RUN" \
    "$REV_TARGETS_INC" \
    "$REV_OUTPUT" \
    "$REV_TARGETS_EXC" > "$RUNLOG" 2>&1; then
    tail -n 100 "$RUNLOG" >&2
    exit 1
  fi
else
  if ! bash "$REV_CODE/calculos/run_revision.sh" \
    "$REV_RUN" \
    "$REV_INV" \
    "$REV_OUTPUT" > "$RUNLOG" 2>&1; then
    tail -n 100 "$RUNLOG" >&2
    exit 1
  fi
fi

# Validaciones finales obligatorias.
[ -s "$REV_OUTPUT/RESUMEN_BASELINES_ABLACION.txt" ]

for target in rmsd core_rmsd_q30 rg sasa helix strand; do
  [ -s "$REV_OUTPUT/temporal_controls_$target/provenance.json" ]
done

if [ "$REV_MODE" = full ]; then
  [ -s "$REV_OUTPUT/rmsd_reference_check.json" ]
  [ -s "$REV_OUTPUT/prefix_without_core/provenance.json" ]
else
  cp -a "$REV_V2DIR" "$REV_OUTPUT/v2_reutilizada"
fi

# Guardar resultados y trazabilidad.
cp "$RUNLOG" "$REV_OUTPUT/ejecucion_v3.log"
mkdir -p "$REV_OUTPUT/slurm_logs"

for channel in out err; do
  file="$REV_LOGDIR/slurm_${SLURM_JOB_ID}.$channel"
  if [ -f "$file" ]; then
    cp "$file" "$REV_OUTPUT/slurm_logs/"
  fi
done

cp "$REV_SETUP/preflight.log" "$REV_OUTPUT/preflight.log"
cp "$REV_SETUP/inventory_audit.json" \
  "$REV_OUTPUT/inventory_audit.json"
cp "$REV_SETUP/training_not_inventory.csv" \
  "$REV_OUTPUT/training_not_inventory.csv"
cp "$REV_SETUP/paquete_revision_scayle_v3.manifest.json" \
  "$REV_OUTPUT/paquete_revision_scayle_v3.manifest.json"

{
  echo "commit=$REV_COMMIT"
  echo "mode=$REV_MODE"
  echo "run=$REV_RUN"
  echo "inventory=$REV_INV"
  echo "job_id=$SLURM_JOB_ID"
  echo "finished=$(date -Is)"
  python3 -m pip freeze |
    grep -Ei '^(numpy|pandas|pymbar|scipy)==' || true
} > "$REV_OUTPUT/PROVENANCE_RUN.txt"

# Comprimir solo resultados, sin trayectorias originales.
ARCHIVE="${REV_OUTPUT}.tar.gz"

tar -C "$(dirname "$REV_OUTPUT")" \
  -czf "$ARCHIVE" "$(basename "$REV_OUTPUT")"

sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

echo "V3_RESULTS=COMPLETE"
echo "RESULT_DIR=$REV_OUTPUT"
echo "PACKAGE=$ARCHIVE"
echo "EXECUTION_LOG=$RUNLOG"
