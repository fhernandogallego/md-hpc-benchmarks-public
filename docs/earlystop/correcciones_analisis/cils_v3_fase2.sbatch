#!/usr/bin/env bash
# CILS v3, fase 2: análisis sobre los objetivos consolidados del bloque A (757 sistemas, solo entrenamiento).
# Un trabajo = una etapa (variable STAGE). Primero se miden tiempo y memoria de pocas etapas y luego se lanza el resto.
# Uso:  sbatch --export=ALL,STAGE=intervals_rmsd cils_v3_fase2.sbatch
# Etapas: intervals_rmsd | intervals_rg | prefix_rmsd | temporal_rmsd | temporal_rg | temporal_sasa | temporal_helix |
#         temporal_strand | temporal_core_rmsd_q30 | intervals_core_rmsd_q30 | temporal_helix_exclude | temporal_strand_exclude
# La salida va a una carpeta NUEVA por etapa (el trabajo se niega a sobrescribir). No se abre validation ni test.
#SBATCH --job-name=cils_v3_fase2
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/fase2_%x_%j.out
#SBATCH --error=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/fase2_%x_%j.err
set -euo pipefail
: "${STAGE:?Falta STAGE (ver la lista de etapas en la cabecera)}"
module purge
module load Python/3.11.5-GCCcore-13.2.0
source /home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

W=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks
S=$W/maintenance/paper2_cils_v3_relaunch_v1
CODE=$W/maintenance/paper2_cils_v3_20261112_101652/package/calculos
TRAIN=$W/results/paper2_automation/auto_20261030_091302/training
INC=$S/merged/targets_include_constants.csv
EXC=$S/merged/targets_exclude_constants.csv
RES=$W/results/paper2_cils_v3_fase2
INC_SHA=b45242bee882b33583101b3b49c1487514c0386f3da91ee7639cce9995520294
EXC_SHA=880ab7d39b164140388468e30042da93eb6d1d819b580787b98a8a197cac7942

# Entradas congeladas: si cambian, no se ejecuta nada.
[ "$(sha256sum "$INC" | cut -d' ' -f1)" = "$INC_SHA" ] || { echo "ENTRADA_ALTERADA: $INC"; exit 3; }
[ "$(sha256sum "$EXC" | cut -d' ' -f1)" = "$EXC_SHA" ] || { echo "ENTRADA_ALTERADA: $EXC"; exit 3; }
OUT=$RES/$STAGE
[ ! -e "$OUT" ] || { echo "LA_SALIDA_YA_EXISTE: $OUT"; exit 2; }
mkdir -p "$RES"

case "$STAGE" in
  intervals_rmsd|intervals_rg|intervals_core_rmsd_q30)
    ONLY=${STAGE#intervals_}; CMD=(python3 "$CODE/reanalyse.py" --run "$TRAIN" --targets "$INC" --only "$ONLY" --mode intervals --out "$OUT") ;;
  prefix_rmsd)
    CMD=(python3 "$CODE/reanalyse.py" --run "$TRAIN" --targets "$INC" --only rmsd --mode prefix --out "$OUT") ;;
  temporal_rmsd)
    CMD=(python3 "$CODE/temporal_controls.py" --run "$TRAIN" --targets "$INC" --only rmsd --prefix-ns 5 10 20 --out "$OUT") ;;
  temporal_helix_exclude|temporal_strand_exclude)
    ONLY=${STAGE#temporal_}; ONLY=${ONLY%_exclude}
    CMD=(python3 "$CODE/temporal_controls.py" --run "$TRAIN" --targets "$EXC" --only "$ONLY" --prefix-ns 20 --out "$OUT") ;;
  temporal_rg|temporal_sasa|temporal_helix|temporal_strand|temporal_core_rmsd_q30)
    ONLY=${STAGE#temporal_}; CMD=(python3 "$CODE/temporal_controls.py" --run "$TRAIN" --targets "$INC" --only "$ONLY" --prefix-ns 20 --out "$OUT") ;;
  *) echo "STAGE_DESCONOCIDA: $STAGE"; exit 2 ;;
esac

printf 'FASE2_START job=%s stage=%s time=%s\n' "$SLURM_JOB_ID" "$STAGE" "$(date -Is)"
printf 'FASE2_CMD %s\n' "${CMD[*]}"
"${CMD[@]}"
printf 'FASE2_PASS job=%s stage=%s time=%s\n' "$SLURM_JOB_ID" "$STAGE" "$(date -Is)"
ls -l "$OUT"
( cd "$OUT" && sha256sum * 2>/dev/null )
