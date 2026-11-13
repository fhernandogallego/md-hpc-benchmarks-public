# EARLYSTOP / CILS v3 — plan de reejecución recuperable (13/11/2026)

**Estado: propuesta técnica documentada, NO autorizada ni enviada.** El `.sbatch`, los logs completos, el manifiesto y el diagnóstico real de recursos están publicados en [`evidencia_3344624/`](evidencia_3344624/). Se ha verificado el tiempo solicitado de 12 horas y el límite publicado de `sapphire`/QoS `normal`. Falta medir un fragmento antes de ratificar estimaciones y realizar una comprobación previa de los ejecutores de fragmentación. Este plan **no es evidencia de resultados corregidos**; las tres afirmaciones científicas siguen retiradas de la memoria.

## 1. Incidente y gasto computacional

- Job `3344624`, `paper2_cils_v3`, `uva_dma_2` / `sapphire`: `TIMEOUT`, `12:00:28`; paso `batch=CANCELLED`, sin artefactos finales en `WORK/results/paper2_cils_v3_20261112_101652/`. Logs archivados en `WORK/results/paper2_cils_v3_20261112_101652_logs/`; `WORK=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks`.
- `ejecucion_v3.log` registra `[100/757]` como último hito de progreso. El script imprime `[1]`, `[2]`, `[3]` y luego cada 100; **100 no es una estimación del total que llegó a completar**.
- Causa de pérdida de trabajo: [`build_targets.py`](fuente_v3/calculos/build_targets.py) acumula `records` durante todo el inventario y **solo al final** escribe `targets_*.csv`. [`run_revision.sh`](fuente_v3/calculos/run_revision.sh) ejecuta **primero include y después exclude**, antes de cualquier sensibilidad/modelado. Ninguna etapa termina y queda marcada si el primer paso agota el walltime.
- Perfil **deducido del código, no medido con profiler**: 757 sistemas × 3 réplicas × 6 observables = **13.626 valoraciones de objetivos por política**, repetidas para `include/exclude` (27.252). Para core, `detect_equilibration` se llama 2.271 veces por política, además de `statistical_inefficiency`; `pandas.read_csv` carga series por sistema y política. Estas operaciones PyMBAR/IO son los principales candidatos a cuello de botella; confirmarlo con tiempos por fragmento y métricas `sacct`.
- **Estimación provisional, no perfilada:** el último hito escrito fue `[100/757]` tras 12 horas. Dado que el programa solo imprime cada 100 sistemas, esto demuestra un avance mínimo de 100, pero no cuántos sistemas posteriores terminó; la aproximación **46–91 h para 757 sistemas y una política** (92–182 h para dos ejecuciones) supone rendimiento lineal y progresión entre 100 y 199, por lo que es **meramente orientativa**. Un fragmento de 10 sistemas podría requerir aproximadamente 0,6–1,2 h por política bajo esa suposición; el tiempo real puede desviarse por sistema y lectura Lustre. Se propone **6 h por fragmento de 10 sistemas, con comprobación piloto**, y subdividir los fragmentos lentos sin repetir los ya completos.
- **Límites comprobados:** `scontrol show partition sapphire -o` declara `MaxTime=UNLIMITED`; `sacctmgr -nP show qos format=Name,MaxWall,MaxTRES` declara para QoS `normal` `MaxWall=5-00:00:00` (5 días). El **límite real del trabajo `3344624` fue el `#SBATCH --time=12:00:00` solicitado** (confirmado por `sacct Timelimit=12:00:00`); también tenía `#SBATCH --qos=normal`. Pueden existir límites adicionales por asociación o prioridad, no investigados. `sinfo` devolvió `Access/permission denied`, pero las dos consultas anteriores sí funcionaron. **Una solicitud de 6 h cabe dentro de los límites publicados**, pendiente de la aprobación normal del planificador.
- **Recursos comprobados en el `.sbatch` original:** `--nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=12:00:00 --account=uva_dma_2 --partition=sapphire --qos=normal`. El script fuerza `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, usa un único proceso Python y no activa paralelismo por sistemas. `sacct` registra `MaxRSS=115987K` y `MaxVMSize=2811M` del paso **batch**: son métricas de dicho paso, **no garantía del pico de todas las lecturas/procesos**. Entorno verificado: `/home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/python3`, módulo `Python/3.11.5-GCCcore-13.2.0`, NumPy `1.26.4`, Pandas `2.2.3`, PyMBAR `4.0.3`. **Propuesta piloto, aún no validada:** 1 CPU / 8 GiB / 6 h / como máximo **2 fragmentos concurrentes**, con QoS `normal`; medir RSS y duración antes de subir concurrencia.

### Comando científico realmente utilizado en el trabajo anterior

El `.sbatch` verificado selecciona `REV_MODE=full`, activa el entorno indicado y ejecuta exactamente:

```bash
bash "$REV_CODE/calculos/run_revision.sh" \
  "$REV_RUN" "$REV_INV" "$REV_OUTPUT" > "$RUNLOG" 2>&1
```

Variables cuyas rutas están respaldadas por el preflight, por `package_root.txt` y por la auditoría:

```bash
WORK=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks
REV_CODE="$WORK/maintenance/paper2_cils_v3_20261112_101652/package"
REV_RUN="$WORK/results/paper2_automation/auto_20261030_091302/training"
REV_INV="$WORK/results/paper2_miti_20261109/inputs/inventory_miti.csv"
REV_OUTPUT="$WORK/results/paper2_cils_v3_20261112_101652"
REV_LOGDIR="$WORK/results/paper2_cils_v3_20261112_101652_logs"
RUNLOG="$REV_LOGDIR/ejecucion_v3.log"
```

El contenido del script y su comando **sí** están verificados; la invocación externa original de `sbatch` y su forma concreta de exportar las variables `REV_*` **no aparece** en el `.sbatch`, por lo que no se inventa. En el preflight se verificaron los 22 archivos del paquete y SHA-256 del tar `f05bc1d96b7233ec3243a267de9b1f23c1c683f715e56f48321433095afdb17f`. Los hashes de `calculos/build_targets.py`, `reanalyse.py`, `temporal_controls.py`, `targets_config.json` y `run_revision.sh` del snapshot coinciden con los del manifiesto. Las cinco parejas de hashes se encuentran en [`evidencia_3344624/DIAGNOSTICO_RECURSOS.txt`](evidencia_3344624/DIAGNOSTICO_RECURSOS.txt).

## 2. Invariantes: nada de modificaciones científicas

- Inventario original de **757 sistemas train**, no selección por facilidad ni reducción de muestra, y `outer_folds.csv` sin cambios. Verificar que todos los IDs de los fragmentos pertenecen a los 757 sistemas de los folds originales, sin duplicados, y que la unión de fragmentos reproduce **byte a byte** el inventario original (excepto división física en archivos).
- Conservar código/configuración original v3, commit fuente [`2cc3d2b4701e62881b8ece6f9fc53ae1725a34dd`](https://github.com/fhernandogallego/md-hpc-benchmarks/commit/2cc3d2b4701e62881b8ece6f9fc53ae1725a34dd), paquete SHA declarado `f05bc1d96b7233ec3243a267de9b1f23c1c683f715e56f48321433095afdb17f`; comprobar igualdad con snapshot ejecutado en SCAYLE. `targets_config.json`: core usa `replica_prefix_rmsd_A` y detecta `t0` propio.
- No abrir ni calcular validation/test. No inventar nueva partición de calibración global: los roles `training/calibration/evaluation` del análisis conformal son subparticiones internas por grupos del entrenamiento existente.
- No reducir `bootstrap_draws=10000`, rejillas, longitudes mínimas, `Neff>=10`, semillas, protocolos, candidatos ni criterios de `constant_policy`. No combinar sensibilidades incompatibles (5/10 sin core versus 20 con core completo).
- El código del paquete v3 es el ejecutor científico. El envoltorio por fragmentos **solo** seleccionará filas del inventario (sin cambiar su orden), lanzará `build_targets.py` con los mismos argumentos y consolidará resultados **después de verificar completitud**.

## 3. Bloque A: reconstrucción de objetivos por sistemas, con checkpoint

**Diseño:** dividir el inventario original, en orden, en **76 archivos** de hasta 10 sistemas; preparar un array Slurm `--array=0-75%2` con recursos iniciales `--time=06:00:00 --cpus-per-task=1 --mem=8G --account=uva_dma_2 --partition=sapphire --qos=normal`. Sus 76 tareas se validan por separado; un TIMEOUT afecta a su fragmento, no a los demás. **Propuesta no enviada:** primero se debe verificar el generador, el consolidador y el consumo del primer fragmento. La concurrencia inicial 2 evita asumir que Lustre tolerará sin coste mayor paralelismo de lecturas.

Para cada fragmento `i`, ejecutar el **mismo script científico** en dos procesos separados:
```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/python3
# SHARD_FILE, CODE, CHUNK_OUT se obtienen del manifiesto congelado de 76 fragmentos.
"$PY" "$CODE/build_targets.py" --inventory "$SHARD_FILE" --config "$CODE/targets_config.json" --constant-policy include --out "$CHUNK_OUT/include.csv"
"$PY" "$CODE/build_targets.py" --inventory "$SHARD_FILE" --config "$CODE/targets_config.json" --constant-policy exclude --out "$CHUNK_OUT/exclude.csv"
```
**Esta es la orden interna del trabajador, no se ha ejecutado.** La implementación segura debe escribir en ubicaciones temporales únicas, verificar `*.summary.json` con `n_problems=0`, hash de cada archivo, 18 filas/ID (6 objetivos × 3 réplicas), ausencia de duplicados y luego promover **atómicamente** cada política y grabar un marcador `complete.json`. En la reanudación se permite saltar únicamente un fragmento/política cuyo `complete.json` y SHA de entrada/código/salida coincidan; nunca confiar en la mera existencia del CSV.

**Consolidación después de completar los 76 fragmentos × 2 políticas:** verificar 757 IDs, 13.626 filas por política, los seis targets por réplica, orden de filas idéntico al inventario original, ausencia de pérdidas/duplicados y diferencias `include` vs `exclude` **solo para ventanas constantes**. Guardar `targets_include_constants.csv`, `targets_exclude_constants.csv`, resumen, SHA y manifiesto `shards_manifest.csv`. Ejecutar `verify_rebuilt_rmsd.py --run .../training --rebuilt-targets .../targets_include_constants.csv --out .../rmsd_reference_check.json` y **detener** todo el flujo si no coincide la máscara y etiquetas RMSD histórico. Este paso usa solo training.

## 4. Bloque B: prefijos sin core (trabajo separado)

Se inicia **solo si A es PASS**. Mantener los 757 sistemas elegibles según etiquetas reconstruidas; no entrenar con otra población. Run original:
`TRAIN=$WORK/results/paper2_automation/auto_20261030_091302/training`.

```bash
"$PY" "$CODE/reanalyse.py" --run "$TRAIN" --targets "$MERGED/targets_include_constants.csv" --only rmsd --mode prefix --out "$RESULT/prefix_without_core"
"$PY" "$CODE/temporal_controls.py" --run "$TRAIN" --targets "$MERGED/targets_include_constants.csv" --only rmsd --prefix-ns 5 10 20 --out "$RESULT/temporal_controls_rmsd"
```
Ejecutar en **dos tareas Slurm independientes** para no volver a encadenar cálculos y permitir reutilizar resultados. Éxito del primer análisis: `prefix_without_core/{replica_predictions.csv,metrics.csv,selection.csv,provenance.json}` completos, predicciones OOF y `prefix_channels=[0,1,5,6,7]` en 5/10/20 ns. Éxito de temporal_controls: `feature_manifest.csv`, `outer_fold_manifest.csv`, `replica_predictions.csv`, `system_predictions.csv`, `metrics.csv`, `contrasts.csv`, `selection.csv`, `baseline_inner_scores.csv`, `provenance.json`; cotejo `FULL_RMSD_PREDICTION_CHECK=PASS` para el 20 ns **con core** y reporte separado de `ridge_without_core_full`. No afirmar comparación libre de fuga mezclando baselines de distinta dimensión.

## 5. Bloque C: calibración independiente (trabajos separados)

**Solo tras A PASS**, ejecutar los seis observables en tareas Slurm independientes, conservando los parámetros originales y los folds originales:
```bash
for TARGET in rmsd core_rmsd_q30 rg sasa helix strand; do
  "$PY" "$CODE/reanalyse.py" --run "$TRAIN" --targets "$MERGED/targets_include_constants.csv" --only "$TARGET" --mode intervals --out "$RESULT/intervals_$TARGET"
done
```
**El bucle anterior expresa el comando original por observable; para evitar otra ejecución monolítica, cada iteración debe enviarse como un job separado**, no ejecutar el bucle completo dentro de un único `.sbatch`. Para la sensibilidad `exclude`, repetir **solo helix y strand** desde el CSV `exclude` original, como hace `run_revision.sh`, también en tareas separadas.

Éxito por tarea: `intervals.csv` (true/predicted/lower/upper/covered/width), `partition_manifest.csv` (grupos en `training/calibration/evaluation` **sin intersección en cada outer fold**), `selection.csv`, `replica_predictions.csv`, `metrics.csv`, `coverage.csv`, `provenance.json`. En `coverage.csv` exigir `nominal={0.90,0.95}`, global `stratum=0` y estratos 1–4 con `n_systems`, `coverage`, `mean_width`, y `coverage_all_systems_in_group` donde aplica. No reinterpretar intervalos condicionados a predicciones OOF como garantía de cobertura por proteína.

## 6. Comprobaciones y seguimiento por fase

**Slurm:** cada tarea requiere `sacct -j <JOB> -X -P -o JobID,State,ExitCode,Elapsed,Timelimit,MaxRSS` con `COMPLETED`/salida `0:0`; guardar stdout/stderr con `JobID`, script/config/targets SHA y versiones. **No basta** un `PASS` de sintaxis ni el marcador de preflight. Si se interrumpe un shard, solo se repite ese shard/política y nunca se da por válido su producto parcial.

**Datos:** congelar hashes `dataset.npz`, `outer_folds.csv`, inventario original, paquete v3, código por fase y targets consolidados; demostrar 757 IDs y ausencia de IDs validation/test. Copiar manifiestos de roles y métricas a la carpeta de evidencia, sin reentrenar a partir de datos holdout. No modificar Figura 2 ni las afirmaciones científicas hasta revisión de resultados.

**Rutas propuestas:** código original `WORK/maintenance/paper2_cils_v3_20261112_101652/package`; nueva preparación `WORK/maintenance/paper2_cils_v3_relaunch_v1/`; nuevas salidas `WORK/results/paper2_cils_v3_relaunch_v1/`. **No sobrescribir** `paper2_cils_v3_20261112_101652` ni su carpeta de logs.

## 7. Bloqueos previos al envío: se investigan sin lanzar análisis

1. **HECHO:** `.sbatch` ejecutado, logs íntegros, manifiesto y preflight publicados en [`evidencia_3344624/`](evidencia_3344624/). Confirmados comando interno y rutas de fuentes, no la línea de envío externa `sbatch`.
2. **HECHO:** recursos históricos (4 CPU, 16 GiB, 12 h) y límites publicados de partición/QoS comprobados. Con 1 CPU / 8 GiB / 6 h / concurrencia 2 la propuesta encaja nominalmente, pero requiere una prueba de tiempo/RSS antes de una tanda de 76 tareas.
3. **PENDIENTE PREVIO AL ENVÍO:** implementar y validar un generador de fragmentos y un consolidador con marcadores atómicos, verificación de IDs/SHA y reanudación por fragmento/política; no es suficiente lanzar `build_targets.py` con un recorte manual sin manifiesto.
4. **PENDIENTE PREVIO AL ENVÍO:** comprobar duración de un primer fragmento, ocupación RAM y presión de E/S, y ajustar concurrencia y tamaño solamente por criterios de capacidad; no por resultados predictivos ni descarte de sistemas.

**Entrega de esta etapa:** propuesta técnica de ejecución recuperable. Estado experimental de las tres correcciones: **PENDIENTE**.

## 8. Implementación entregada y primer piloto (13/11/2026)

- Ejecutable: [`cils_v3_shards.py`](cils_v3_shards.py), con `init` (freeze de inventario y 76 fragmentos de 10/último 7), `run --shard N --policy include|exclude` (checkpoint **por fragmento y política**, SHA, validación 18 filas/sistema y promociones atómicas), `verify`, `consolidate` (rechaza faltantes, duplicados y discrepancias entre políticas) y `selftest`. Solo el subcomando `run` llama al científico v3 original, sin modificar algoritmo ni criterios.
- Pruebas sintéticas ejecutadas en un contenedor local: [`PRUEBAS_SINTETICAS.md`](PRUEBAS_SINTETICAS.md), [JSON](PRUEBAS_SINTETICAS.json): **11/11 PASS** con interrupción simulada y corrupción de filas/hash. **SCAYLE debe volver a ejecutar estas pruebas sobre el archivo publicado antes del piloto**, para garantizar que la instalación real pasa los mismos controles.
- Piloto de **un único fragmento (shard 000)**: [`cils_v3_pilot.sbatch`](cils_v3_pilot.sbatch), `--account=uva_dma_2 --partition=sapphire --qos=normal --cpus-per-task=1 --mem=8G --time=06:00:00`. Ejecuta primero `include`, después `exclude` con marcadores independientes. El resto de **75 fragmentos NO se envía**.
- Ruta nueva sin sobrescritura de evidencias: `WORK/maintenance/paper2_cils_v3_relaunch_v1/`. Checkpoints verificables: `parts/include_000/complete.json`, `parts/exclude_000/complete.json`, cada uno con CSV, summary, log, SHA-256, walltime y RSS máximo del proceso hijo. Logs Slurm: `logs/pilot_JOBID.{out,err}`.
- **Piloto ATLAS942**: aún NO ejecutado ni acreditado en esta revisión documental. Una vez que SCAYLE informe `COMPLETED 0:0`, ambos `SHARD_PASS` y verificaciones de hashes/filas, se publicarán el registro de Slurm, logs, checkpoints y medidas reales en esta carpeta con el commit resultante.
- La consolidación de los **757 sistemas** solo está autorizada tras los **152 checkpoints** (76 fragmentos × 2 políticas), no tras el piloto.
