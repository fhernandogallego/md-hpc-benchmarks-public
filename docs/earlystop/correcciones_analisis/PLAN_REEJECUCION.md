# EARLYSTOP / CILS v3 — plan de reejecución recuperable (13/11/2026)

**Estado: plan técnico preparado; NO autorizado ni enviado.** Se debe terminar el inventario del `.sbatch`, comprobar el límite/QoS real y verificar la identidad del inventario original antes de ejecutar. Este plan **no es evidencia de resultados corregidos**, y las tres afirmaciones científicas siguen retiradas de la memoria.

## 1. Incidente y gasto computacional

- Job `3344624`, `paper2_cils_v3`, `uva_dma_2` / `sapphire`: `TIMEOUT`, `12:00:28`; paso `batch=CANCELLED`, sin artefactos finales en `WORK/results/paper2_cils_v3_20261112_101652/`. Logs archivados en `WORK/results/paper2_cils_v3_20261112_101652_logs/`; `WORK=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks`.
- `ejecucion_v3.log` registra `[100/757]` como último hito de progreso. El script imprime `[1]`, `[2]`, `[3]` y luego cada 100; **100 no es una estimación del total que llegó a completar**.
- Causa de pérdida de trabajo: [`build_targets.py`](fuente_v3/calculos/build_targets.py) acumula `records` durante todo el inventario y **solo al final** escribe `targets_*.csv`. [`run_revision.sh`](fuente_v3/calculos/run_revision.sh) ejecuta **primero include y después exclude**, antes de cualquier sensibilidad/modelado. Ninguna etapa termina y queda marcada si el primer paso agota el walltime.
- Perfil **deducido del código, no medido con profiler**: 757 sistemas × 3 réplicas × 6 observables = **13.626 valoraciones de objetivos por política**, repetidas para `include/exclude` (27.252). Para core, `detect_equilibration` se llama 2.271 veces por política, además de `statistical_inefficiency`; `pandas.read_csv` carga series por sistema y política. Estas operaciones PyMBAR/IO son los principales candidatos a cuello de botella; confirmarlo con tiempos por fragmento y métricas `sacct`.
- **Tiempo estimado preliminar**: si el ritmo fuera tan bajo como **100 sistemas/12h**, un barrido de 757 podría tomar unas **91 horas CPU/una tarea** (dos políticas ≈182 h); la posición real al TIMEOUT y la dispersión por sistema no se conocen, por lo que **no es una predicción validada**. Un fragmento de 10 sistemas sería ≈1,2 h bajo ese supuesto; se propone solicitar inicialmente **6h por fragmento** (ajustar tras la primera ejecución controlada). Si algún fragmento excede 6h, subdividirlo sin perder los ya correctos.
- **Límite permitido**: todavía **NO ACREDITADO** por `scontrol show partition sapphire` + QoS del job; el `TIMEOUT` a las 12h solo prueba que **a este job** se le impuso ese tiempo, no el `MaxTime` global de la partición. Registrar `sacct -j 3344624 -X -P -o JobID,State,Timelimit,AllocCPUS,ReqMem,Elapsed,MaxRSS`, `sinfo -p sapphire -o '%P|%l|%c|%m|%D'`, `scontrol show partition sapphire -o`, `sacctmgr -nP show qos format=Name,MaxWall,MaxTRES` si el acceso lo permite.
- **Recursos**: cuenta/partición verificadas = `uva_dma_2` / `sapphire`, entorno Python real = `/home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/python3`, con `Python/3.11.5-GCCcore-13.2.0`, NumPy `1.26.4`, Pandas `2.2.3`, PyMBAR `4.0.3`. Los `--cpus-per-task` / `--mem` del job original quedan pendientes de recuperar del `.sbatch`. Propuesta inicial para cada fragmento: **1 CPU / 8 GiB / 6 horas / concurrencia máxima 4**, condicionada a comprobar consumo RSS y QoS; no declarar esos recursos como comprobados.

## 2. Invariantes: nada de modificaciones científicas

- Inventario original de **757 sistemas train**, no selección por facilidad ni reducción de muestra, y `outer_folds.csv` sin cambios. Verificar que todos los IDs de los fragmentos pertenecen a los 757 sistemas de los folds originales, sin duplicados, y que la unión de fragmentos reproduce **byte a byte** el inventario original (excepto división física en archivos).
- Conservar código/configuración original v3, commit fuente [`2cc3d2b4701e62881b8ece6f9fc53ae1725a34dd`](https://github.com/fhernandogallego/md-hpc-benchmarks/commit/2cc3d2b4701e62881b8ece6f9fc53ae1725a34dd), paquete SHA declarado `f05bc1d96b7233ec3243a267de9b1f23c1c683f715e56f48321433095afdb17f`; comprobar igualdad con snapshot ejecutado en SCAYLE. `targets_config.json`: core usa `replica_prefix_rmsd_A` y detecta `t0` propio.
- No abrir ni calcular validation/test. No inventar nueva partición de calibración global: los roles `training/calibration/evaluation` del análisis conformal son subparticiones internas por grupos del entrenamiento existente.
- No reducir `bootstrap_draws=10000`, rejillas, longitudes mínimas, `Neff>=10`, semillas, protocolos, candidatos ni criterios de `constant_policy`. No combinar sensibilidades incompatibles (5/10 sin core versus 20 con core completo).
- El código del paquete v3 es el ejecutor científico. El envoltorio por fragmentos **solo** seleccionará filas del inventario (sin cambiar su orden), lanzará `build_targets.py` con los mismos argumentos y consolidará resultados **después de verificar completitud**.

## 3. Bloque A: reconstrucción de objetivos por sistemas, con checkpoint

**Diseño:** dividir el inventario original, en orden, en **76 archivos** de hasta 10 sistemas; lanzar como array Slurm `--array=0-75%4` con recursos provisionales `--time=06:00:00 --cpus-per-task=1 --mem=8G --account=uva_dma_2 --partition=sapphire`, si `sapphire`/QoS lo permiten. No enviar hasta validar límite y RSS.

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

1. Recuperar y publicar el **.sbatch realmente ejecutado, los logs completos y el manifiesto**, desde SCAYLE, como `evidencia_3344624/`, y extraer el comando exacto.
2. Medir **`MaxTime` de sapphire, QoS efectivo, `Timelimit`, `AllocCPUS`, `ReqMem` y `MaxRSS`**; usar esos datos para ratificar/cambiar `6h/1 CPU/8 GiB/%4`. No atribuir el límite de 12h a la partición si podría ser el solicitado en el `.sbatch`.
3. Resolver la ruta exacta del inventario 757 y del directorio `calculos/` **desplegados** desde el `.sbatch`/preflight, y cotejar SHA-256 con el paquete v3. Preparar y validar sintácticamente los ejecutores y el consolidador *antes* de enviarlos.
4. **Medir duración de un primer fragmento y su RSS** al autorizar la futura ejecución; actualizar estimación de calendario/coste y decidir si reducir a 5 sistemas por fragmento para los casos lentos, **sin cambiar población ni criterios**.

**Entrega de esta etapa:** propuesta técnica de ejecución recuperable. Estado experimental de las tres correcciones: **PENDIENTE**.
