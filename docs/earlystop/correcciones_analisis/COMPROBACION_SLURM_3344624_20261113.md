# Comprobación de Slurm 3344624 — 13/11/2026

**Fuente:** salida de la auditoría de solo lectura ejecutada en SCAYLE el 13/11/2026 a las 09:33, proporcionada por el responsable de la ejecución. Se documentan las líneas de `sacct`, preflight, logs y el inventario de productos; **no se adjuntan los logs íntegros**, que permanecen en SCAYLE. No se ejecutó ningún análisis ni se accedió a validation/test para esta comprobación.

## Dictamen

**El run `full` de CILS v3 NO finalizó: `TIMEOUT` tras doce horas.** No se encontró ninguna de las salidas corregidas esperadas en el directorio de resultados de la ejecución. Por consiguiente, no se dispone de métricas, predicciones, intervalos ni calibración verificados que sustituyan las tres evidencias retiradas de la memoria.

## Scheduler y recursos

```text
sacct -j 3344624 -P --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,Account,Partition
3344624|paper2_cils_v3|TIMEOUT|0:0|2026-11-12T10:18:56|2026-11-12T22:19:24|12:00:28|uva_dma_2|sapphire
3344624.batch|batch|CANCELLED|0:15|2026-11-12T10:18:56|2026-11-12T22:19:26|12:00:30|uva_dma_2|
3344624.extern|extern|COMPLETED|0:0|2026-11-12T10:18:56|2026-11-12T22:19:26|12:00:30|uva_dma_2|
```

El código `0:0` en la fila de asignación principal **no indica éxito**: prevalece `TIMEOUT` del trabajo y la cancelación del paso `batch`. `squeue` y `scontrol show job` responden `Invalid job id specified` después de su salida de la cola.

## Preflight satisfactorio (pero NO ejecución final)

Del archivo `preflight.log`:

```text
PACKAGE_V3_SHA256=PASS
Verified files: 22
PYTHON_ENV=PASS /home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/python3
Python: 3.11.5
NumPy: 1.26.4
Pandas: 2.2.3
PyMBAR: 4.0.3
INVENTORY_AUDIT: {"n_inventory": 757, "n_training_folds": 757, "n_folds_outside_inventory": 0, "n_core_csv_checked": 2271, "n_errors": 0, "errors": []}
INVENTORY_PREFLIGHT=PASS
MODO_SELECCIONADO=full
V3_SUBMITTED_JOB=3344624
```

La verificación SHA del paquete (22 archivos) demuestra preflight aprobado, **no que cada producto del análisis se haya generado**. El commit de origen de los scripts es `2cc3d2b4701e62881b8ece6f9fc53ae1725a34dd`; no se ha extraído aquí un commit explícito del snapshot desplegado. La falta de JAX aparece como aviso informativo de PyMBAR, **no como error de la ejecución**.

## Logs y progreso

- `ejecucion_v3.log`: últimos hitos impresos `[1/757] 1a62_A`, `[2/757] 1ah7_A`, `[3/757] 1aol_A`, `[100/757] 1skz_A`. El último mensaje de progreso visible **no permite precisar cuánto trabajo se completó entre dos mensajes**.
- `slurm_3344624.err`:
  ```text
  [2026-11-12T22:19:25.008] error: *** JOB 3344624 ON cn7191 CANCELLED AT 2026-11-12T22:19:25 DUE TO TIME LIMIT ***
  ```
- `slurm_3344624.out`: `JOB_ID=3344624`, `MODO=full`, entorno Python y versiones. No consta marcador de éxito del análisis completo.
- **Ningún error científico o de programa demostrado antes de la cancelación.** La causa registrada del cese es el límite temporal; no queda demostrado qué operación interna produjo la mayor demora.

## Rutas SCAYLE verificadas

Prefijo `WORK=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks`.

- Carpeta de resultados existente, sin productos relevantes encontrados: `WORK/results/paper2_cils_v3_20261112_101652/`.
- Logs existentes:
  - `WORK/results/paper2_cils_v3_20261112_101652_logs/ejecucion_v3.log`
  - `WORK/results/paper2_cils_v3_20261112_101652_logs/slurm_3344624.out`
  - `WORK/results/paper2_cils_v3_20261112_101652_logs/slurm_3344624.err`
- Script de envío/configuración archivado: `WORK/maintenance/paper2_cils_v3_20261112_101652/paper2_cils_v3.sbatch` (existencia y tamaño 2919 bytes comprobados; **contenido/comando `sbatch` no inspeccionados**).
- Preflight y manifiesto: `WORK/maintenance/paper2_cils_v3_20261112_101652/preflight.log`, `paquete_revision_scayle_v3.manifest.json`.
- Registro de auditoría de solo lectura: `WORK/maintenance/earlystop/comprobacion_cils_v3_3344624.log` (ruta prevista por el script que imprimió la salida; no se realizó una lectura independiente de este archivo desde GitHub).

## Productos corregidos buscados y NO encontrados

La comprobación de la carpeta de resultados indicó `ARCHIVOS_RELEVANTES: 0` y `NO_EN_RAIZ` para todos los siguientes: `targets_include_constants.csv`, `targets_exclude_constants.csv`, `rmsd_reference_check.json`, `prefix_without_core/`, `temporal_controls_rmsd/`, `intervals_rmsd/`, `intervals_core_rmsd_q30/`, `RESUMEN_BASELINES_ABLACION.txt`.

**No existen resultados reportados** para comparar predicción/etiqueta, MAE, contrastes, IC, `partition_manifest.csv`, cobertura, anchura y recuentos por estratos de la ejecución 3344624. No inferirlos a partir de resultados anteriores.

## Estado científico

1. **Prefijos 5/10:** corrección implementada en v3, sin resultados reales verificables por cancelación del job.
2. **Core RMSD:** código de `t0` propio, sin CSV de objetivos reconstruidos ni validez comprobada en el run.
3. **Calibración:** separación implementada en código, sin predicciones/roles/intervalos/estadísticos de cobertura verificados.

**Mantener las tres evidencias fuera de la memoria.** No se requieren simulaciones ni más análisis para *responder al estado* de esta ejecución. Una futura restauración de las evidencias exigiría una ejecución corregida y verificable; esta auditoría **no la solicita ni la inicia**.
