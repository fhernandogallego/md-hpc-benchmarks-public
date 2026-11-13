# CILS v3: pruebas en SCAYLE y envío de piloto 3387582

Fecha de registro: **13/11/2026**. Fuente: salida de terminal CALÉNDULA aportada tras ejecutar el instalador; este documento **no implica** lectura independiente de los archivos en SCAYLE. **No se han abierto validation/test ni enviado los 75 fragmentos restantes.**

## Entorno y distribución

- `ENTORNO=PASS`: Python **3.11.5**, NumPy **1.26.4**, Pandas **2.2.3**, PyMBAR **4.0.3**.
- Fallos previos solucionados antes del lanzamiento: dependencia de `libpython3.11.so.1.0` mediante módulo EasyBuild; límite de hilos OpenBLAS `OPENBLAS_NUM_THREADS=1`; validación HTTPS desde CALÉNDULA usando `certifi/cacert.pem`, con `TLS_GITHUB=PASS`. No se desactivó la validación TLS.
- Fuentes instaladas con hash de Git verificado: `cils_v3_shards.py` (26.065 bytes) y `cils_v3_pilot.sbatch` (1.571 bytes). La comparación de hashes del paquete científico v3 se realiza en `init`, sobre la copia conservada de la ejecución anterior.
- `INIT_PASS systems=757 shards=76 chunk=10`: inventario train congelado, 76 fragmentos; 10 sistemas en el primero y siete en el último. No implica que la reconstrucción molecular haya terminado.

## Pruebas sintéticas efectivamente ejecutadas en SCAYLE

**`SELFTEST=PASS`, 11/11 comprobaciones:**

```text
TEST_plan_5_systems_3_shards=PASS
TEST_interruption_detected=PASS
TEST_completed_shard_preserved=PASS
TEST_partial_shard_has_no_checkpoint=PASS
TEST_resume_skips_verified_result=PASS
TEST_missing_shards_detected=PASS
TEST_duplicate_row_detected=PASS
TEST_missing_row_detected=PASS
TEST_hash_tampering_detected=PASS
TEST_merge_complete=PASS
TEST_merge_resume_verifies=PASS
```

Se construyeron 3 fragmentos artificiales (5 sistemas sintéticos y 90 filas por política); una interrupción artificial dejó el fragmento anterior intacto. La reanudación confirmó `SHARD_REUSED=PASS`; la consolidación válida informó `MERGE_PASS systems=5 rows_per_policy=90`. Memoria sintética registrada ≈16–17 MiB, **no representa el consumo de PyMBAR sobre ATLAS942**.

Registro JSON en SCAYLE:
`/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/earlystop/SELFTEST_CILS_V3_SCAYLE.json`.

## Envío real del piloto

```text
PILOT_SUBMITTED_JOB=3387582
PILOT_STDOUT=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/pilot_3387582.out
PILOT_STDERR=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/paper2_cils_v3_relaunch_v1/logs/pilot_3387582.err
squeue: JOBID=3387582 PARTITION=sapphire ST=PD TIME=0:00 NODES=1
```

- Lanzador: [`cils_v3_pilot.sbatch`](cils_v3_pilot.sbatch), cuenta `uva_dma_2`, partición `sapphire`, QoS `normal`; **1 CPU, 8 GiB, 6 h**.
- Contenido: **solo shard 000 (10 sistemas de entrenamiento)**, primero `include`, luego `exclude`; checkpoints individuales `parts/include_000` y `parts/exclude_000`.
- **Estado conocido en el momento de la captura: PENDIENTE (PD)**. No se aportó posteriormente `sacct`, log del trabajo finalizado, dos `SHARD_PASS`, duración real, RSS real ni comprobaciones de resultados ATLAS. **No calificarlos como PASS**.
- La validación del primer fragmento consiste en 180 filas esperadas por política, esquema de seis observables × tres réplicas, hashes de `targets.csv`, `targets.summary.json` y `build.log`, `n_problems=0` y estado `COMPLETED 0:0` del trabajo y paso batch.
- Verificador tras la salida del job: `bash /scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/earlystop/verificar_piloto_cils_v3.sh`.
- Para recuperar medidas del piloto: `sacct -j 3387582 -P --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS` y los `complete.json` de ambos modos.

## Estado científico

**Pruebas de recuperación: PASS en SCAYLE. Piloto ATLAS942: ENVIADO, resultado pendiente de acreditación.** No se han generado ni confirmado análisis OOF de prefijos, core ni intervalos; las tres evidencias siguen retiradas de la memoria. No se recomienda enviar los 75 fragmentos restantes hasta revisar la duración y la memoria reales del piloto.
