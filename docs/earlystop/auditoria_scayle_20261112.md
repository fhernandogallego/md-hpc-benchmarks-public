# Evidencias de auditoría SCAYLE — EARLYSTOP (12-11-2026)

**Fuente principal de conclusiones:** [cierre_diego_miti_20261112.md](cierre_diego_miti_20261112.md). Esta página es el índice técnico de las tres comprobaciones de SCAYLE, sin consultar resultados de test.

## Ejecución y metadatos

- Slurm 3262490: COMPLETED, 0:0, cuenta `uva_dma_2`, partición `sapphire`.
- RUN: `/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/results/paper2_automation/auto_20261030_091302`.
- Código básico efectivo `pipeline_code/src/pipeline/extract_atlas_basic.py` SHA256 `f739b02858a6517820f0747fd39706296d75dc39706fb7bad82af7d4b0bd2620` y `pipeline_manifest.json`: SHA256_OK.
- Piloto `full_v2_3262025/prueba_atlas_v2.py`: SHA256 `0b3304050c7e7243f96cc8cae2f84f2f88b5e2905d21d753e0f1417fff909343`; congelado también en `audit_v2_500_3268243/scripts/prueba_atlas_v2.py`.
- Piloto `full_v2_3262025/launcher.sbatch`: SHA256 `153da31c9a2832b1c728202b4c034375e0bf1d68f827a55ac68f0610f4b4e794`; idéntico a `scripts/original_launcher.sbatch` de la auditoría posterior.
- Config real ATLAS942: `project=/mnt/lustre_fs/_HPC/SCRATCH/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/results/paper2_hito0/wave3_remaining117_20261014_v1/code`.

## Primera auditoría (11:39)

`WORK/maintenance/earlystop/auditoria_20261112_113942/`

- `INFORME.txt`: código, hashes, recuentos y particiones.
- `recuentos_por_observable.csv`: train924 y subconjunto 757.
- `motivos_por_observable.csv`: conteos por causa y observable.
- `detalle_no_retenidos.csv`: detalle por sistema y réplica de los objetivos del master no retenidos.
- `estados_rmsd_futuro.csv`: comparación de validez.
- `manifiestos_particiones.csv`: inventario de manifiestos y sumas de comprobación.
- `incidencias_master.csv`, `sistemas_sin_datos_completos.csv`: incidencias y ausencia de datos.
- El universo de cinco observables del master tenía 924/924 archivos presentes.

## Diagnóstico intermedio

`WORK/maintenance/earlystop/diagnostico_pendientes.txt`: muestra dónde están piloto, posibles tablas core y repositorios; búsqueda inicial de `calibration` sin resultados en las rutas inspeccionadas.

## Segunda auditoría (12:05)

`WORK/maintenance/earlystop/pendientes_20261112_120511/`

- `INFORME_v2.txt`: código del piloto, disponibilidad core y procedencia de particiones.
- `core_q30_disponibilidad.csv`: 924 registros de train; identifica cuál sistema no tiene archivo core y el detalle de réplicas/series por sistema. **No contiene validez estadística de objetivos.**
- `incidencias_core.csv`: incidencias de lectura; que esté vacío **no** implica cobertura completa (existe un sistema sin archivo).
- Para 924 sistemas train, q30 `replica_prefix`: 923 sistemas, 2769 réplicas con resumen/serie.
- Para 757 sistemas con RMSD futuro etiquetado: 756 sistemas, 2268 réplicas con resumen/serie.
- Etiquetas RMSD: 1519 `original=false/future=false`, 10 `true/false`, 1243 `true/true`.

## Partición congelada

`WORK/results/paper2_automation/auto_20261030_091302/cohort/system_assignment.csv` SHA256 `bb12305d7ed029ec7c072e1789b357c77420ef35aeca39ed69469217a7109d37`.

- Train: 924 sistemas, 817 grupos.
- Validation: 9 sistemas, 9 grupos.
- Test: 9 sistemas, 9 grupos.
- Grupos compartidos: 0.
- **No hay** prueba de manifiesto `development/calibration/test`; no se accedió a resultados de test.
- `git` no existe en PATH en la sesión de SCAYLE; el commit generador sigue `NO ACREDITADO`. Commit de la publicación ATLAS942: `aeac56891e06bb002d50ebdd3a1068c84fd2c600`.

## Trazabilidad adicional

Los conteos/motivos resumidos se publican además como CSV en `docs/earlystop/`. Los CSV granulares originales **permanecen en SCAYLE**. No se ha recalculado ningún observable.
