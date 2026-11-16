# EARLYSTOP — índice de entregables y ubicaciones (12-11-2026)

**Estado:** análisis documental de los puntos 1–3 completado con límites declarados. La evaluación de test no se ha ejecutado ni abierto.

## A. Documentos publicados en GitHub

Repositorio: `fhernandogallego/md-hpc-benchmarks-public`
Rama: `earlystop-cierre-diego-miti-20261112`

| Documento | Ruta en el repositorio | Finalidad |
|---|---|---|
| Informe de cierre | `docs/earlystop/cierre_diego_miti_20261112.md` | Conclusiones, comparación 500 vs 757, recuentos, limitaciones |
| Auditoría SCAYLE | `docs/earlystop/auditoria_scayle_20261112.md` | Evidencia y hashes de dos auditorías |
| SAXS experimental | `docs/earlystop/ficha_saxs_alfa_sinucleina_20261112.md` | DOI, datos, incertidumbres y compatibilidad pendiente |
| Ficha OneDrive (texto) | `docs/earlystop/Respuesta_Diego_Miti_cierre.txt` | Entrega condensada |
| Índice de entregables | `docs/earlystop/INDICE_ENTREGABLES.md` | Este archivo |
| Recuentos CSV | `docs/earlystop/recuentos_observables_verificados_20261112.csv` | Recuentos consolidados por universo |
| Motivos CSV | `docs/earlystop/motivos_exclusion_verificados_20261112.csv` | Desglose INVALID_* por observable |
| Core CSV | `docs/earlystop/core_q30_resumen_20261112.csv` | Disponibilidad, **no** validez |
| Partición CSV | `docs/earlystop/manifiesto_particiones_resumen_20261112.csv` | 924/9/9 sistemas y 817/9/9 grupos |

Origen de publicación ATLAS942: `paper2-atlas942-results-20261103`, commit `aeac56891e06bb002d50ebdd3a1068c84fd2c600`. Este es el commit público de los resultados, **no el commit generador del manifiesto**. Para el commit documental vigente, consultar HEAD de la rama de cierre (no incluir el propio SHA en una versión de archivo que produciría otro commit).

## B. Evidencia original en SCAYLE

Definir `WORK=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks`.

| Evidencia | Ruta absoluta (tras prefijo WORK/) |
|---|---|
| Primera auditoría | `maintenance/earlystop/auditoria_20261112_113942/` |
| Informe primera auditoría | `maintenance/earlystop/auditoria_20261112_113942/INFORME.txt` |
| Recuentos originales | `maintenance/earlystop/auditoria_20261112_113942/recuentos_por_observable.csv` |
| Motivos originales | `maintenance/earlystop/auditoria_20261112_113942/motivos_por_observable.csv` |
| Exclusiones por réplica | `maintenance/earlystop/auditoria_20261112_113942/detalle_no_retenidos.csv` |
| Cruce validez RMSD | `maintenance/earlystop/auditoria_20261112_113942/estados_rmsd_futuro.csv` |
| Manifiestos detectados | `maintenance/earlystop/auditoria_20261112_113942/manifiestos_particiones.csv` |
| Incidencias master | `maintenance/earlystop/auditoria_20261112_113942/incidencias_master.csv` |
| Sistemas incompletos | `maintenance/earlystop/auditoria_20261112_113942/sistemas_sin_datos_completos.csv` |
| Diagnóstico intermedio | `maintenance/earlystop/diagnostico_pendientes.txt` |
| Segunda auditoría | `maintenance/earlystop/pendientes_20261112_120511/` |
| Informe segunda auditoría | `maintenance/earlystop/pendientes_20261112_120511/INFORME_v2.txt` |
| Disponibilidad q30 sistema a sistema | `maintenance/earlystop/pendientes_20261112_120511/core_q30_disponibilidad.csv` |
| Incidencias lectura q30 | `maintenance/earlystop/pendientes_20261112_120511/incidencias_core.csv` |
| Script primera auditoría | `maintenance/earlystop/audit_earlystop_1_3.py` |
| Script segunda auditoría | `maintenance/earlystop/audit_earlystop_pendientes.py` |
| Run ATLAS942 | `results/paper2_automation/auto_20261030_091302/` |
| Manifiesto original | `results/paper2_automation/auto_20261030_091302/cohort/system_assignment.csv` |
| Copia congelada | `releases/paper2_atlas942_20261103_v1/cohort/system_assignment.csv` |
| Piloto 500 | `results/pid_atlas_poc/full_v2_3262025/` |
| Auditoría piloto | `results/pid_atlas_poc/audit_v2_500_3268243/` |

**Los archivos detallados/grandes no se han copiado a GitHub**; sus rutas y SHA-256 de fuentes principales figuran en los informes. Las rutas son de la misma instalación SCAYLE, no copias de datos de test. No modificar los bundles congelados.

## C. OneDrive

Archivo `Respuesta_Diego_Miti_cierre.txt` (copia de referencia en GitHub arriba): **subida a OneDrive confirmada por el usuario el 12-11-2026**. La ubicación exacta en OneDrive y la presencia del archivo no se han verificado de forma independiente desde esta conversación.

## D. No acreditado

- Validez y motivos de exclusión del objetivo core RMSD q30: solo disponibilidad física de series.
- ID y motivo del único sistema que carece de core q30: localizable en `core_q30_disponibilidad.csv`.
- Manifiesto independiente de `development/calibration/test`: no hallado en las rutas inspeccionadas, sin equivalencia autorizada `validation = calibration`.
- Commit de **generación** del CSV de asignación: no acreditado. `git` ausente del PATH de la sesión de SCAYLE.
- Compatibilidad exacta entre el SAXS monomérico y una simulación propia: no confirmada.
