# EARLYSTOP — comprobación de correcciones de Paper 2 CILS (13/11/2026)

**Dictamen: tres correcciones IMPLEMENTADAS en el código v3; ejecución ATLAS942 corregida y resultados NO ACREDITADOS en esta revisión.** No se ha lanzado ningún análisis nuevo ni abierto resultados de validation/test. Los tres bloques de resultados anteriores deben permanecer retirados de la memoria hasta disponer de una ejecución corregida, verificable y con resultados trazables.

## Proveniencia

- Revisión local fechada el 10/11/2026: `paper/paper 2/cils/revision/LEEME_REVISION.txt`, repo privado `fhernandogallego/md-hpc-benchmarks`.
- Commit exacto del **paquete fuente**: [`2cc3d2b4701e62881b8ece6f9fc53ae1725a34dd`](https://github.com/fhernandogallego/md-hpc-benchmarks/commit/2cc3d2b4701e62881b8ece6f9fc53ae1725a34dd), publicado el 10/11/2026. No atribuirlo al código ejecutado sin contrastar hashes del snapshot de SCAYLE.
- Paquete `paquete_revision_scayle_v3.tar.gz`: SHA-256 declarado en el manifiesto original `f05bc1d96b7233ec3243a267de9b1f23c1c683f715e56f48321433095afdb17f`; no verificado contra el archivo desplegado en Slurm aquí.
- Fuentes originales en el repo privado: [`calculos/`](https://github.com/fhernandogallego/md-hpc-benchmarks/tree/2cc3d2b4701e62881b8ece6f9fc53ae1725a34dd/paper/paper%202/cils/revision/calculos). Se conservan enlaces exactos a scripts/configuración, sin afirmar que constituyan resultados.

## 1. Prefijos 5/10 ns, sin fuga de core de 20 ns

- [`reanalyse.py`](fuente_v3/calculos/reanalyse.py) `--mode prefix`: `SAFE_CHANNELS=(0,1,5,6,7)` en **5, 10 y 20 ns**; excluye los tres canales derivados del core definido con 20 ns. El mismo código recorta la serie hasta el horizonte evaluado. Esta sensibilidad tiene 30 resúmenes, **no** los 48 originales.
- [`temporal_controls.py`](fuente_v3/calculos/temporal_controls.py) restringe los canales para 5/10 ns; **el `ridge_all_full` de 20 ns sí conserva core**, mientras que `ridge_without_core_full` lo excluye. No confundir la comparación de ablaciones con la sensibilidad homogénea sin core de `reanalyse.py`.
- Estado: **código corregido, sin ejecución real auditada**. Los resultados originales de 5/10 ns no se pueden reincorporar.

## 2. Objetivos core RMSD con equilibrado propio

- [`targets_config.json`](fuente_v3/calculos/targets_config.json) especifica `core_rmsd_q30`, `core_R{r}_csv`, columna `replica_prefix_rmsd_A` y `equilibration="detect"`.
- [`build_targets.py`](fuente_v3/calculos/build_targets.py) usa `pymbar.timeseries.detect_equilibration` sobre la serie core y calcula objetivo futuro en `t>20 ns`, `t>=t0_core`, mínimo 20 ns y `Neff>=10`. Para serie constante, usa convención `g=1`, incluye/excluye en dos sensibilidades; no implica independencia estadística.
- Estado: **reconstrucción definida en código, nuevos targets ATLAS no verificados**. No sustituir resultados anteriores de core hasta verificar archivos de targets y metadatos.

## 3. Calibración externa por grupos

- [`reanalyse.py`](fuente_v3/calculos/reanalyse.py) `--mode intervals`: `partition()` divide grupos `training/calibration/evaluation` sin solapamiento; ~20% de grupos no evaluados se destinan a calibración, el resto a entrenamiento. La selección de alpha, fit, normalización, suelo de escala y umbrales de estratos usan solamente training propio; los residuos de calibración usan un modelo ajustado fuera de los grupos calibrados/evaluados.
- La cobertura por sistema con residuos absolutos es diagnóstico bajo dependencia; variantes por máximo de grupo requieren intercambiabilidad. No hay cobertura ni anchura corregida **confirmada en ATLAS**.
- Estado: **protocolo implementado, resultados y manifiestos de una ejecución real no verificados**.

## Trabajo posterior (NO verificado)

- Referencia histórica de envío: Slurm **3344624**, modo `full`, enviado el 12/11/2026. Último estado conocido anteriormente: `PD`; **no se ha comprobado su estado actual ni su salida**.
- Posibles rutas del trabajo: `WORK/results/paper2_cils_v3_20261112_101652` y `WORK/results/paper2_cils_v3_20261112_101652_logs`, con `WORK=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks`. Rutas indicativas, **sin confirmación de existencia de resultados finales**.
- Entrada de referencia: `WORK/results/paper2_automation/auto_20261030_091302/training`.
- Entry point **documentado**, no comando Slurm realmente recuperado: `bash calculos/run_revision.sh RUN_ATLAS942 INVENTARIO.csv OUTPUT_NUEVO`. Comando `sbatch` real, recursos, log y hash del snapshot ejecutado: **NO ACREDITADOS**.

## Matriz de resultados que se solicitaron

| Evidencia | 1. Prefijos | 2. Core RMSD | 3. Calibración |
|---|---|---|---|
| Predicciones y etiquetas | NO RECUPERADAS (se esperan `prefix_without_core/replica_predictions.csv` y/o `temporal_controls_rmsd/{replica_predictions,system_predictions}.csv`) | NO RECUPERADAS (`intervals_core_rmsd_q30/replica_predictions.csv`) | NO RECUPERADAS (`intervals_*/replica_predictions.csv`) |
| Métricas, contrastes, IC | NO RECUPERADOS | NO RECUPERADOS | NO RECUPERADOS |
| Grupos/roles de evaluación | NO VERIFICADOS (se espera `outer_fold_manifest.csv`) | NO VERIFICADOS | NO VERIFICADOS (se espera `partition_manifest.csv`) |
| Cobertura y anchura global/estratos con n | No aplica | NO RECUPERADAS | NO RECUPERADAS (`coverage.csv` e `intervals.csv`) |
| Objetivos corregidos y provenance | Requiere verificar el archivo de targets usado | NO RECUPERADOS (se esperan `targets_include_constants.csv`, `targets_exclude_constants.csv` y sus resúmenes) | NO RECUPERADOS |

Los nombres de archivos son **salidas esperadas por el código**, no evidencia de que existan. No hay comprobación de salida `COMPLETED`/exit 0, log `PASS`, identidades/roles, valores y hashes finales v3.

## Conclusión: sustitución posible

1. Resultado anterior 5/10 ns con fuga: **retirado; no sustituido por valores nuevos verificables**.
2. Objetivo anterior core con `t0` heredado: **retirado; v3 ofrece algoritmo correcto, pero no se ha verificado ejecución ATLAS**.
3. Intervalos/cobertura con calibración contaminada: **retirados; no sustituibles sin comprobar una ejecución y sus manifiestos/estadísticos**.

**Decisión:** mantener las tres evidencias **fuera** de la memoria; no abrir test, crear particiones ni ejecutar análisis nuevos solo para responder a esta solicitud.
