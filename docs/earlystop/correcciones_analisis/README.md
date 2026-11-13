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

## Trabajo posterior: `3344624` — TIMEOUT confirmado

La **auditoría de SCAYLE del 13/11/2026** ha comprobado el estado del trabajo `paper2_cils_v3` lanzado el 12/11/2026. **NO TERMINÓ**: `sacct` devuelve `State=TIMEOUT`, `Elapsed=12:00:28`; el paso `3344624.batch` aparece como `CANCELLED (0:15)`. El log de error indica explícitamente `DUE TO TIME LIMIT`. El `ExitCode=0:0` del registro principal no demuestra éxito.

- Preflight **PASS**: paquete v3 (22 ficheros verificados), Python 3.11.5, NumPy 1.26.4, Pandas 2.2.3, PyMBAR 4.0.3; inventario de 757 sistemas con 2271 CSV de core revisados sin error.
- Log de procesamiento: últimos mensajes `[1/757]`, `[2/757]`, `[3/757]`, `[100/757]`. No consta progreso final ni `PASS` del flujo.
- Directorio de resultados `WORK/results/paper2_cils_v3_20261112_101652/` **existe**, pero la auditoría encontró **0 productos corregidos relevantes**. No existen en la salida verificada `targets_include_constants.csv`, `prefix_without_core/`, `intervals_rmsd/`, `intervals_core_rmsd_q30/` ni `coverage.csv`.
- Logs: `WORK/results/paper2_cils_v3_20261112_101652_logs/{ejecucion_v3.log,slurm_3344624.out,slurm_3344624.err}`. Preflight y script `paper2_cils_v3.sbatch` existentes bajo `WORK/maintenance/paper2_cils_v3_20261112_101652/`.
- El script de envío Slurm está **localizado pero su contenido no se recuperó** en esta comprobación; el comando realmente ejecutado no se atribuye sin inspeccionarlo. El entrypoint documentado de v3 sigue siendo `bash calculos/run_revision.sh RUN_ATLAS942 INVENTARIO.csv OUTPUT_NUEVO`.
- **Evidencia de ejecución:** [`COMPROBACION_SLURM_3344624_20261113.md`](COMPROBACION_SLURM_3344624_20261113.md) y [`ESTADO_EJECUCION.md`](ESTADO_EJECUCION.md).

La cancelación documentada es por límite de tiempo. No se ha demostrado un fallo metodológico de v3 ni tampoco que las tres correcciones hayan producido resultados ATLAS válidos. La no disponibilidad de JAX aparece como advertencia de rendimiento de PyMBAR, no como excepción fatal.

## Evidencia original y plan de reejecución (13/11/2026)

- [`evidencia_3344624/`](evidencia_3344624/): **.sbatch ejecutado**, logs íntegros, preflight, manifiesto v3 y [`DIAGNOSTICO_RECURSOS.txt`](evidencia_3344624/DIAGNOSTICO_RECURSOS.txt). SHA-256 del tar v3 verificado; hashes de cinco fuentes desplegadas coinciden con el manifiesto.
- [`PLAN_REEJECUCION.md`](PLAN_REEJECUCION.md): reconstrucción por **76 fragmentos de hasta 10 sistemas**, dos políticas con checkpoints verificados; después, prefijos sin core y calibración por observable como trabajos separados, **sin envío**.
- Recursos y límites verificados: `#SBATCH --cpus-per-task=4 --mem=16G --time=12:00:00 --qos=normal` en el trabajo anterior; `sapphire MaxTime=UNLIMITED`; QoS `normal MaxWall=5-00:00:00`. El límite que detuvo `3344624` fue el **solicitado de 12h**. Recursos tentativos nuevos: `1 CPU / 8 GiB / 6 h / simultaneidad 2`, **por medir**, no recursos ya validados para el nuevo flujo.
- Diagnóstico de coste: dos reconstrucciones consecutivas leían 757 sistemas y generaban CSV solo al terminar cada pasada; el `TIMEOUT` ocurrió antes del primer CSV final. El último `[100/757]` no cuantifica progreso posterior con precisión.
- **Falta antes de cualquier envío:** crear y verificar el envoltorio de fragmentación, consolidación y marcadores atómicos, y medir duración/RSS del primer fragmento. Se prohíbe relanzar la orden `full` original de forma monolítica con la expectativa de reanudación.

## Matriz de resultados que se solicitaron

| Evidencia | 1. Prefijos | 2. Core RMSD | 3. Calibración |
|---|---|---|---|
| Predicciones y etiquetas | NO RECUPERADAS (se esperan `prefix_without_core/replica_predictions.csv` y/o `temporal_controls_rmsd/{replica_predictions,system_predictions}.csv`) | NO RECUPERADAS (`intervals_core_rmsd_q30/replica_predictions.csv`) | NO RECUPERADAS (`intervals_*/replica_predictions.csv`) |
| Métricas, contrastes, IC | NO RECUPERADOS | NO RECUPERADOS | NO RECUPERADOS |
| Grupos/roles de evaluación | NO VERIFICADOS (se espera `outer_fold_manifest.csv`) | NO VERIFICADOS | NO VERIFICADOS (se espera `partition_manifest.csv`) |
| Cobertura y anchura global/estratos con n | No aplica | NO RECUPERADAS | NO RECUPERADAS (`coverage.csv` e `intervals.csv`) |
| Objetivos corregidos y provenance | Requiere verificar el archivo de targets usado | NO RECUPERADOS (se esperan `targets_include_constants.csv`, `targets_exclude_constants.csv` y sus resúmenes) | NO RECUPERADOS |

Los nombres de archivos son **salidas esperadas por el código**, no evidencia de que existan. Sí se ha comprobado el estado de ejecución: **TIMEOUT por límite de 12 horas**, paso batch cancelado; **sin** log `PASS`, ni predicciones, identidades/roles, métricas, intervalos o hashes de productos finales v3.

## Conclusión: sustitución posible

1. Resultado anterior 5/10 ns con fuga: **retirado; no sustituido por valores nuevos verificables**.
2. Objetivo anterior core con `t0` heredado: **retirado; v3 ofrece algoritmo correcto, pero no se ha verificado ejecución ATLAS**.
3. Intervalos/cobertura con calibración contaminada: **retirados; no sustituibles sin comprobar una ejecución y sus manifiestos/estadísticos**.

**Decisión tras auditoría del 13/11/2026:** mantener las tres evidencias **fuera** de la memoria. Esta ejecución es **incompleta por TIMEOUT** y no ofrece ningún resultado corregido verificable. No abrir test, crear particiones ni ejecutar análisis nuevos solo para responder al estado del trabajo.
