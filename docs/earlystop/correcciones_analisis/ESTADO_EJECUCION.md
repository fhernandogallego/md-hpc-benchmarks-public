# Estado de ejecución CILS v3 — actualizado el 13/11/2026

**Dictamen:** trabajo Slurm `3344624` **TIMEOUT**, no completó el flujo `full`. Esto descarta una validación experimental exitosa de cualquiera de las tres correcciones mediante esta ejecución. Evidencia detallada: [`COMPROBACION_SLURM_3344624_20261113.md`](COMPROBACION_SLURM_3344624_20261113.md).

| Dato requerido | Comprobación |
|---|---|
| Commit fuente v3 | `2cc3d2b4701e62881b8ece6f9fc53ae1725a34dd`; fuentes archivadas en `fuente_v3/` |
| Integridad de paquete en SCAYLE | `PACKAGE_V3_SHA256=PASS`; 22 archivos verificados en `preflight.log` |
| Script Slurm | `WORK/maintenance/paper2_cils_v3_20261112_101652/paper2_cils_v3.sbatch` localizado; contenido no leído en esta auditoría |
| Comando exacto `sbatch` | **NO RECUPERADO**; envío/ID confirmados |
| Cuenta y partición | `uva_dma_2`, `sapphire` |
| Trabajo | `3344624`, modo `full`; `TIMEOUT`, duración `12:00:28`; `batch=CANCELLED (0:15)` |
| Log de ejecución | **DISPONIBLE**, `..._logs/ejecucion_v3.log`; último progreso `[100/757]` |
| Log de cancelación | **DISPONIBLE**, `..._logs/slurm_3344624.err`; `DUE TO TIME LIMIT` |
| Preflight del inventario | **PASS**, 757 sistemas y 2271 rutas core comprobadas |
| Resultados corregidos de prefijos | **NO GENERADOS/NO ENCONTRADOS** en la salida verificada |
| Nuevos targets core | **NO GENERADOS/NO ENCONTRADOS** en la salida verificada |
| Manifiestos de roles `training/calibration/evaluation` | **NO ENCONTRADOS** |
| Predicciones y etiquetas | **NO ENCONTRADAS** |
| Métricas, contrastes e IC | **NO ENCONTRADOS** |
| Cobertura y anchura global, por estratos y recuentos | **NO ENCONTRADOS** |
| Validation/test | No consultados |

El `0:0` del registro principal del job **no es éxito**, ya que `State=TIMEOUT` y el `batch` fue cancelado. El preflight y la sintaxis/código reparados no equivalen a haber evaluado ATLAS con un resultado final.

**Conclusión:** no reintroducir a la memoria las tres evidencias retiradas. No se necesita ejecutar análisis adicionales para documentar el estado de este trabajo. Consultar también [`README.md`](README.md).

`WORK=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks`.

## Actualización 16/11/2026 — piloto de reanudación

El piloto `3387582` (fragmento 000, 10 sistemas, políticas `include` y `exclude`) terminó **`COMPLETED 0:0`** en 01:36:49, con 180 filas por política y 131,6 MiB de memoria máxima; el verificador de salida da `PILOT_VALIDATED=PASS` y `PAIR_VALIDATED=PASS` (hashes, esquema, orden y `n_problems=0`). Mide solo la construcción de objetivos. Los 75 fragmentos restantes **no** se han lanzado y validation/test no se han abierto. Detalle, límites y propuesta de recursos: [`PILOTO_3387582_RESULTADO.md`](PILOTO_3387582_RESULTADO.md).

## Actualización 18/11/2026 — bloque A (objetivos de los 757 sistemas)

Array `3406133` (150/150 `COMPLETED`) y consolidación `3406134` (`MERGE_PASS systems=757 rows_per_policy=13626`). Comprobación de RMSD frente a la referencia `PASS` (máscara idéntica, diferencia 0,0 Å). Recuentos idénticos a la auditoría salvo el **core**, que pasa de 1 136 a 947 réplicas válidas por su equilibrado propio. **No hay aún resultados científicos corregidos** (calibración, prefijos, baselines, core). Detalle en [`BLOQUE_A_RESULTADO.md`](BLOQUE_A_RESULTADO.md).

## Actualización 18/11/2026 — fase 2, primera ronda (RMSD)

`intervals_rmsd`, `prefix_rmsd` y `temporal_rmsd` ejecutados y validados (particiones sin solapamiento, cobertura recomputada, ridge de 48 descriptores reproduce la referencia). A 5 y 10 ns el ridge mejora a persistencia y a los baselines simples; a 20 ns la ventaja sobre el mejor baseline simple no se distingue de cero. Cobertura de los intervalos al 90/95 % global cercana a la nominal, pero con infracobertura en los sistemas más variables (variante absoluta). Pendientes Rg, core, SASA, hélice y lámina. Detalle en [`FASE2_PRIMERA_RONDA_20261118.md`](FASE2_PRIMERA_RONDA_20261118.md).
