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
