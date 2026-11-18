# CILS v3 — bloque A: objetivos reconstruidos de los 757 sistemas (resultado)

Registro del **18/11/2026**. Fuente: salidas de terminal en CALÉNDULA pegadas y copiadas por Miti ([`evidencia_bloque1_3406133/`](evidencia_bloque1_3406133/)); no se accedió a SCAYLE de forma independiente. Solo se reconstruyeron **objetivos**: **no** hay todavía predicciones, métricas, contrastes ni intervalos corregidos (bloques B y C). Entrenamiento únicamente; no se abrieron validation ni test; criterios, grupos y folds sin cambios.

## Qué se ejecutó

| Elemento | Valor |
|---|---|
| Motor | `cils_v3_shards.py` (SHA-256 `f9e8e191…c188`, commit `178e425`), 76 fragmentos de ≤ 10 sistemas |
| Piloto previo | job `3387582`: fragmento 000, ver [`PILOTO_3387582_RESULTADO.md`](PILOTO_3387582_RESULTADO.md) |
| Array | job `3406133`, 150 tareas (fragmentos 001–075 × `include`/`exclude`), 1 CPU, 2 GiB, 2 h; lanzado con `sbatch --parsable --array=1-150%3 cils_v3_array.sbatch` |
| Consolidación | job `3406134`, `sbatch --parsable --dependency=afterok:3406133 cils_v3_consolidar.sbatch` |
| Scripts | [`cils_v3_array.sbatch`](cils_v3_array.sbatch) `0315c4bb…25d5`; [`cils_v3_consolidar.sbatch`](cils_v3_consolidar.sbatch) `24b6831f…0596`; [`resumen_objetivos_consolidados.py`](resumen_objetivos_consolidados.py) `5c904f28…c102` |
| Entradas | inventario congelado `inventory_miti.csv` (SHA-256 `19365fc2…ab9e`), 757 sistemas de entrenamiento; `builder_sha256` `facb86a5…b12e` y `config_sha256` `90352d9c…fb4e20`, iguales a los del paquete v3 |
| Cuenta, partición | `uva_dma_2`, `sapphire` / `normal` |

## Resultado de la ejecución

- **Array:** 150 de 150 `COMPLETED`; ningún `.err` con contenido; ningún resto `.partial`.
- **Tiempo por tarea:** máximo **3331 s (55,5 min)**; las tres mayores, 3248, 3305 y 3331 s (el piloto dio 48,4 min por tarea de 10 sistemas).
- **Memoria máxima (paso batch):** 140 718 KiB ≈ **137,4 MiB** (de 2 GiB solicitados).
- **Consolidación:** job `3406134` `COMPLETED 0:0` en 14 s; `VERIFIED_CHECKPOINTS=152/152`, `NOT_VERIFIED=[]`, **`MERGE_PASS systems=757 rows_per_policy=13626`**; `include` y `exclude` difieren solo en ventanas constantes. Creado el 17/11/2026 22:39 UTC.

## Comprobaciones de salida

| Comprobación | Resultado |
|---|---|
| Etiquetas y máscara de RMSD frente a la referencia archivada ([`rmsd_reference_check.json`](evidencia_bloque1_3406133/rmsd_reference_check.json), `verify_rebuilt_rmsd.py`, job interactivo `3412217`) | **`RMSD_REFERENCE_CHECK=PASS`**: máscara de réplicas válidas **idéntica**; diferencia máxima de etiquetas **0,0 Å**; 1 243 réplicas válidas |
| Resumen de recuentos y exclusiones (`resumen_objetivos_consolidados.py`, job `3412223`) | **`RESUMEN_OBJETIVOS=PASS`**, 0 problemas: los recuentos de `*.summary.json` coinciden con las filas y toda réplica válida cumple el criterio (ventana futura ≥ 20 ns, neff ≥ 10, valor finito) |
| `n_problems` de ambos resúmenes | 0 |

## Recuentos (réplicas posibles: 757 sistemas × 3 = 2 271 por observable)

Política `include` (la de referencia):

| Observable | Réplicas válidas | Sistemas con ≥ 1 válida | Sistemas con 1 / 2 / 3 válidas | Inválidas: ventana constante | duración < 20 ns | neff < 10 | ambas |
|---|---|---|---|---|---|---|---|
| RMSD | 1 243 | 757 | 363 / 302 / 92 | 0 | 1 019 | 8 | 1 |
| Core RMSD q30 | **947** | **595** | 301 / 236 / 58 | 0 | 1 322 | 1 | 1 |
| Rg | 992 | 610 | 304 / 230 / 76 | 0 | 1 274 | 3 | 2 |
| SASA | 1 384 | 701 | 202 / 315 / 184 | 0 | 884 | 2 | 1 |
| Hélice | 1 549 | 728 | 168 / 299 / 261 | 4 | 706 | 0 | 12 |
| Lámina | 1 547 | 728 | 169 / 299 / 260 | 102 | 609 | 2 | 11 |

Política `exclude`: idéntica salvo hélice (1 542 válidas, 727 sistemas, 11 ventanas constantes) y lámina (**1 299** válidas, 646 sistemas, **350** ventanas constantes). La tabla completa está en [`recuentos_objetivos_consolidados.csv`](evidencia_bloque1_3406133/resumen_objetivos/recuentos_objetivos_consolidados.csv). Los motivos se clasifican con el criterio del constructor; no hay otras causas (`inval_otro = 0`, `sin ventana futura = 0`).

## Comparación con la auditoría archivada del 10/11 (`cohort_flow.csv`)

- **RMSD, Rg, SASA, hélice y lámina: recuentos idénticos** (réplicas válidas, sistemas válidos y reparto por 1/2/3 réplicas). Las 11 y 350 ventanas constantes de hélice y lámina con `exclude` coinciden con el registro público previo.
- **Core RMSD q30: cambia.** Antes: 712 sistemas y 1 136 réplicas válidas (363 / 274 / 75), con el tiempo de equilibrado del RMSD global; ahora: **595 sistemas y 947 réplicas** (301 / 236 / 58), con equilibrado propio sobre su serie. Es una reducción de unas 189 réplicas (≈ 17 %). **No se deben mezclar ambas cifras.** Los resultados del core (bloque de análisis) siguen pendientes.

## Lo que NO está verificado

- **Corrección científica del core.** Los recuentos y las comprobaciones son de estructura y consistencia; el RMSD se contrastó con la referencia, pero los otros cinco observables y el core no tienen una referencia independiente de estos objetivos reconstruidos (salvo la coincidencia de recuentos con la auditoría para los que no son core).
- **Definición del core con los primeros 20 ns.** El código exige `replica_prefix_rmsd_A` y equilibrado propio (`detect`), pero no se ha comprobado que las series de núcleo archivadas se hayan definido solo con el prefijo de 20 ns.
- **Inventario de los 924 sistemas** de entrenamiento y su manifiesto de grupos: no consta; el análisis cubre los 757 evaluables.
- **No acceso a validation/test:** garantía indirecta (plan y entradas congeladas por hash, solo `train`); no hay registro de accesos.
- **No hay resultados científicos.** Falta el análisis posterior sobre estos objetivos: calibración independiente (RMSD y Rg), prefijos 5/10/20 ns sin core, baselines del tramo final, ablaciones y core. Sus recursos aún no se han medido.

## Archivos de objetivos

Los CSV grandes quedan en SCAYLE (`…/paper2_cils_v3_relaunch_v1/merged/`) y se identifican por SHA-256: `targets_include_constants.csv` `b45242be…0294`, `targets_exclude_constants.csv` `880ab7d3…7942`, `MERGE_PASS.json` `e82609e4…30fd`. Evidencia pequeña en [`evidencia_bloque1_3406133/`](evidencia_bloque1_3406133/).
