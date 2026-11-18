# CILS v3 — fase 2, primera ronda: RMSD (intervalos, prefijos 5/10/20 ns y baselines)

Registro del **18/11/2026**. Fuente: salidas de CALÉNDULA copiadas por Miti ([`evidencia_fase2_20261118/`](evidencia_fase2_20261118/)); no se accedió a SCAYLE de forma independiente. **Solo RMSD**; Rg, SASA, hélice, lámina y core RMSD siguen pendientes (ver el final). Entrenamiento únicamente (validación y test no se abren); los 757 sistemas, los criterios y los folds originales no cambian. Son análisis **exploratorios y descriptivos**: los intervalos de los contrastes condicionan los ajustes fuera de muestra fijos y no cuantifican la incertidumbre del ajuste ni de la calibración (`provenance.json`); no se hace ningún contraste confirmatorio.

## Qué se ejecutó

Entrada: objetivos consolidados del [bloque A](BLOQUE_A_RESULTADO.md), `targets_include_constants.csv` (SHA-256 `b45242be…0294`, comprobado antes de ejecutar). Script: [`cils_v3_fase2.sbatch`](cils_v3_fase2.sbatch) (SHA-256 `2897a179…6bd1`). Cuenta `uva_dma_2`, `sapphire`/`normal`, 1 CPU. Entorno: Python 3.11.5, NumPy 1.26.4.

| Etapa | Job | Orden (v3, `fuente_v3/calculos`) | Tiempo | Memoria |
|---|---|---|---|---|
| `intervals_rmsd` | 3412484 | `reanalyse.py --only rmsd --mode intervals` | 19 s | 3 MB (no fiable) |
| `prefix_rmsd` | 3412485 | `reanalyse.py --only rmsd --mode prefix` | 21 s | 3 MB (no fiable) |
| `temporal_rmsd` | 3412486 | `temporal_controls.py --only rmsd --prefix-ns 5 10 20` | 41 s | **181 MB** |

Los tres `COMPLETED 0:0`, con `.err` vacío. Las dos memorias de 3 MB son un artefacto de que Slurm muestrea cada ~30 s y estas tareas duraron menos; solo la de `temporal_rmsd` es fiable. Esta fase cuesta segundos, a diferencia de los 48 min por fragmento de la construcción de objetivos.

## Comprobaciones realizadas

| Comprobación | Resultado |
|---|---|
| Ridge de 48 descriptores frente a la referencia archivada (`temporal_controls.py`) | `FULL_RMSD_PREDICTION_CHECK=PASS`: diferencia máxima de etiquetas 0,0 y de predicciones 6,4e-14 Å, 757 sistemas |
| Máscara y etiquetas de RMSD frente a `train_future_rmsd_rows.csv` | máscara idéntica, diferencia 0,0 Å, 1 243 réplicas válidas |
| Entradas | `targets_sha256`, `dataset_sha256` `a4e2083c…1db6`, `folds_sha256` `fea8e852…e00b` en los tres `provenance.json`; ámbito `training cohort only` |
| Particiones de calibración (`partition_manifest.csv`, [`validar_fase2.py`](validar_fase2.py), `VALIDACION_FASE2=PASS`) | 5 folds; en cada uno, grupos de `training`, `calibration` y `evaluation` **sin solapamiento**; cada grupo se evalúa **una sola vez**; todo sistema cae en el rol de evaluación de su fold |
| Cobertura y anchura | recomputadas desde `intervals.csv` (757 sistemas × 3 variantes × 2 niveles = 4 542 filas) y coinciden con `coverage.csv` (diferencia 7e-14); `covered` coherente con los límites |
| Canales del prefijo (`prefix_rmsd`) | `[0, 1, 5, 6, 7]`: **sin canales de core** en 5, 10 y 20 ns |

## Resultados de RMSD (MAE en Å; agregado por sistema, 757 sistemas)

### Prefijos 5/10/20 ns, comparación homogénea sin canales core (`prefix_rmsd`)

| Prefijo | Persistencia | Media del prefijo | Ridge | Ridge − persistencia [IC95 % por grupos] | Ridge − media del prefijo |
|---|---|---|---|---|---|
| 5 ns | 1,062 | 1,344 | **0,792** | −0,269 [−0,342, −0,199] | −0,552 [−0,642, −0,464] |
| 10 ns | 0,825 | 1,085 | **0,708** | −0,117 [−0,164, −0,070] | −0,378 [−0,443, −0,312] |
| 20 ns | 0,673 | 0,822 | **0,591** | −0,082 [−0,120, −0,044] | −0,231 [−0,284, −0,178] |

Con canales completos a 20 ns (incluidos los de core definidos con esa información) el ridge da 0,610 Å, idéntico al resultado archivado; sin ellos, 0,591 Å. Los intervalos incluyen los 20 ns porque el prefijo ya los tiene.

### Baselines del tramo final frente al ridge (`temporal_rmsd`, contraste primario `ridge_all_full − simple_selected`)

| Prefijo | ridge_all_full | Mejor baseline simple elegido en folds internos | Diferencia [IC95 %] |
|---|---|---|---|
| 5 ns | 0,792 | 1,052 | −0,260 [−0,330, −0,191] |
| 10 ns | 0,708 | 0,819 | −0,111 [−0,157, −0,067] |
| 20 ns | 0,610 | 0,628 | **−0,018 [−0,059, +0,022]** |

A 5 y 10 ns el ridge mejora a una media de la parte final del prefijo (ventana o semivida elegida dentro de entrenamiento). **A 20 ns la ventaja deja de distinguirse** de ese baseline simple (el intervalo incluye 0). Este resultado negativo es coherente con la advertencia del artículo de que los controles simples son muy competitivos. Hay 24, 25 y 26 procedimientos por horizonte; el mejor de ellos a 20 ns es `ridge_target_full` (0,582), sin conclusiones confirmatorias por comparaciones múltiples.

### Calibración independiente, 20 ns (`intervals_rmsd`; cobertura nominal 90 % y 95 %)

| Variante | Cobertura global 90 % | 95 % | Anchura media (Å) 90 % / 95 % |
|---|---|---|---|
| absoluta | 0,898 | 0,952 | 3,07 / 4,63 |
| máximo por grupo | 0,917 | 0,962 | 3,25 / 5,15 |
| máximo por grupo escalado | 0,897 | 0,948 | 3,00 / 4,08 |

Por estrato de variabilidad del prefijo (1 = menor, 4 = mayor), cobertura nominal 90 %:

| Variante | Estrato 1 (n=194) | 2 (186) | 3 (184) | 4 (193) |
|---|---|---|---|---|
| absoluta | 0,995 | 0,968 | 0,913 | **0,720** |
| escalada | 0,845 | 0,887 | 0,924 | 0,933 |

La variante absoluta cubre el nivel nominal en conjunto pero **infracubre** claramente los sistemas más variables (0,72 al 90 %); la escalada reparte mejor la cobertura (rango 0,85–0,93), con anchuras de 1,0 a 6,3 Å según el estrato frente a una anchura casi constante en la absoluta. Son coberturas marginales con sistemas dependientes, no cobertura condicional por proteína; los intervalos al 95 % siguen el mismo patrón (tabla completa en [`coverage.csv`](evidencia_fase2_20261118/resultados/intervals_rmsd/coverage.csv)).

## Lo que NO está verificado

- **Solo RMSD.** Rg, SASA, hélice, lámina y core RMSD no se han analizado en esta ronda; la calibración de Rg (prioridad de Fran) está pendiente.
- **Definición del core con los primeros 20 ns:** no se ha confirmado que los canales de core de `dataset.npz` se hayan definido solo con el prefijo de 20 ns; en `intervals_rmsd` y `temporal_rmsd` a 20 ns se usan esos canales, mientras que a 5 y 10 ns no se usa ninguno.
- **Estratos por variabilidad:** la definición de los cuatro estratos sale del código (`reanalyse.py`) y no se ha revisado aquí línea a línea; las cifras son descriptivas.
- **Incertidumbre:** los intervalos de los contrastes no remuestrean el ajuste ni la calibración; hay muchas comparaciones y no se declara ninguna confirmatoria.
- **No acceso a validation/test:** garantía indirecta (entradas congeladas por hash, `selection_scope = training cohort only` y particiones comprobadas).

## Siguiente

Etapas aún sin ejecutar (el mismo script, con `--mem=2G --time=01:00:00` en la línea de `sbatch`, que sobrescribe los `#SBATCH`): `intervals_rg` (prioridad), `intervals_core_rmsd_q30`, `temporal_rg`, `temporal_sasa`, `temporal_helix`, `temporal_strand`, `temporal_core_rmsd_q30`, `temporal_helix_exclude` y `temporal_strand_exclude`.
