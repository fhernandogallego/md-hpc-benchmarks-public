# CILS v3 — fase 2, segunda ronda: Rg, core RMSD, SASA, hélice y lámina

Registro del **18/11/2026**, continuación de [`FASE2_PRIMERA_RONDA_20261118.md`](FASE2_PRIMERA_RONDA_20261118.md). Fuente: salidas de CALÉNDULA copiadas por Miti ([`evidencia_fase2_20261118/`](evidencia_fase2_20261118/)); sin acceso independiente a SCAYLE. Mismo script ([`cils_v3_fase2.sbatch`](cils_v3_fase2.sbatch), `2897a179…6bd1`, lanzado con `--mem=2G --time=01:00:00`), mismos objetivos consolidados del [bloque A](BLOQUE_A_RESULTADO.md) y misma advertencia: análisis **exploratorios y descriptivos** sobre entrenamiento, sin validation ni test, con intervalos condicionados a los ajustes fuera de muestra y sin conclusiones confirmatorias.

## Ejecución

Nueve trabajos, todos con `FASE2_PASS` y `.err` vacío: `intervals_rg` 3412549, `intervals_core_rmsd_q30` 3412550, `temporal_rg` 3412551, `temporal_sasa` 3412552, `temporal_helix` 3412553, `temporal_strand` 3412554, `temporal_core_rmsd_q30` 3412555, `temporal_helix_exclude` 3412556 y `temporal_strand_exclude` 3412557. Tiempo y memoria de estos nueve no se han registrado aquí (la medida de la primera ronda: 19–41 s y ≤ 181 MB por etapa).

## Comprobaciones

[`validar_fase2.py`](validar_fase2.py) sobre las 12 etapas: **`VALIDACION_FASE2=PASS`, 0 problemas.**
- Intervalos de Rg y core: 5 folds con grupos de `training`, `calibration` y `evaluation` sin solapamiento, cada grupo evaluado una sola vez, cobertura y anchura recomputadas desde `intervals.csv` (coinciden), niveles 90/95 % y estratos 0–4.
- Todas las etapas usan el `targets_sha256` esperado: `b45242be…` (`include`) y `880ab7d3…` (`exclude` para hélice y lámina).
- **Los recuentos de cada etapa coinciden con los del bloque A:** Rg 992 réplicas/610 sistemas, core 947/595, SASA 1 384/701, hélice 1 549/728 (`exclude`: 1 542/727), lámina 1 547/728 (`exclude`: 1 299/646).

## Resultados a 20 ns: MAE agregado por sistema

Contraste primario: `ridge_all_full − simple_selected` (baseline simple elegido dentro de entrenamiento: media de la parte final del prefijo o media exponencial). Negativo favorece al ridge. Intervalo del 95 % por grupos.

| Observable (n sistemas) | Persistencia | Media del prefijo | Ridge (todos los canales) | Baseline simple elegido | Ridge − persistencia [IC] | **Ridge − baseline simple [IC]** | Mejor de los 26 |
|---|---|---|---|---|---|---|---|
| Rg, Å (610) | 0,304 | 0,322 | 0,345 | 0,294 | +0,041 [+0,018, +0,066] | **+0,051 [+0,030, +0,073]** | EWMA 1 ns, 0,287 |
| Core RMSD q30, Å (595) | 0,147 | 0,150 | 0,136 | 0,131 | −0,011 [−0,023, +0,002] | +0,005 [−0,006, +0,017] | EWMA 2 ns, 0,129 |
| SASA, Å² (701) | 170,3 | 168,9 | 149,0 | 150,3 | −21,3 [−30,4, −12,2] | −1,3 [−8,0, +5,3] | ridge sin core, 146,9 |
| Hélice, fracción (728) | 0,01212 | 0,00766 | 0,00760 | 0,00743 | −0,00452 [−0,00529, −0,00375] | +0,00017 [−0,00013, +0,00048] | ridge con media/dispersión/último/pendiente, 0,00732 |
| Lámina, fracción (728) | 0,00766 | 0,00584 | 0,00603 | 0,00541 | −0,00163 [−0,00217, −0,00109] | **+0,00062 [+0,00037, +0,00087]** | EWMA 2 ns, 0,00534 |
| Hélice, `exclude` (727) | 0,01214 | 0,00767 | 0,00765 | 0,00745 | −0,00449 [−0,00525, −0,00374] | +0,00020 [−0,00010, +0,00052] | ridge, 0,00733 |
| Lámina, `exclude` (646) | 0,00865 | 0,00660 | 0,00688 | 0,00611 | −0,00177 [−0,00238, −0,00116] | **+0,00076 [+0,00047, +0,00105]** | EWMA 2 ns, 0,00604 |

Con RMSD (primera ronda) el mismo contraste a 20 ns da −0,018 [−0,059, +0,022].

**Lectura (descriptiva):**
- El ridge mejora a la persistencia en RMSD, SASA, hélice y lámina, y **empeora en Rg** (+0,041 Å).
- Frente a un baseline simple que promedia la parte final del prefijo, **el ridge no mejora en ningún observable a 20 ns**: no se distingue en RMSD, core, SASA y hélice, y **es peor en Rg y en lámina** (intervalos que excluyen 0). Buena parte de su ventaja sobre la persistencia se explica, por tanto, por promediar la parte reciente del prefijo, no por el modelo.
- La política `exclude` de ventanas constantes casi no cambia los resultados de hélice; en lámina (350 réplicas constantes excluidas) las conclusiones son las mismas.
- Hay 26 procedimientos por observable y muchas comparaciones: los «mejor de los 26» son orientativos, sin conclusiones confirmatorias.

## Calibración independiente a 20 ns (cobertura marginal; 90 % y 95 % nominales)

| Observable (n) | Variante | Global 90 % | Global 95 % | Anchura media 90 % / 95 % (Å) | Estrato 4 (mayor variabilidad) al 90 % |
|---|---|---|---|---|---|
| Rg (610) | absoluta | 0,920 | 0,962 | 1,89 / 3,27 | **0,744** |
| Rg (610) | máximo por grupo | 0,920 | 0,959 | 1,87 / 2,82 | 0,738 |
| Core RMSD q30 (595) | absoluta | 0,921 | 0,963 | 0,78 / 1,72 | **0,794** |
| Core RMSD q30 (595) | máximo por grupo | 0,921 | 0,963 | 0,78 / 1,72 | 0,794 |
| RMSD (757, primera ronda) | absoluta | 0,898 | 0,952 | 3,07 / 4,63 | 0,720 |
| RMSD (757, primera ronda) | escalada por grupo | 0,897 | 0,948 | 3,00 / 4,08 | 0,933 |

La cobertura global es igual o algo superior a la nominal en los tres observables, pero **infracubre los sistemas más variables** en Rg y core (estrato 4: 0,74 y 0,79 al 90 %; 0,88 al 95 %), igual que RMSD en su variante absoluta. La variante escalada solo está implementada para RMSD y es la única que reparte la cobertura entre estratos. Son coberturas marginales con sistemas dependientes, no cobertura condicional por proteína. Tablas completas por estrato: [`coverage.csv` de Rg](evidencia_fase2_20261118/resultados/intervals_rg/coverage.csv) y [del core](evidencia_fase2_20261118/resultados/intervals_core_rmsd_q30/coverage.csv).

## Lo que NO está verificado

- **Definición del core con los primeros 20 ns.** Los **objetivos** del core son los reconstruidos con equilibrado propio (bloque A: 947 réplicas); los **canales** de entrada del ridge siguen siendo los de `dataset.npz` y no se ha confirmado que se definieran solo con el prefijo de 20 ns. Los resultados del core deben leerse con esa reserva y **se mantienen fuera de la memoria** hasta confirmarlo.
- **Multiplicidad y selección retrospectiva.** Los objetivos válidos se seleccionan con información posterior al prefijo (criterio de la ventana futura); no hay contrastes confirmatorios.
- **Estratos de variabilidad y escalado.** Definidos por el código; no se han revisado aquí línea a línea.
- **No acceso a validation/test:** garantía indirecta (entradas congeladas por hash, `training cohort only`, particiones comprobadas).
- **Tiempo y memoria** de las nueve etapas nuevas: no registrados.
