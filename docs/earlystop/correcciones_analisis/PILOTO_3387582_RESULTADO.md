# CILS v3 — resultado del piloto 3387582 (fragmento 000) y proyección de recursos

Registro del **16/11/2026**. Fuente: capturas de terminal en CALÉNDULA pegadas por Miti (`evidencia_3387582/`); no se ha accedido a SCAYLE de forma independiente. El piloto se lanzó el 13/11/2026 (ver [`PILOTO_3387582_ESTADO_INICIAL.md`](PILOTO_3387582_ESTADO_INICIAL.md)). **No se han lanzado los 75 fragmentos restantes ni se han abierto validation/test.**

## Resultado

**El piloto terminó correctamente.** `sacct`: trabajo y paso batch `COMPLETED 0:0`; `.err` vacío; `squeue` ya no lo lista.

| Medida | Valor | Origen |
|---|---|---|
| Duración del trabajo | **01:36:49** (5809 s) de un límite de 06:00:00; inicio 13/11 13:54:53, fin 15:31:42 (+01:00), nodo `cn7180` | `sacct` |
| CPU total | 01:35:42 (5742 s) con 1 CPU asignada | `sacct` |
| Memoria máxima (paso batch) | **134 795 KiB ≈ 131,6 MiB** de 8 GiB solicitados | `sacct` |
| Memoria máxima por fragmento | 117 424 KiB (include) y 120 340 KiB (exclude) | `complete.json` |
| Tiempo por fragmento y política | 2902,4 s (**48,4 min**) para 10 sistemas, es decir ≈ **290 s por sistema y política** | `complete.json` |
| Filas por política | 180 (10 sistemas) | `SHARD_PASS`, `complete.json` |

Las dos políticas se ejecutaron en serie (`include` y luego `exclude`): 2902,4 + 2902,3 s ≈ 5804,7 s, frente a 5809 s del trabajo (≈ 4 s de arranque). CPU/wall ≈ 0,99: la tarea ocupa un único núcleo todo el tiempo, por lo que más núcleos por tarea no la aceleran con este código (`OMP_NUM_THREADS=1`).

## Verificaciones de salida realizadas

| Comprobación | Resultado |
|---|---|
| Estado del trabajo y del paso batch | `COMPLETED 0:0` |
| `SHARD_PASS` de ambas políticas | sí: `systems=10 rows=180` |
| Filas esperadas por política | 180 = 10 sistemas × 18 (seis observables × tres réplicas, según el diseño); coincide |
| `stderr` | vacío |
| `shard_sha256` y `plan_sha256` | **idénticos** en `include` y `exclude`: ambas políticas trabajaron sobre el mismo fragmento y el mismo plan |
| `builder_sha256` `facb86a5…b12e` | = SHA-256 de `calculos/build_targets.py` de [`fuente_v3/`](fuente_v3/) y del manifiesto v3 (comprobado) |
| `config_sha256` `90352d9c…fb4e20` | = SHA-256 de `targets_config.json` del manifiesto v3 (comprobado) |
| Motor en SCAYLE `cils_v3_shards.py` | SHA-256 `f9e8e191…c188` = el del archivo de esta rama (commit `178e425`, 26 065 bytes) (comprobado) |
| Hashes de `targets.csv`, `targets.summary.json` y `build.log` | registrados en `complete.json`; **no recalculados** |

## Lo que NO está verificado

- **Contenido de la salida.** Falta `n_problems=0` y el esquema por observable y réplica: no se vio el contenido de `targets.summary.json` ni de `build.log`, y el script verificador no se ejecutó (se retiró de la captura). Los 180 = 10 × 18 son coherentes con el diseño, no una comprobación de las filas.
- **Hashes de archivos.** No se recalculó `sha256sum` de `targets.csv`, `targets.summary.json` y `build.log` frente a `complete.json`.
- **Representatividad.** Son 10 de 757 sistemas; el coste real depende del tamaño de cada sistema y puede ser mayor en los sistemas grandes.
- **Alcance de la medida.** El piloto mide **solo la construcción de objetivos** (`build_targets.py`). **No** incluye el análisis temporal de v3 (prefijos sin información posterior, calibración separada, baselines del tramo final y ablaciones), que se ejecuta después sobre los objetivos consolidados. Sus recursos están sin medir.
- **Orden de magnitud previo.** La ejecución antigua `3344624` llegó a `[100/757]` en 12 h (≈ 432 s por sistema); el piloto da ≈ 290 s por sistema y política. Son cifras del mismo orden, pero no equivalentes (la carga y la política de esa ejecución no están documentadas aquí).

## Proyección de recursos (aritmética de `proyeccion_recursos_cils_v3.py`)

Supuestos: coste lineal por sistema e igual al del fragmento 000; 757 sistemas en 76 fragmentos (75 × 10 + 7); tareas independientes (un fragmento × una política), cada una de 1 CPU.

```text
Tarea (10 sistemas, 1 política): 2902.4 s = 48.4 min; 290.2 s por sistema y política
MaxRSS hijo: 117.5 MiB; CPU/wall = 0.989
Resto (150 tareas: 148 de 10 sistemas + 2 de 7): 433613 s = 120.4 h de CPU (1 núcleo)
Total 757 sistemas × 2 políticas: 122.1 h de CPU

| Concurrencia (tareas a la vez) | Oleadas | Tiempo de reloj aprox. |
|---|---|---|
| 5 | 30 | 24.2 h |
| 10 | 15 | 12.1 h |
| 15 | 10 | 8.1 h |
| 25 | 6 | 4.8 h |
| 50 | 3 | 2.4 h |
| 75 | 2 | 1.6 h |
```

Pendientes tras el piloto: 150 tareas (fragmentos 001–075 × 2 políticas).

## Propuesta para el resto (no ejecutada; decisión de Fran)

- **Una tarea por fragmento y política** (`run --shard N --policy P`), no 6 h seguidas con las dos políticas como en el piloto: así un fallo o un `TIMEOUT` solo afecta a una tarea, y la reanudación ya reutiliza los fragmentos con hash verificado.
- **Recursos por tarea:** 1 CPU, **2 GiB** de memoria (se observaron ≈ 0,13 GiB; el margen cubre sistemas mayores) y **tiempo límite de 2 h** (se observaron 48 min; margen ≈ 2,5×). Conviene comprobar cada tarea al terminar con `sacct` y reajustar si alguna supera el 70 % del tiempo.
- **Concurrencia:** las tareas son independientes, así que la limitación es de cuota, no técnica. Con 10 a la vez, el resto tarda ≈ 12 h de reloj; con 25, ≈ 5 h. **Antes de elegir hay que confirmar los límites de la cuenta `uva_dma_2` en la partición `sapphire`** (trabajos simultáneos y `RLIMIT_NPROC`, que ya dio un fallo el 13/11). Sugerencia: empezar con 10 a la vez, comprobar que las primeras tareas coinciden con el tiempo del piloto y subir si la cuota lo permite.
- **Antes de lanzar:** completar las verificaciones pendientes del fragmento 000 (contenido de `targets.summary.json` y `build.log`, `sha256sum` de los tres archivos) y medir en un fragmento los recursos del análisis temporal.
- Se conservan los **757 sistemas, los criterios y los folds**; no se abren validation ni test.

## Evidencia

[`evidencia_3387582/`](evidencia_3387582/): `sacct_3387582.txt`, `pilot_3387582.out.txt`, `complete_include_000.json`, `complete_exclude_000.json`, `proyeccion_recursos.txt`. Código: [`cils_v3_shards.py`](cils_v3_shards.py), [`cils_v3_pilot.sbatch`](cils_v3_pilot.sbatch), [`proyeccion_recursos_cils_v3.py`](proyeccion_recursos_cils_v3.py). Pruebas sintéticas: [`PRUEBAS_SINTETICAS.md`](PRUEBAS_SINTETICAS.md) (11/11 en local y en SCAYLE).
