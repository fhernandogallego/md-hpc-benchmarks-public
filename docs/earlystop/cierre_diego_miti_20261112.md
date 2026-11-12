# EARLYSTOP — cierre documental Diego y Miti (12-11-2026)

**Estado: auditoría documental REALIZADA, con limitaciones explícitas.** Se han inspeccionado archivos ya existentes en SCAYLE y el repositorio público; **no** se han realizado nuevas simulaciones, entrenamientos, selección de resultados favorables ni consultas de métricas/etiquetas de test. El contenido de este documento sustituye las cifras preliminares históricas cuando difieren.

## 1. Código y configuración: dos protocolos distintos

### ATLAS942 (cohorte completa; análisis etiquetado de 757 sistemas)

- Ejecución: `/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/results/paper2_automation/auto_20261030_091302`.
- Controlador Slurm `3262490`: `COMPLETED`, `0:0`; cuenta `uva_dma_2`; partición `sapphire`.
- Código congelado: `pipeline_code/src/pipeline/extract_atlas_basic.py`, SHA256 `f739b02858a6517820f0747fd39706296d75dc39706fb7bad82af7d4b0bd2620`. Coincide con `pipeline_manifest.json` (`SHA256_OK`).
- RMSD: selección `protein and backbone`; referencia fotograma 0 de **cada réplica**; superposición por mínimos cuadrados; `weights=None`, es decir, uniforme.
- Rg: `universe.select_atoms("protein").radius_of_gyration()`; átomos de proteína completa; ponderación implícita por masas atómicas.
- Metadatos de un ejemplo de réplica original `1a62_A_R1_basic_observables.json`, SHA256 `421be64c799b0260aa10ebf0bd0fd5840800a0f3c634204625b3449659e10ca4`, confirman `canonical_rmsd` y `canonical_rg`.
- Rg de la trayectoria no se calcula con el conjunto de Cα, a diferencia del piloto.

### Piloto de 500 sistemas: código congelado identificado

- Ejecución original: `/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/results/pid_atlas_poc/full_v2_3262025`.
- Fuente `prueba_atlas_v2.py`, SHA256 `0b3304050c7e7243f96cc8cae2f84f2f88b5e2905d21d753e0f1417fff909343`; copia idéntica en `results/pid_atlas_poc/audit_v2_500_3268243/scripts/prueba_atlas_v2.py`.
- Lanzador `launcher.sbatch`, SHA256 `153da31c9a2832b1c728202b4c034375e0bf1d68f827a55ac68f0610f4b4e794`; copia en `audit_v2_500_3268243/scripts/original_launcher.sbatch` con el mismo hash.
- RMSD: `name CA`; centrado y alineamiento Kabsch al primer fotograma; ponderación uniforme.
- Rg: coordenadas de Cα exclusivamente; sin ponderación por masa.
- Objetivo operativo del piloto: media de la **segunda mitad** de cada trayectoria (fotogramas 500:1001, ~50–100 ns), que no debe interpretarse como equilibrio termodinámico demostrado.
- Auditoría del piloto exige 1500 réplicas por descriptor (500 sistemas × 3), con 1001 fotogramas por réplica y 100 ps entre fotogramas. La auditoría posterior `audit_v2_500_3268243/cohort_assignment.csv` lista **524 train + 9 validation + 9 test**: es un manifiesto de **cohorte diferente**, no un recuento alternativo del experimento concreto de 500. No combinar estas cifras.

**Conclusión:** RMSD, Rg y protocolo de etiquetado del piloto **no son equivalentes** a los de ATLAS942.

## 2. Recuentos congelados de ATLAS942 y motivos de exclusión

Fuentes efectivas:
- `cohort/system_assignment.csv`, SHA256 `bb12305d7ed029ec7c072e1789b357c77420ef35aeca39ed69469217a7109d37`.
- `cohort/source_inventory.csv`, SHA256 `6379f09b03b313e2d4b5247c50cfe382f43eb27a8c8a024be89fb84a3737f428`.
- `cohort/train_future_rmsd_rows.csv`, SHA256 `a403c51c3da35c5a9899928b7b7fe90265124abfb163dd214ad10971ceb9d81c`.
- Se leyeron 924/924 `master_targets_long.csv` de **train**: sin filas no verificables para los cinco observables del master.

Cohorte: 942 sistemas / 835 grupos. Desarrollo asignado `train`: 924 sistemas / 817 grupos / 2772 posiciones de réplica. Etiquetados `future_valid` RMSD: 757 sistemas / 689 grupos / 1243 réplicas. Hay 167 sistemas train sin al menos una réplica con etiqueta RMSD futuro válida.

| Universo | Observable del master | Sistemas entrada | Réplicas posibles | Sistemas retenidos | Réplicas retenidas | Sistemas sin réplica válida | Réplicas inválidas |
|---|---|---:|---:|---:|---:|---:|---:|
| train | RMSD | 924 | 2772 | 760 | 1253 | 164 | 1519 |
| train | Rg | 924 | 2772 | 725 | 1154 | 199 | 1618 |
| train | SASA | 924 | 2772 | 860 | 1675 | 64 | 1097 |
| train | Hélice | 924 | 2772 | 878 | 1864 | 46 | 908 |
| train | Lámina | 924 | 2772 | 878 | 1864 | 46 | 908 |
| future-RMSD-labeled | RMSD | 757 | 2271 | 757 | 1250 | 0 | 1021 |
| future-RMSD-labeled | Rg | 757 | 2271 | 610 | 995 | 147 | 1276 |
| future-RMSD-labeled | SASA | 757 | 2271 | 701 | 1386 | 56 | 885 |
| future-RMSD-labeled | Hélice | 757 | 2271 | 728 | 1547 | 29 | 724 |
| future-RMSD-labeled | Lámina | 757 | 2271 | 728 | 1547 | 29 | 724 |

La clasificación por replica es `INVALID_SHORT_WINDOW` (SW), `INVALID_LOW_NEFF` (NEFF), y `INVALID_SHORT_WINDOW_AND_LOW_NEFF` (ambos). Categorías **excluyentes como etiquetas**:

| Universo | Observable | SW | NEFF | Ambos | Total inválidas |
|---|---|---:|---:|---:|---:|
| train | RMSD | 1516 | 1 | 2 | 1519 |
| train | Rg | 1616 | 0 | 2 | 1618 |
| train | SASA | 1096 | 0 | 1 | 1097 |
| train | Hélice | 873 | 4 | 31 | 908 |
| train | Lámina | 873 | 4 | 31 | 908 |
| future-RMSD-labeled | RMSD | 1019 | 1 | 1 | 1021 |
| future-RMSD-labeled | Rg | 1274 | 0 | 2 | 1276 |
| future-RMSD-labeled | SASA | 884 | 0 | 1 | 885 |
| future-RMSD-labeled | Hélice | 699 | 2 | 23 | 724 |
| future-RMSD-labeled | Lámina | 699 | 2 | 23 | 724 |

**RMSD original frente a futuro:** 1519 réplicas `original_valid=false/future_valid=false`, 10 `true/false` y 1243 `true/true`, total 2772. Así se explica exactamente `1253 - 1243 = 10`. Los motivos tabulados arriba son del master, **no** una taxonomía demostrada para `future_valid=false`.

### Core RMSD q30 (fuente distinta al master)

- El master tiene cinco observables: `rmsd, rg, sasa, helix, strand`, **no core RMSD**.
- `core_rmsd/core_rmsd_summary.csv` y las correspondientes series q30 para `replica_prefix` existen en **923/924 sistemas train, 2769/2772 réplicas**.
- Dentro del subconjunto de 757 existen en **756/757 sistemas, 2268/2271 réplicas**.
- Un sistema carece de resumen q30 en ambos universos. **ID y causa no recuperados del log resumido**, aunque se pueden consultar sin evaluar nada en `core_q30_disponibilidad.csv` de la auditoría.
- Metadatos de q30: selección de residuos Cα; RMSD de los átomos backbone de los residuos seleccionados; alineamiento en el mismo core; referencia frame 0; `weights=None`. La variante `replica_prefix` está identificada como elegible para entrada de trayectoria individual. Las variantes `*_full_oracle` son **solo diagnóstico**.
- **No existe `target_valid` en el resumen de series q30 examinado**. Por eso 2268 series disponibles **no equivalen** a 2268 etiquetas válidas. Recuentos históricos MITI de core `712 sistemas/1136 réplicas` no se han reproducido como validez en la fuente actual. No se asigna causa de exclusión q30 sin la tabla correspondiente.

## 3. Manifiestos de partición, grupos y commits

`auto_20261030_091302/cohort/system_assignment.csv` (SHA256 anterior) se copió idéntico en `WORK/releases/paper2_atlas942_20261103_v1/cohort/system_assignment.csv`.

| Partición **registrada** | Sistemas | Grupos |
|---|---:|---:|
| `train` | 924 | 817 |
| `validation` | 9 | 9 |
| `test` | 9 | 9 |
| **Total** | **942** | **835** |

Grupos de secuencia compartidos entre particiones: **0** (comprobación sobre asignaciones). Las réplicas permanecen dentro de su sistema en esta partición. Si los tres archivos/replicas de todos los sistemas están disponibles, las posiciones teóricas por partición son 2772/27/27; la auditoría **no verificó** 27 archivos de validation/test y no los abrió.

**No se encontró** un manifiesto que especifique `development/calibration/test`: `train/validation/test` no autoriza rebautizar `validation` como calibración. Se declara **NO ACREDITADA** la partición independiente de calibración. No se abrieron predicciones, métricas ni etiquetas nuevas de validation/test.

- Commit del informe ATLAS942 publicado: `aeac56891e06bb002d50ebdd3a1068c84fd2c600` (rama `paper2-atlas942-results-20261103`). **Es commit de publicación, no commit de generación demostrado para el CSV**.
- Commit de origen/generación del manifiesto: **NO ACREDITADO**. En SCAYLE `git` no está disponible en PATH (`GIT_BINARIO: None`), aunque los dos checkouts conservan directorio `.git`. Los manifiestos SHA-256 de `tool` y `pipeline` sí están disponibles.

## 4. Candidato experimental alfa-sinucleína

Ahmed et al. (2021), DOI https://doi.org/10.3389/fmolb.2021.654333, perfil experimental SEC-SAXS con incertidumbres positivas accesibles: [ficha detallada](ficha_saxs_alfa_sinucleina_20261112.md). Compatibilidad con nuestras simulaciones: **NO CONFIRMADA**.

## 5. Entregables y procedencia

Ver [INDICE_ENTREGABLES.md](INDICE_ENTREGABLES.md) para **todas** las rutas y distinción entre documentos publicados y CSV íntegros solo en SCAYLE. Ficha condensada para OneDrive: [Respuesta_Diego_Miti_cierre.txt](Respuesta_Diego_Miti_cierre.txt). **OneDrive: NO SUBIDO**, pendiente de conexión autorizada o traslado manual.

### Alcance de la conclusión

**Cierre de la auditoría documental: realizado con salvedades** (no se inventan datos faltantes). Quedan explícitamente no acreditados: validez estadística/desglose por motivo de core q30; identidad/causa de la única ausencia q30 en el resumen proporcionado; manifiesto independiente `development/calibration/test`; commit generador de `system_assignment.csv`; compatibilidad SAXS con simulación propia. No se ha evaluado test ni superado HITO 0.
