# EARLYSTOP — verificación SCAYLE de ATLAS942 (12-11-2026)

**Alcance:** resultados de la auditoría *de solo lectura* ejecutada en SCAYLE el 12-11-2026. No se realizaron simulaciones, nuevos entrenamientos ni consultas de resultados de test. Fuente local de la auditoría:

`/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/maintenance/earlystop/auditoria_20261112_113942/INFORME.txt`

Esta evidencia **actualiza y prevalece** sobre las cifras históricas preliminares de `docs/earlystop/cierre_diego_miti_20261112.md` cuando difieren.

## 1. Código y observables del flujo ATLAS942

- Ejecución: `results/paper2_automation/auto_20261030_091302`. Controlador `3262490`, `sacct`: `COMPLETED`, salida `0:0`, cuenta `uva_dma_2`, partición `sapphire`.
- Código congelado: `pipeline_code/src/pipeline/extract_atlas_basic.py`. SHA-256: `f739b02858a6517820f0747fd39706296d75dc39706fb7bad82af7d4b0bd2620`. El hash concuerda con `pipeline_manifest.json` (`SHA256_OK`).
- RMSD: `MDAnalysis.analysis.rms.RMSD(universe, reference=universe, select="protein and backbone", ref_frame=0, weights=None)`; referencia primer frame de **cada réplica**, con superposición por mínimos cuadrados y pesos uniformes.
- Rg: `protein = universe.select_atoms("protein")`; `protein.radius_of_gyration()`, por masa de los átomos de `protein`.
- Verificación adicional desde metadatos originales: `1a62_A_R1_basic_observables.json` SHA-256 `421be64c799b0260aa10ebf0bd0fd5840800a0f3c634204625b3449659e10ca4`, con `canonical_rmsd` y `canonical_rg` concordantes.
- Presencia del código congelado y de metadatos confirma la definición documentada. No prueba que toda otra ejecución reutilice esa definición.
- **Piloto de 500:** el escaneo actual no ha expuesto código relevante dentro de `results/pid_atlas_poc/full_v2_3262025`. No confirmar aún selección, referencia y ponderaciones del piloto a partir de hipótesis previas. Se halló `results/pid_atlas_poc/audit_v2_500_3268243/cohort_assignment.csv` con **524 sistemas train**, 9 validation y 9 test: es un universo diferente de «500 exactos» y requiere identificar su procedencia, no reinterpretarlo por el nombre del archivo.

## 2. Validez por observable

Fuentes de partición y objetivos: `cohort/system_assignment.csv` SHA-256 `bb12305d7ed029ec7c072e1789b357c77420ef35aeca39ed69469217a7109d37`, `cohort/source_inventory.csv` SHA-256 `6379f09b03b313e2d4b5247c50cfe382f43eb27a8c8a024be89fb84a3737f428` y `cohort/train_future_rmsd_rows.csv` SHA-256 `a403c51c3da35c5a9899928b7b7fe90265124abfb163dd214ad10971ceb9d81c`.

Cobertura: 942 sistemas / 835 grupos; train = 924 sistemas / 817 grupos / 2772 posiciones de réplica; RMSD **futuro** válido = 757 sistemas y 1243 réplicas. Todos los `master_targets_long.csv` de train estaban presentes (924/924).

### Universo completo de train (924 sistemas, 2772 posiciones de réplica)

| Observable | Sistemas retenidos | Réplicas retenidas | Sistemas excluidos (tres réplicas inválidas) | Réplicas inválidas |
|---|---:|---:|---:|---:|
| RMSD (master) | 760 | 1253 | 164 | 1519 |
| Rg | 725 | 1154 | 199 | 1618 |
| SASA | 860 | 1675 | 64 | 1097 |
| Hélice | 878 | 1864 | 46 | 908 |
| Lámina | 878 | 1864 | 46 | 908 |

### Subuniverso de 757 sistemas etiquetados RMSD futuro (2271 posiciones de réplica)

| Observable | Sistemas retenidos | Réplicas retenidas | Sistemas excluidos | Réplicas inválidas |
|---|---:|---:|---:|---:|
| RMSD (master) | 757 | 1250 | 0 | 1021 |
| Rg | 610 | 995 | 147 | 1276 |
| SASA | 701 | 1386 | 56 | 885 |
| Hélice | 728 | 1547 | 29 | 724 |
| Lámina | 728 | 1547 | 29 | 724 |

**Importante:** `future_valid` es una etiqueta distinta de `target_valid` del master. No son intercambiables los 1243 RMSD futuro válidos y los 1250 RMSD del master en el universo de 757 sistemas. El core RMSD q30 **no está presente** entre los observables del master; su desglose requiere otra fuente.

### Motivos exactos de exclusión por réplica

Los estados `INVALID_SHORT_WINDOW` (SW), `INVALID_LOW_NEFF` (NEFF) y `INVALID_SHORT_WINDOW_AND_LOW_NEFF` (ambos) son **mutuamente excluyentes como etiquetas**. Para train: RMSD 1516/1/2 (SW/NEFF/ambos); Rg 1616/0/2; SASA 1096/0/1; hélice 873/4/31; lámina 873/4/31. Para los 757: RMSD 1019/1/1; Rg 1274/0/2; SASA 884/0/1; hélice 699/2/23; lámina 699/2/23.

No se detectaron posiciones de réplica no verificables en estas cinco tablas. Esta clasificación describe la validez de **objetivos del master**, no el diagnóstico de `future_valid`.

## 3. Partición y procedencia

El fichero `cohort/system_assignment.csv` tiene SHA-256 `bb12305d7ed029ec7c072e1789b357c77420ef35aeca39ed69469217a7109d37`. Particiones: **train 924 sistemas / 817 grupos; validation 9 / 9; test 9 / 9**. Grupos compartidos entre particiones: **cero**, según auditoría de asignaciones; la copia de release tiene el mismo SHA-256.

**No se ha verificado** un manifiesto distinto que asigne explícitamente `development/calibration/test` ni que autorice renombrar `validation` como `calibration`. El commit que **generó** la asignación en SCAYLE aparece como `NO_ACREDITADO`; el commit público conocido `aeac56891e06bb002d50ebdd3a1068c84fd2c600` es el de la publicación ATLAS942, no evidencia suficiente de creación inicial del CSV.

## Pendientes concretos

1. Reconstruir el origen del piloto exactamente **500** (código y `config` efectivos).
2. Localizar fuente independiente de validez core RMSD q30 y calcular el mismo desglose por observable.
3. Identificar y documentar, si existe, la partición explícita de calibración distinta de desarrollo/test y el commit/procedencia generadores. Si no existe, declarar `NO DISPONIBLE`.
