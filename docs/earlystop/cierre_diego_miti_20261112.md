# EARLYSTOP: cierre documental Diego / Miti — 2026-11-12

> Estado: PARCIAL Y AUDITABLE. Este documento no certifica comprobaciones pendientes de acceso al SCAYLE. No se han lanzado simulaciones ni consultado predicciones/métricas nuevas de validation o test.

## 1. Procedencia y definición del observable

- Repositorio público de comparación: https://github.com/fhernandogallego/md-hpc-benchmarks-public
- Rama origen de la instantánea ATLAS942: `paper2-atlas942-results-20261103`; commit origen: `aeac56891e06bb002d50ebdd3a1068c84fd2c600`.
- Código público inspeccionado: `src/pipeline/extract_atlas_basic.py`.
- Ejecución de interés: controlador Slurm 3262490, `/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/results/paper2_automation/auto_20261030_091302`.
- Para esa implementación pública, RMSD: `MDAnalysis.analysis.rms.RMSD(universe, reference=universe, select="protein and backbone", ref_frame=0, weights=None)`, con superposición previa por mínimos cuadrados. Referencia: fotograma 0 de cada réplica. Ponderación: uniforme.
- Para esa implementación pública, Rg: `protein = universe.select_atoms("protein")`; `protein.radius_of_gyration()`, sobre toda la selección `protein` con masas atómicas según el comportamiento de MDAnalysis.
- **Pendiente esencial:** confrontar este código con `pipeline_code`, metadatos y `config.json` de la ejecución 3262490. La existencia del script en el repositorio no prueba que fuera el archivo efectivamente ejecutado.

### Piloto anterior (500 sistemas): conservar la distinción

- Carpeta documentada anteriormente: `/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks/results/pid_atlas_poc/full_v2_3262025`.
- Inventario histórico: 500 sistemas, 441 grupos, tres réplicas por sistema y 1001 fotogramas por réplica (0 a 100 ns, cada 100 ps).
- Documentación previa describía RMSD sobre Cα, alineamiento Kabsch al primer frame y ponderación uniforme, y Rg sobre Cα sin ponderación; **no se ha contrastado aún con el código/configuración congelados del piloto**.
- Por tanto, no se debe asumir equivalencia de Cα y backbone, ni de Rg Cα y Rg de proteína completa.

## 2. Recuentos

Resumen publicado ATLAS942: https://github.com/fhernandogallego/md-hpc-benchmarks-public/blob/paper2-atlas942-results-20261103/reports/paper2_atlas942_cv_20261103/summary.json

| Medida | Dato documental |
|---|---:|
| Cohorte ATLAS942 | 942 sistemas; 2.826 muestras/posibles réplicas |
| Grupos de la cohorte completa | 835 |
| Sistemas asignados a train | 924 |
| Sistemas etiquetados válidos RMSD | 757 |
| Grupos RMSD etiquetados | 689 |
| Réplicas RMSD con etiqueta válida | 1.243 |
| Sistemas train sin etiqueta RMSD | 167 (924 - 757; derivado) |
| Reparto CV por sistema | 152 / 152 / 151 / 151 / 151 |

Las cifras siguientes proceden de informes de la anterior entrega MITI y quedan **pendientes de comprobación en sus tablas de auditoría originales**; el universo es el inventario de 757 sistemas con hasta 2.271 réplicas.

| Observable | Sistemas retenidos (histórico MITI) | Réplicas retenidas (histórico MITI) | Motivos de exclusión |
|---|---:|---:|---|
| RMSD | 757 | 1243 | Pendiente de desglose por motivo/replica |
| Core RMSD q30 | 712 | 1136 | Pendiente de desglose por motivo/replica |
| Rg | 610 | 992 | Pendiente de desglose por motivo/replica |
| SASA | 701 | 1384 | Pendiente de desglose por motivo/replica |
| Fracción hélice | 728 | 1549 | Pendiente de desglose por motivo/replica |
| Fracción lámina | 728 | 1547 | Pendiente de desglose por motivo/replica |

No atribuir un motivo específico de exclusión sin la tabla fila-a-fila. Las réplicas no retenidas no equivalen automáticamente a réplicas fallidas: pueden excluirse por no satisfacer reglas de equilibrado, validez del objetivo o pertenencia a la cohorte.

## 3. Manifiesto desarrollo/calibración/test

- `src/paper2/make_hito0_split.py` define una partición a nivel de proteína en `train`, `validation`, `test` y escribe `protein_split.csv` y `split_metadata.json`, sin inspeccionar objetivos.
- Una partición `train/validation/test` **no equivale** automáticamente al manifiesto solicitado de `development/calibration/test`; no se debe renombrar `validation` como `calibration` sin justificación documental.
- **PENDIENTE** localizar el archivo exacto y SHA-256 congelados de desarrollo/calibración/test; número de sistemas, réplicas y grupos independientes de cada subconjunto; commit del código/configuración que lo generó.
- **Prohibido en este cierre:** abrir o calcular nuevas métricas de validation/test. Se permite inspeccionar exclusivamente archivos de asignación/identificadores, si son manifiestos congelados.

## 4. Candidato SAXS

Ficha detallada: [ficha_saxs_alfa_sinucleina_20261112.md](ficha_saxs_alfa_sinucleina_20261112.md).

Candidato preferente: Ahmed et al. (2021), DOI 10.3389/fmolb.2021.654333. Perfil SEC-SAXS de alfa-sinucleína humana monomérica con datos + incertidumbres públicamente accesibles (`SECSAXS8-chromixs_RAW.dat`, 1805 filas numéricas con tres columnas, sigma positiva en todas ellas, verificación de archivo desde GitHub). **Compatibilidad con nuestras simulaciones de alfa-sinucleína: NO CONFIRMADA**, porque faltan comprobaciones de secuencia constructo, estado terminal, temperatura, tampón, concentración, fuerza iónica, condiciones de solvente y unidad de q.

## 5. Qué falta para darlo por cerrado

1. Comparar hashes y fragmentos relevantes de código/configuración efectivamente ejecutados en ambas campañas (757 y piloto 500).
2. Derivar desde los inventarios congelados las cifras de entradas, retenidos y excluidos, con motivo verificable por sistema/réplica/observable.
3. Identificar **el** manifiesto de desarrollo/calibración/test, su SHA-256, grupos disjuntos y commit de procedencia, sin abrir resultados de test.
4. Comprobar la equivalencia biológica/fisicoquímica del candidato SAXS con la simulación monomérica concreta.
5. Actualizar la ficha final y subirla a OneDrive; **subida pendiente** hasta verificar acceso al destino.

## 6. Límites de interpretación

- Las trayectorias ATLAS son simulaciones ya ejecutadas; estos análisis no miden ahorro de tiempo de simulación NAMD.
- OOF y `full_train` son productos distintos. Ningún resultado OOF demuestra rendimiento en test externo.
- El informe publicado declara `validation_test_evaluated=false` y `HITO0_passed=false`.
- La nueva ejecución CILS v3 del 12-11-2026 no se ha usado como fuente de nuevos resultados en este cierre.
