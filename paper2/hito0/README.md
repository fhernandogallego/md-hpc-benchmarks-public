# Paper 2 — HITO 0: escala antes de parada adaptativa

Estado: **preparación experimental**. Este directorio no contiene resultados del
segundo paper ni constituye un protocolo final congelado.

## Pregunta inmediata

Comprobar con proteínas nuevas si un modelo temporal entrenado a mayor escala
aporta señal predictiva sobre los primeros 20 ns. La comparación se hará por
proteína, conservando las tres réplicas juntas y las máscaras de validez de
cada observable.

El plan del 2 de septiembre incluía Ridge y estructura sola. La revisión del
primer paper mostró que **last-value persistence** es un control esencial;
por tanto, también forma parte del HITO 0. `trained_mean` sirve como control
de escala. La mejora del modelo temporal frente a estos controles es una
**hipótesis**, no un resultado obtenido.

## Datos y fronteras

1. Reutilizar el extractor y los criterios de validez del primer paper para
   procesar más proteínas ATLAS. Auditar versiones, time grid, orden de los
   ocho canales, targets, máscaras, procedencia y coste de almacenamiento.
2. Generar el split por `system_id` **antes de examinar los targets de los
   nuevos sistemas**. Los 20 sistemas usados en el primer paper quedan siempre
   en *train*: no pueden convertirse en un nuevo test independiente después de
   haber influido en el desarrollo previo.
3. Usar *train* para el ajuste y para la selección de alpha de Ridge mediante
   CV agrupada; *validation* para desarrollo; *test* una sola vez cuando el
   protocolo, la arquitectura, los observables principales y el criterio de
   decisión estén fijados. Registrar SHA-256 de dataset, split y código.
4. Mantener mdCATH como cohorte **separada** hasta verificar formato, duración,
   temperaturas, correspondencia de los cinco targets y agrupación por dominio.
   No concatenar sus trayectorias al tensor ATLAS asumiendo que comparten
   malla temporal y definición de target. Un dominio presente a varias
   temperaturas tampoco puede quedar a ambos lados de un split.
5. Registrar homología/secuencia de las proteínas ATLAS para analizar después
   si compartir familias entre train y test afecta a la estimación. El split
   implementado protege identidad de proteína, no identidad de familia.

Los scripts de este directorio preparan el split y los controles ATLAS; todavía
faltan el inventario y procesado a escala en SCAYLE, el modelo temporal,
el predictor de estructura sola y la validación mdCATH. El HITO 0 no estará
resuelto hasta compararlos sobre las mismas proteínas y targets válidos.

## Preparar un dataset ATLAS ampliado

Partir de los paquetes individuales `prefix_sequences_raw.npz` ya validados
por el pipeline. Ejecutar estos comandos desde un checkout de este repositorio
que contenga la rama de trabajo del segundo paper. Primero inventariar las
ejecuciones que existen en SCAYLE:

```bash
python -m src.paper2.inventory_atlas_packages \
  --runs-root "$SCRATCH/md-hpc-benchmarks/results/atlas_pipeline_runs" \
  --output-dir "$SCRATCH/md-hpc-benchmarks/results/paper2_hito0_inventory"
```

El inventario informa de duplicados y de targets válidos. Si hay varias
ejecuciones del mismo sistema, **no elige ninguna automáticamente**: se debe
seleccionar una ejecución validada por proteína. El CSV `atlas_sources.csv`
se genera solo cuando no hay duplicados. Necesita **una fila por proteína**,
con columnas `system,release`; `release` es el directorio que contiene
exactamente un `prefix_sequences_raw.npz`. Por ejemplo:

```csv
system,release
1k5n_A,/ruta/validada/1k5n_A
1bzy_A,/ruta/validada/1bzy_A
```

Desde la raíz del repositorio en SCAYLE:

```bash
python -m src.dataset.assemble_multisystem_dataset \
  --source-manifest "$SCRATCH/md-hpc-benchmarks/results/paper2_hito0_inventory/atlas_sources.csv" \
  --output-dir /ruta/atlas_hito0_dataset

python -m src.paper2.make_hito0_split \
  --dataset-npz /ruta/atlas_hito0_dataset/multisystem_dataset_raw.npz \
  --prior-samples releases/atlas20_revision_20260925/data/atlas20_samples.csv \
  --output-dir /ruta/atlas_hito0_split

python -m src.paper2.run_hito0_baselines \
  --dataset-npz /ruta/atlas_hito0_dataset/multisystem_dataset_raw.npz \
  --split-csv /ruta/atlas_hito0_split/protein_split.csv \
  --stage development \
  --output-dir /ruta/atlas_hito0_baselines_dev
```

El ensamblador valida la homogeneidad de las fuentes. El split no consulta
los valores de los targets y conserva los sistemas ATLAS20 en *train*. El
runner verifica los hashes congelados, ajusta y escala Ridge únicamente con
*train*, selecciona alpha por CV agrupada en proteínas de *train* y evalúa
`trained_mean`, `last_value_persistence` y Ridge en *validation*. Agrega las
predicciones de réplicas válidas antes de calcular MAE/RMSE/R² e IC bootstrap
de MAE por proteína. Los CSV y metadatos de salida se guardan fuera del repo.

`--stage final` abre la partición *test*. Ejecutarlo **después** de fijar el
modelo temporal y el criterio de éxito. Las estimaciones de este baseline por
sí solas no prueban que una regla de parada sea válida ni que ahorre core-hours.

## Próximos bloques

- Comprobar en SCAYLE cuántos paquetes ATLAS validados existen realmente y
  cuántos faltan para la escala prevista; estimar espacio y coste por sistema.
- Implementar un modelo temporal de capacidad adecuada con el mismo split y
  las mismas máscaras, y añadir el predictor de estructura sola. Seleccionar
  arquitectura y ajuste con *train/validation*, sin asumir que el optimizador
  Improved-GWO sea la opción más fuerte tras la revisión del primer paper.
- Predefinir el criterio de éxito y los observables de interés **antes** de
  acceder al test. Comparar errores pareados por proteína, con incertidumbre.
- Si la señal predictiva supera el control fuerte, abordar calibración y
  parada secuencial. La garantía válida bajo parada requiere una evaluación
  propia: los intervalos a un tiempo prefijado no bastan para reclamarla.
