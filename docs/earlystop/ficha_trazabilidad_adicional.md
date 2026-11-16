# EARLYSTOP - cierre adicional de trazabilidad

Fecha: 12/11/2026.
No se realizaron simulaciones ni se abrieron resultados de test.

## Auditorias originales
Se incorporan los cuatro CSV y los dos scripts originales.
SHA256 verificados; ver SHA256_ARCHIVOS_AUDITORIA.csv.

## RMSD original / futuro
1519 replicas invalidas en ambas ventanas;
10 validas en master pero invalidas en futuro;
1243 validas en ambas.
Motivos de las diez: {"NEFF_BAJO": 10}
Detalle: rmsd_diez_replicas_motivos.csv.

## C3 / master
C3 SHA256:
326d9e377eec1ecaf7d0a8694c2404908ab51c3589ac01d4ee628bb00ed20d07
Tablas: c3_vs_master_recuentos.csv y
c3_motivos_por_observable.csv.
Los motivos numericos C3 son diagnosticos por duracion/Neff;
no sustituyen el protocolo propio de C3.
La Figura 2 no se modifica. Master y C3 aplican
criterios de validez diferentes.

## Core q30
Disponibles: 923 sistemas/2769 series en train;
756 sistemas/2268 series en la cohorte RMSD de 757.
Sin serie: 1k5n_A. Causa exacta NO ACREDITADA.
Ver core_q30_1k5n_A.txt.
Disponibilidad de serie NO equivale a target valido.

## Manifiesto historico
SHA256:
bb12305d7ed029ec7c072e1789b357c77420ef35aeca39ed69469217a7109d37
Train 924/817 grupos; validation 9/9; test 9/9.
Commit generador NO ACREDITADO.
Commit publicacion ATLAS942:
aeac56891e06bb002d50ebdd3a1068c84fd2c600

## SAXS Ahmed / GB99dms
Ver nota_saxs_ahmed_gb99dms_20261112.md.
Constructo exacto/PTM experimentales NO CONFIRMADOS.
q RAW/PEPSI: factor 10; conversion inferida.
GB99dms 300 K, solvente implicito, kappa 0.7 nm^-1.
PDB con fasudil FAS. Compatibilidad NO DEMOSTRADA.
