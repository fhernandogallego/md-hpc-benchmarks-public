# EARLYSTOP — cotejo documental SAXS Ahmed et al. (2021) / GB99dms
Fecha: 2026-11-12. **Sin simulaciones nuevas ni ajuste a datos SAXS.** Esta ficha se limita a los datos experimentales y scripts/PDB públicos indicados.

## 1. Material SAXS experimental

- Fuente: Ahmed et al., *Refinement of α-Synuclein Ensembles Against SAXS Data: Comparison of Force Fields and Methods*, Frontiers in Molecular Biosciences 8:654333 (2021). DOI: https://doi.org/10.3389/fmolb.2021.654333
- Archivo medido: https://raw.githubusercontent.com/KULL-Centre/papers/main/2021/aSYN-ahmed-et-al/SAXS/SECSAXS8-chromixs_RAW.dat
- Archivo convertido para PEPSI: https://raw.githubusercontent.com/KULL-Centre/papers/main/2021/aSYN-ahmed-et-al/SAXS/saxs_PEPSI_SAXS_fit.dat
- El RAW contiene **1805** filas numéricas `q I sigma` con incertidumbres `sigma>0`; el archivo para PEPSI contiene 688 filas de la zona de q inicial, con `I` y `sigma` idénticas a las del RAW y `q_PEPSI ≈ q_RAW/10` (proporción de q entre 9.9995 y 10.0004 por redondeo). Ejemplo: `q_RAW=0.0973763` y `q_PEPSI=0.009738`.
- **Unidad q:** la cabecera del RAW *no especifica la unidad*. La conversión exacta de escala ×10 es compatible con RAW expresado en **nm⁻¹** y archivo PEPSI en **Å⁻¹**; se identifica como **inferencia respaldada por la correspondencia numérica de ambos ficheros**, no como una declaración explícita de la cabecera experimental. Antes de emplear q en modelos, fijar explícitamente la unidad y la versión del archivo.
- El artículo documenta alfa-sinucleína humana monomérica (aproximadamente 140 aminoácidos), obtenida por expresión/purificación/liofilización mediante el procedimiento citado de van Maarschalkerweerd et al. (2014). **No aporta en los métodos consultados un fichero FASTA exacto del constructo SAXS ni prueba específica de secuencia, etiquetas, extremos o modificaciones (incluida acetilación N-terminal).** Esos extremos quedan **NO CONFIRMADOS**; no equiparar automáticamente una secuencia canónica con un constructo experimental secuenciado. Referencia de purificación: https://doi.org/10.1021/bm500937p
- Condiciones: PBS con 20 mM de fosfato (Na2HPO4), 150 mM de NaCl, pH 7.4; tampón de corrida con 2% (v/v) glicerol; muestra 4.5 mg/ml **antes** de SEC; medida SEC-SAXS a **20 °C = 293.15 K** en pico monomérico. Ningún ligando fasudil indicado en la preparación SAXS. No asumir concentración exacta en cada fotograma del pico eluido. Fuente: sección *Experimental Data* del artículo Ahmed et al.

## 2. GB99dms: distinguir proteína libre y experimento con ligando

- Trabajo: Greener J. G., *Differentiable simulation to develop molecular dynamics force fields for disordered proteins*, *Chemical Science* 15, 4897–4909 (2024), DOI https://doi.org/10.1039/D3SC05230C. Repo: https://github.com/greener-group/GB99dms
- Guion `sim.py`: temperatura **300 K = 26.85 °C**, ~**6.85 °C por encima** del SAXS; solvente **implícito**; `implicitSolventKappa=0.7/nanometer` (equivalente aproximado a **100 mM** de sal a 300 K, según README). No especifica pH termodinámico; estado de protonación queda implícito en la topología/átomos, pero **pH comparable de simulación no confirmado**.
- La muestra SAXS usa **150 mM NaCl + 20 mM fosfato** y 2% glicerol en tampón SEC, NO la misma parametrización de sales/solvente. La concentración efectiva de 100 mM de GB99dms es aproximación de apantallamiento, no una disolución explícita de NaCl.
- El repositorio tiene **dos tipos de estructura** que no deben confundirse:
  - `structures/disordered/alpha-syn.pdb`: 140 residuos canónicos y ninguna entrada `HETATM` de ligando (modelo sin ligando).
  - `structures/alpha-synuclein/GB99dms/GB99dms_1.pdb` ... `GB99dms_5.pdb`: 140 residuos canónicos de alfa-sinucleína, más un residuo `HETATM FAS` (fasudil) de 37 átomos en cada archivo; por tanto **sí contienen ligando**. Estos modelos se relacionan con el escenario de interacción α-sinucleína/fasudil, no con una muestra SAXS sin fasudil declarado.
- Referencias directas: https://github.com/greener-group/GB99dms/blob/main/sim.py y https://github.com/greener-group/GB99dms/tree/main/structures/alpha-synuclein/GB99dms

## 3. Matriz de compatibilidad documental

| Propiedad | Ahmed 2021 SAXS | GB99dms publicado | Cotejo |
|---|---|---|---|
| Proteína | α-sinucleína humana monomérica | Secuencia estándar de 140 aa en PDB disponibles | Coincidencia de identidad general; **constructo experimental exacto desconocido** |
| Temperatura | 293.15 K | 300 K en `sim.py` | **Difieren: +6.85 K** |
| pH | 7.4 | No consta pH explícito en `sim.py` | No confirmable |
| Sal | 150 mM NaCl y fosfato | Pantalla Debye–Hückel κ=0.7 nm⁻¹, equivalente nominal ~100 mM | **No idénticos** |
| Medio | Tampón SEC con 2% glicerol | Solvente implícito | **No idénticos** |
| Ligando | Fasudil no declarado | PDB de `structures/alpha-synuclein/GB99dms`: FAS presente; `structures/disordered/alpha-syn.pdb`: sin FAS | **Incompatible con estructuras ligadas; aclarar qué conjunto GB99dms se pretende contrastar** |
| Modificaciones del constructo | No confirmadas a nivel secuencia/PTM exacto | No evidencia de modificaciones especiales en PDB canónicos consultados | No asegurar concordancia |
| q | Perfil RAW nm⁻¹ y versión procesada Å⁻¹, conversión inferida de factor 10 | No aplica a trayectorias sin cálculo de scattering | Definir unidades antes de un contraste SAXS |

**Conclusión:** existe un perfil experimental con incertidumbres, pero **NO queda demostrada compatibilidad de condiciones**. Si «GB99dms» significa los cinco PDB con fasudil, hay una **diferencia explícita de ligando**. Ningún resultado de concordancia SAXS o Rg ha sido generado o seleccionado en esta tarea.

**Fuentes:** https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2021.654333/full ; https://github.com/KULL-Centre/papers/tree/main/2021/aSYN-ahmed-et-al/SAXS ; https://github.com/greener-group/GB99dms ; https://doi.org/10.1039/D3SC05230C
