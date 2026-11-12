# Candidato SAXS experimental de alfa-sinucleína monomérica

Fecha de búsqueda: 2026-11-12. Alcance: localizar un primer perfil I(q) con incertidumbres, **sin buscar concordancias favorables con simulaciones**.

## Candidato primario: Ahmed et al. (2021)

- Publicación: Ahmed et al., *Refinement of α-Synuclein Ensembles Against SAXS Data: Comparison of Force Fields and Methods*, *Frontiers in Molecular Biosciences* (2021).
- DOI: https://doi.org/10.3389/fmolb.2021.654333
- Artículo y metodología: https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2021.654333/full
- Repositorio público de los autores: https://github.com/KULL-Centre/papers/tree/main/2021/aSYN-ahmed-et-al/SAXS
- **Archivo directo:** https://raw.githubusercontent.com/KULL-Centre/papers/main/2021/aSYN-ahmed-et-al/SAXS/SECSAXS8-chromixs_RAW.dat
- Página del archivo: https://github.com/KULL-Centre/papers/blob/main/2021/aSYN-ahmed-et-al/SAXS/SECSAXS8-chromixs_RAW.dat

### Identidad de la muestra

- Proteína: alfa-sinucleína humana (SNCA), monomérica, 140 aminoácidos.
- Secuencia canónica de referencia UniProt `P37840` / NCBI `NP_000336.1` (no equivale a verificación de constructo experimental):

```fasta
>sp|P37840|SYUA_HUMAN canonical reference
MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTK
EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDP
DNEAYEMPSEEGYQDYEPEA
```

### Condiciones experimentales documentadas

- Solución inicial: PBS, 20 mM Na2HPO4, 150 mM NaCl, pH 7,4.
- Tampón SEC: PBS con 2 % (v/v) glicerol.
- Concentración de muestra antes de la SEC: 4,5 mg/ml; inyección de 50 µl.
- Separación por cromatografía (SEC) sobre columna Superdex 200 Increase 5/150 GL.
- SAXS en línea P12, PETRA III (DESY/EMBL Hamburg), a 20 °C.
- Perfil final: promedio de fotogramas del pico monomérico, con sustracción de señal del tampón mediante CHROMIXS.

### Integridad y precauciones

- El archivo de los autores tiene 1.805 filas numéricas de tres columnas, interpretables como q, I(q), incertidumbre; comprobado a través de lectura del archivo GitHub.
- Todas las incertidumbres de la tercera columna son > 0; conteo de sigma <= 0: 0. No se han ajustado ni comparado los datos con simulaciones.
- La **unidad de q no consta en la cabecera de este archivo**: verificar en metadatos y flujo del software antes de comparación. Los datos SAXS no son directamente valores de Rg de MD.
- Confirmar del material original la secuencia *exacta* del constructo, etiquetas, modificaciones postraduccionales/acetilación, pureza y estado oligomérico más allá de la descripción general de la publicación.

### Compatibilidad con simulaciones propias

**NO CONFIRMADA.** No se ha identificado aquí una simulación propia de alfa-sinucleína monomérica con coincidencia explícita de secuencia, pH/protonación, sal, temperatura y tratamiento del solvente. No comparar intensidades SAXS o Rg hasta contrastar sus metadatos. El análisis ATLAS942 de proteínas variadas no debe asumirse como una simulación de alfa-sinucleína.

### Referencia alternativa descartada para contraste cuantitativo inmediato

- Dey et al., *Int. J. Biol. Macromol.* 288:138614 (2024), DOI https://doi.org/10.1016/j.ijbiomac.2024.138614; depósito https://www.sasbdb.org/data/SASDQK7/.
- Advertencia de SASBDB: *Experimental errors are incorrectly specified (negative values)*. Por este motivo no utilizar `SASDQK7` como perfil cuantitativo con incertidumbres válidas sin corrección documentada.
- Otro depósito disponible: SASD2L3 (WT, 2026; 20 mM HEPES pH 7,2, sin NaCl, 20 °C) https://www.sasbdb.org/data/SASD2L3/; no se identificó DOI de publicación asociado ni se validó localmente su fichero de incertidumbres.

Fuentes de identidad secuencial: https://www.ncbi.nlm.nih.gov/gene/6622 y UniProt P37840.
