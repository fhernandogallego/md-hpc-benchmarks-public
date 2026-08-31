# Core-RMSD POC

System: ATLAS 1k5n_A
Date: 2026-08-04

## Operational policy

- Model-eligible core: replica_prefix
- Core detection data: RMSF from 0.1 to 20 ns
- RMSD atoms: backbone atoms of selected residues
- Reference: frame 0
- Alignment: same core backbone atoms
- Weights: equal atom weights

The q30, q40 and q50 core-RMSD series are retained during the POC.
q40 is used as a provisional reporting value, but no global quantile
is selected using only 1k5n_A.

The full-oracle and consensus-full variants are diagnostic only and
must never be used as model inputs.

R1 shows substantial late movement even in the prefix-derived core,
whereas R2 and R3 show much more stable core-RMSD profiles.
