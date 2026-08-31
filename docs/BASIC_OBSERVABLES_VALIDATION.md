# Basic observables validation

System: ATLAS 1k5n_A
Date: 2026-08-03

## Validated calculations

- RMSD: protein backbone, aligned against frame 0, equal atom weights.
- Radius of gyration: all protein atoms, mass weighted.
- Full trajectories: 10001 frames, 10 ps per frame.
- Prefix 20 ns: 2001 frames, including t=0 and t=20 ns.

## Comparison against official ATLAS values

Unweighted RMSD:
- MAE: approximately 0.029-0.032 A
- Pearson correlation: greater than 0.99997

Radius of gyration:
- MAE: approximately 0.0025 A
- Pearson correlation: greater than 0.99994

Mass-weighted RMSD produced slightly larger errors and was rejected.

The unweighted implementation is the canonical implementation for the project.
