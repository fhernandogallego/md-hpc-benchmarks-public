# RMSF validation

System: ATLAS 1k5n_A
Date: 2026-08-04

Canonical RMSF protocol:

- Atom selection: protein and name CA
- Trajectory: ATLAS protein trajectory, already PBC-corrected and aligned
- Additional alignment: none
- First included time: 0.1 ns
- RMSF reference: mean position of each alpha carbon

Comparison against official ATLAS RMSF:

- Mean MAE: approximately 0.0101 A
- Mean RMSE: approximately 0.0132 A
- Minimum Pearson correlation: 0.999819
- Mean bias: approximately 0.0008 A

The direct_from_0.1ns protocol is the canonical implementation.
