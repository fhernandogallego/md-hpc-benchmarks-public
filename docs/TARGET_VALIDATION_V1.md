# Target validation v1

System: ATLAS 1k5n_A
Date: 2026-08-03

Canonical time resolution: 10 ps
PyMBAR fast: False
PyMBAR nskip: 10

A target is statistically valid when:

- equilibrated duration >= 20 ns
- effective sample size Neff >= 10

Each observable has an independent t0.

A final 80-100 ns window is also calculated as a sensitivity
diagnostic. The current target_valid field does not yet include a
robustness criterion based on the difference between the automatic
window and the final fixed window.

Results:

- R1 RMSD: valid
- R1 Rg: valid
- R2 RMSD: invalid, short window
- R2 Rg: valid
- R3 RMSD: valid, but sensitive to the selected window
- R3 Rg: invalid, short window
