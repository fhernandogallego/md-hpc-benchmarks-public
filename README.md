# md-hpc-benchmarks (public)

Code, configuration and pipeline accompanying the manuscript:

> **Metaheuristic Optimization of Temporal Surrogates for Early Prediction of
> Protein Conformational Stability from Short Molecular-Dynamics Trajectories**

This repository contains the analysis code, feature/target-construction pipeline,
optimizer implementation, model-training scripts, HPC (Slurm) job scripts,
configuration files and manifests needed to reproduce the study.

## Data provenance

- **Primary benchmark:** the public **ATLAS** molecular-dynamics dataset
  (Vander Meersche et al., *Nucleic Acids Research*, 2024;
  https://www.dsimb.inserm.fr/ATLAS/). ATLAS is released under CC BY-NC 4.0.
- **External cross-system validation:** in-house p53 DNA-binding-domain
  trajectories generated with NAMD.

**Not included in this repository:**
- Raw trajectory files (DCD/XTC), which are large and, for the p53 systems,
  distributed separately.
- Any restricted data that cannot be publicly redistributed.

Large intermediate/processed arrays are regenerable from public inputs using the
pipeline scripts; see `slurm/` and `src/pipeline/`.

## Repository layout

- `src/` — feature extraction, target construction, optimizer, models, validation.
- `slurm/` — HPC job scripts used on the SCAYLE cluster.
- `configs/`, `config/` — pipeline and optimizer configuration; pinned requirements.
- `manifests/` — system/sample manifests and checksums.
- `docs/` — validation and diagnostic notes.

## Reproducibility

Software versions are pinned in the configuration files. Random seeds, splits and
SHA-256 manifests are recorded alongside the released artifacts.

## Citation

Please cite the manuscript above once published. This repository is referenced in
its Data Availability Statement.

## License

Code is released under the MIT License (see `LICENSE`). Note that the underlying
ATLAS data are subject to their own CC BY-NC 4.0 terms.
