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
- **Preliminary cross-engine zero-shot assessment:** in-house p53 DNA-binding-domain
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


## ATLAS20 major-revision reproducibility snapshot

The complete reproducibility snapshot used for the major revision of
manuscript `mathematics-4566470` is available at:

`releases/atlas20_revision_20260925/`

It contains the frozen ATLAS20 protein-level splits, primary results,
reviewer-requested baselines and sensitivity analyses, the high-capacity
fixed-GRU control, optimizer audits, and the completed matched-budget
comparison of Improved-GWO, Random Search and Bayesian Optimization.

The HPO comparison contains 180 optimizer runs
(60 per method), with 30 algorithmic candidate evaluations per run and
five inner folds per candidate. Random-Search- and Bayesian-selected
configurations were not evaluated on held-out outer tests; therefore
the HPO comparison concerns optimization of the frozen inner-validation
objective rather than a held-out predictive ranking.

The snapshot contains its own file inventory and SHA-256 manifest.
Raw ATLAS trajectories and in-house p53/NAMD trajectories are not
redistributed.

## Citation

Please cite the manuscript above once published. This repository is referenced in
its Data Availability Statement.

## License

Code is released under the MIT License (see `LICENSE`). Note that the underlying
ATLAS data are subject to their own CC BY-NC 4.0 terms.
