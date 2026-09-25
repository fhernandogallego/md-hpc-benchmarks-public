# ATLAS20 revision reproducibility bundle

Reproducibility bundle for the major revision of manuscript
`mathematics-4566470`.

## Benchmark

The primary benchmark contains 20 ATLAS proteins, three independent
100 ns trajectories per protein, and 60 trajectories in total.

Large ATLAS trajectory files are not redistributed in this bundle.
They can be obtained from the public ATLAS molecular-dynamics dataset.

## Included material

This bundle contains:

- ATLAS20 sample and target-validity manifests;
- frozen protein-level LOPO splits;
- frozen GRU training and held-out evaluation code;
- primary ATLAS20 predictions and metrics;
- trained-mean and last-value-persistence controls;
- Ridge alpha sensitivity;
- paired bootstrap and Wilcoxon-Holm analyses;
- the high-capacity fixed GRU control;
- Improved-GWO compute and search-boundary audits;
- the completed matched-budget HPO comparison;
- reproducibility metadata and checksums.

## Matched-budget HPO comparison

The final comparison contains:

- 60 Improved-GWO optimizer runs;
- 60 Random Search runs;
- 60 Bayesian Optimization runs;
- 20 outer contexts x 3 optimizer seeds;
- 30 algorithmic candidate evaluations per run;
- five protein-group inner folds per candidate.

All three methods use the same frozen ATLAS20 trainer and search space.

The comparison concerns the best frozen inner-validation objective only.
Random-Search- and Bayesian-selected configurations were not evaluated
on the held-out outer tests, so these HPO results do not establish a
held-out predictive ranking among the three optimizers.

The observed ordering of mean best inner-validation fitness was:

1. Bayesian Optimization: 0.201329767335
2. Improved-GWO: 0.213147180580
3. Random Search: 0.230418826053

Paired Wilcoxon tests with Holm correction across the three optimizer
comparisons support all three pairwise differences.

## Numerical failures and caching

No selected best configuration contained a numerically penalized inner fold.

Improved-GWO used candidate caching and recorded 180 cache hits.
Random Search and Bayesian Optimization recorded zero cache hits.

The optimizer comparison is therefore described as using a matched
algorithmic evaluation budget, not equal wall-clock cost or an identical
number of trainer invocations.

## External p53 data

Raw in-house p53/NAMD trajectories are not included.

The p53 analysis in the manuscript is a preliminary cross-engine
zero-shot feasibility assessment and is not a confirmatory external
benchmark.

## Release status

This directory is the public reproducibility snapshot assembled after
completion of the reviewer-requested HPO experiments.

It was released in the public project repository on 2026-09-25 under:

`releases/atlas20_revision_20260925/`
