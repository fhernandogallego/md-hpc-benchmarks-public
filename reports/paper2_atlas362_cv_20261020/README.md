# ATLAS362 exploratory future-RMSD CV result record

This record transcribes the supplied successful controller summary and metric CSV.
The reported cohort contains 362 PDB-chain systems and 1,086 trajectories. Of the
344 train systems, 286 systems in 258 sequence groups have at least one valid
future-RMSD label, covering 477 valid replicas. The 1,032 train input samples
include replicas without valid labels; they are not all used as supervised targets.
The 318 groups count covers the whole cohort. The cache count of 612 denotes
previous target-table rows, including invalid rows.

Five outer folds contain 58/57/57/57/57 labeled systems, with sequence groups
kept together and grouped inner hyperparameter selection. Metrics aggregate
valid replicas within each system and then weight systems equally. Targets retain
the original window start and use t > 20 ns and t >= t0; the original operational
duration and effective-sample-size criteria remain in force.

GRU MAE is 0.751605 A, persistence MAE 0.790727 A and Ridge MAE 0.772463 A.
The paired GRU-minus-persistence difference is -0.039122 A, with an exploratory
95% whole-group bootstrap interval [-0.076778, -0.003907] A. The GRU-minus-Ridge
interval [-0.102876, +0.056521] A includes zero. Ridge has slightly lower RMSE
and higher R2 than GRU. These intervals condition on the fitted OOF predictions;
they do not include variation from refitting or repeated model/data decisions.
They support an exploratory development finding rather than confirmatory superiority.

The expanded cohort and folds differ from ATLAS222. Their aggregate MAEs alone
do not establish a learning curve or deterioration/improvement on the same
evaluation population. Original validation and test assignments were retained;
neither was evaluated in this run. Snapshot-zero Ridge remains a first-frame
control, not the pending genuine static-structure comparator.

The 2wpq_C package passed recovery after an empty consensus_full_oracle diagnostic
was recorded as unavailable. Prefix-core criteria and model feature values were
preserved. Full-oracle diagnostic completeness must be reported separately from
model-input/package validity. The earlier exclusion of 3txs_C remains in force.

Full-train GRU and Ridge research checkpoints were reported saved on the cluster.
They are distinct from the OOF models used for the metrics. Their selected
hyperparameters, checkpoint hashes and the complete local evidence bundle still
need to be attached to the public record; no checkpoint or new dataset hash is
invented here. Input/code/OOF artifacts are to be frozen before optional XTC
cleanup. The public record contains summary data only, without host paths/accounts.

HITO 0 is still pending. This record establishes neither held-out performance,
general superiority over Ridge, sequential/anytime-valid coverage, nor measured
simulation or HPC savings. The static comparator, mdCATH compatibility and stopping
policy experiments remain outside this run.
