# Paper 2: reusable SCAYLE batch controller

This controller automates the previously executed future-RMSD experiment. It
does not implement the pending static-structure comparator, mdCATH analysis,
stopping policy or a new claim that HITO 0 is satisfied.

## Run on CALÉNDULA

```bash
bash automation/paper2_batch_v1/run.sh launch --parallel 8
```

Uploads are discovered at `$SCRATCH/md-hpc-benchmarks/raw/atlas/GROUP/SYSTEM/archives`.
Both `SYSTEM_analysis.zip` and `SYSTEM_protein.zip` must exist. Existing systems
in the successful base cohort and the known excluded `3txs_C` are skipped.
The selected systems and their archive hashes are frozen before processing.
Partially uploaded or unsupported directory layouts are not selected.

To select one upload group, add `--group paper2_wave4_v1`. Alternatively use
`--manifest /path/input.csv`, with columns `system,dataset_group` and optional
`analysis_sha256,protein_sha256`. Supplied transfer hashes are checked. Without
them, the controller records local SHA-256 and validates full ZIP CRC integrity;
it does not claim an independent cross-machine transfer-hash comparison.

The first batch extends the audited ATLAS222 artifacts on SCAYLE. Subsequent
batches extend the last successfully completed cohort recorded in
`results/paper2_automation/LATEST_SUCCESS.json`. `--base /path/cohort_state.json`
selects a different completed controller cohort. Failed runs never advance this
pointer. The pipeline defaults to the already executed snapshot at
`results/paper2_hito0/wave3_remaining117_20261014_v1/code`; `--project` overrides it.

## Automatic steps

1. Snapshot code; test the installed NumPy/pandas/PyMBAR/PyTorch runtime on
   synthetic inputs. Check ZIP integrity, paths, free space and available hashes.
2. Unpack safely, reusing identical existing files. Run the real trajectory
   preflight and the system orchestrator's dry-run for every selected system.
3. Submit up to eight independent system pipelines, register job IDs, wait for
   finalization and validate final status, checksums and complete finite features.
4. Assemble a new version. Extract observed sequences, run all-against-all BLAST,
   and join components at >=30% identity and >=80% coverage of both sequences in
   one HSP, with exact-sequence and same-PDB links. Preserve existing assignments;
   stop if a component crosses protected partitions. Added systems enter train.
5. Reuse old future-RMSD targets only when original source CSV hashes still agree.
   New targets retain original t0, use t>20 ns and t>=t0 at 10 ps, and require
   duration >=20 ns and Neff>=10. No validation/test series or future labels are
   used to fit or select models. The assembled original-target cohort remains a
   distinct artifact; the training NPZ contains train systems only.
6. Freeze five group-separated outer folds. Run the published Ridge, persistence,
   trained-mean, first-frame control and 32-unit residual GRU with group-separated
   inner selection. Generalization changes cohort counts, not model hyperparameters.
7. Audit OOF predictions and replica aggregation. Compute exploratory paired
   whole-group bootstrap intervals conditional on the fixed OOF predictions.
8. Select hyperparameters internally on all train data and refit GRU and Ridge
   research candidates, saving `training/full_train/gru_model.pt` and
   `ridge_model.npz`. This refit is separate from the OOF evaluation and does not
   imply model selection, deployment approval or HITO 0 success.

Successful runs print `PAPER2_AUTO=PASS`. Read `summary.json`,
`training/system_metrics.csv`, `training/paired_audit.json`, `state.json` and
`PRISM_PROMPT.txt` in the returned run directory. The prompt is a documentation
handoff; no manuscript or external GitHub branch is modified by the controller.

## Status, recovery and optional cleanup

```bash
bash automation/paper2_batch_v1/run.sh status --run /path/to/run
bash automation/paper2_batch_v1/run.sh resume --run /path/to/run
```

A lock prevents concurrent controllers from advancing incompatible bases.
Completed stages are hash-checked and reused. Jobs already registered as submitted
are polled rather than resubmitted. A submission interrupted before a reliable
job register exists stops for inspection, rather than risk duplicate jobs. Slurm
accounting unknown for five minutes also stops without resubmission. Completed
folds are retained; partial fold attempts are preserved before retry on resume.
Failed scientific system pipelines are reported and block training; they are not
silently discarded or automatically reinterpreted as valid/missing features.
The controller does not automatically retry those failed pipelines.

`--plan-only` verifies, unpacks, preflights and dry-runs without submitting real
system pipelines or training. Use a new launch to execute after a plan-only run.

Add `--cleanup-unpacked` to a launch to remove only the newly selected unpacked
XTC after all training and audits pass. All files must first match archived
members by SHA-256, and ZIP integrity is rechecked. Originals, PDB/TPR, processed
series, targets, previous cohorts and failed systems remain intact. Deletions are
journaled and resumable. Cleanup is disabled by default.

## Validation and scope

```bash
python3 -m unittest discover -s automation/paper2_batch_v1/tests -v
```

Synthetic tests cover damaged archives, traversal/symlinks, free space, duplicate
uploads, partial transfers, pipeline path pinning, job-state ambiguity, phase order,
interruption recovery, protected partitions, source-cache use, future windows,
variable fold sizes, train-only NPZ creation, paired bootstrap, tampering and cleanup.
Slurm, BLAST and MD execution are simulated in orchestration tests. The local
development runtime lacks PyTorch and PyMBAR, so real GRU fitting is not claimed
as locally tested; an actual scientific-runtime smoke test runs in SCAYLE before
processing each batch. Full-site integration still requires its first real run.

The `refs` directory vendors the executed reference code from commit
`260b8e030461676c14b7e2f5029f4ccbc5ca1552`, release
`releases/paper2_atlas222_20261016_v1`. The frozen worker is generalized through
an AST transformation of cohort-size literals. New full-train fitting reuses its
model class, normalization, optimizer, residual target, seed setup and fit function.
