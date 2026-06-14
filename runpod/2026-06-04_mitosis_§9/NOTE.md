# §9 run archive — 2026-06-04 mitosis run (RTX 3090)

This is the third-act run of the Molequla paper (`docs/molequla_paper.md`, §9):
four organisms grown embryo→adult on a single RTX 3090, no corpus seeding,
with two natural divide events captured.

## What is committed here (the run logs)

- `work_{fire,air,water,earth}/train*.log` — per-stage climb logs for all four
  organisms (`train.log`, `train_climb_*.log`, `train_resume*_*.log`). All four
  reach `stage=5` (adult).
- `work_fire/train.log:51` — Fire's divide on the **loss** path
  (`overload=true (e=false l=true)` → child `org_1780540885_6400`).
- `work_air/train_climb_air.log:273` + `work_air/train_resume2_air.log:63` — Air's
  divide on the **entropy** path (`high=8/8 mean=6.256`, `e=true l=false`
  → child `org_1780527018_6475`).
- `capture/util.log` — nvidia-smi GPU-utilization samples (the four-org run held
  the 0–20 % band; the 0→99 % launch-bound fix figures are from a separate
  dedicated fix-verification pod).
- `capture/dna_snap/{air,earth,fire,water}/gen_*.txt` — DNA voice snapshots
  written by the organisms during the climb.
- `org_1780540885_6400/birth.json` — the first child's birth manifest
  (parent config + inherited burst history + checkpoint path).
- `mitosis_artifacts.tgz` — small bundled divide artifacts.

## What is NOT in git (mirrored off-repo)

The weight checkpoints are hundreds of MB each and exceed GitHub's 100 MB
per-file limit, so they are kept off-repo (on the polygon node, available on
request), not in this archive on GitHub:

- `final_weights.tgz` (~422 MB) — all four adult weight bundles.
- `final_state/work_{fire,air,water,earth}/molequla_ckpt.json` (~209–266 MB each)
  — final adult checkpoints.
- `org_1780540885_6400/parent_ckpt.json` (~250 MB) — the checkpoint Fire's child
  inherited (real lineage; the child loaded the parent's adult weights, n_embd 320).
- `child_artifacts.tgz` (~106 MB) — the spawned child's bundled artifacts.

The logs above are sufficient to reproduce every numerical claim in §9. The
weights are available for re-instantiation on request.
