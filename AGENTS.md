# Repository agent instructions

- Do not provide debug scripts or tests to the user.
- Use `C:\Users\MADST\Desktop\dawn-spatial\.venv-jax062\Scripts\python.exe` for local DAWN-SRW tests and validation.
- If validation requires a missing Python package, install it into that virtual environment and continue validation.
- Outside this Windows checkout, use an equivalent repository-local JAX 0.6.2 environment and record the resolved Python, JAX, and Flax versions in `EXPERIMENT.md`. Do not silently validate with a different JAX release line.
- The shared repository is `madst0614/DAWN-SRW`. Record the exact branch, commit, config, checkpoint, mesh, and dirty-worktree state for every material experiment; never rely on a moving branch name alone.
- Keep every TPU launcher and watcher on tmux session `train` and remote log `~/train.log`. Do not add per-task session/log names or `--session`/`--log` override options. Clear and start `~/train.log` before remote repo/dependency setup, capture setup failures there, and append the tmux task stream with combined stdout/stderr so `tail -f ~/train.log` never shows a stale prior run.

## Default paper checkpoints

- The machine-readable registry is `configs/paper_checkpoint_registry.yaml`.
- Treat the following user-designated GCS run roots as the default checkpoints
  for paper planning and future evaluation setup:
  - DAWN-SRW v4172 400M:
    `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201`
  - DAWN-SRW v4172 1.3B:
    `gs://dawn-tpu-data-c4/checkpoints/train_config_v4172_1p3B_c4_20B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260717_173556_3201`
  - baseline-JAX 400M:
    `gs://dawn-tpu-data-c4/checkpoints/baseline_400M_c4_40B_v4_64/run_vbaseline-JAX_20260713_194029_3201`
  - baseline-JAX 1.3B:
    `gs://dawn-tpu-data-c4/checkpoints/baseline_jax_1p3B_c4_20B_v4_64/run_vbaseline-JAX_20260729_093206_3201`
- This registry is descriptive only. A run must still resolve and record one
  committed numeric checkpoint step. These paths do not choose, authorize,
  create, resize, or otherwise grant access to any TPU resource.

## Canonical integration and research archive policy

- `main` is the sole canonical integration, baseline-source, and final-reporting branch. The former `codex/v4167-poc` integration branch was fast-forwarded into `main` during the 2026-07-31 project consolidation.
- A feature, optimization, profiling, or comparison branch is candidate evidence only. A final accepted architecture, parameter/config schema, variable or metric name, runtime/launcher change, optimization, and baseline decision is not canonical or complete until it is synchronized back to `main` and pushed.
- Experiments may run on short-lived candidate branches, but final reporting is centralized on `main`: after every material attempt, synchronize the resulting `EXPERIMENT.md` entry and current-state decision to `main`, whether the candidate was accepted, rejected, blocked, or inconclusive.
- Rejected or inconclusive source must not be merged into active production code merely for ancestry. Preserve it as commit patches under `research/branch_archive/`, record its exact commit identity in the manifest, and retain an immutable `research-archive/...` tag before deleting its working branch.
- Measure every new canonical baseline from a clean exact commit reachable from `origin/main`. Preserve the actual branch and commit of historical measurements instead of relabeling old evidence after consolidation.
- Before starting or reporting an experiment, fetch `origin/main` and verify that every accepted decision relevant to the run is present there. Archived patches and tags are research evidence, not canonical executable source.

## TPU access policy

- Use these connection defaults; they are connection settings, not authorization to choose a TPU:
  - GCP project: `dawn-486218`
  - GCP zone: `us-central2-b`
  - Remote SSH user: `madst0614`
  - Remote repository: `/home/madst0614/DAWN-SRW`
  - Bundled gcloud Python: `C:\Users\MADST\gcloud\google-cloud-sdk\platform\bundledpython\python.exe`
  - gcloud entry point: `C:\Users\MADST\gcloud\google-cloud-sdk\lib\gcloud.py`
- On the current Windows host, default to the bundled invocation:
  `& '<bundled-gcloud-python>' '<gcloud.py>' compute tpus tpu-vm ssh 'madst0614@<EXPLICIT_TPU_NAME>' --zone='us-central2-b' --project='dawn-486218' ...`
- In another environment, use its authenticated `gcloud` installation with the same explicit project, zone, SSH user, and TPU target. Machine-specific executable paths are not part of experiment identity.
- There is no default TPU target. A TPU becomes authorized when the user explicitly names that existing TPU resource in the current task or its follow-up conversation. That authorization persists for the rest of the task and its follow-up turns until the user changes or revokes it; do not ask the user to repeat the exact resource name on every turn. Do not infer authorization from memory, logs, configs, launcher defaults, or currently running jobs.
- If no exact TPU resource has been named anywhere in the current task conversation, or the intended target is ambiguous, stop and ask the user which existing TPU to use before running any TPU gcloud, SSH, launcher, watcher, or remote process command.
- If the user names multiple TPU resources, use only those exact resources and do not discover, select, contact, or modify any others.
- TPU creation and resource allocation are prohibited. Never create, queue, reserve, resize, recreate, provision, or change TPU capacity/topology, and never invoke a keeper or launcher mode that performs those actions. This includes `tpu-vm create`, queued-resource creation, reservations, and equivalent API or script operations.
- On an explicitly named existing TPU, perform the in-scope connection, inspection, training, benchmark, launcher, and log-watching actions needed for the task. Existing placeholder or keep-alive training may be stopped and replaced without asking again unless the user explicitly says to preserve that run.

## Experiment continuity

- `EXPERIMENT.md` is the canonical shared experiment notebook and the source of continuity across machines and agents.
- Before experiment-related work, read its `Current State`, `Fixed Context`, and relevant recent entries. Historical TPU names in the notebook are evidence only and never grant access authorization.
- After every material implementation or runtime attempt, append one concise KST-dated entry, including failed, blocked, and inconclusive attempts. Use `YYYY-MM-DD/#NN` as the stable experiment identifier.
- Update `Current State` only when the accepted conclusion, current blocker, or next experiment changes. Preserve old entries; record corrections in a later entry instead of rewriting history.
- Separate TPU measurements, local validation, user-provided context, and inference. Never report an unrun validation as successful, and write `not measured` when evidence is absent.
- Record the hypothesis, change, exact run identity, result, lesson, decision, next step, and stable evidence location. Keep raw logs out of Git; preserve them remotely and link their paths.
- Candidate code/config may be committed on a short-lived experiment branch. Commit its final notebook report separately on `main`; for runtime-only or rejected/inconclusive results, use a reporting-only commit and archive the candidate patch series. An experiment is not complete until the centralized notebook entry is committed and pushed to GitHub.
- Never stage or commit unrelated dirty-worktree changes while publishing an experiment record.
