# Repository agent instructions

- Do not provide debug scripts or tests to the user.
- Use `C:\Users\MADST\Desktop\dawn-spatial\.venv-jax062\Scripts\python.exe` for local DAWN-SRW tests and validation.
- If validation requires a missing Python package, install it into that virtual environment and continue validation.
- Outside this Windows checkout, use an equivalent repository-local JAX 0.6.2 environment and record the resolved Python, JAX, and Flax versions in `EXPERIMENT.md`. Do not silently validate with a different JAX release line.
- The shared repository is `madst0614/DAWN-SRW`. Record the exact branch, commit, config, checkpoint, mesh, and dirty-worktree state for every material experiment; never rely on a moving branch name alone.
- Keep every TPU launcher and watcher on tmux session `train` and remote log `~/train.log`. Do not add per-task session/log names or `--session`/`--log` override options. Clear and start `~/train.log` before remote repo/dependency setup, capture setup failures there, and append the tmux task stream with combined stdout/stderr so `tail -f ~/train.log` never shows a stale prior run.

## Canonical POC integration policy

- `codex/v4167-poc` is the canonical integration branch and canonical baseline-source branch for all POC work.
- A feature, optimization, profiling, or comparison branch is candidate evidence only. A final accepted architecture, parameter/config schema, variable or metric name, runtime/launcher change, optimization, and baseline decision is not canonical or complete until it is synchronized back to `codex/v4167-poc` and pushed.
- Rejected or inconclusive candidates stay on their experiment branches and are recorded in `EXPERIMENT.md`; do not merge them merely to make every branch an ancestor of POC.
- Measure every new canonical baseline from a clean exact commit reachable from `origin/codex/v4167-poc`. Preserve the actual branch and commit of historical measurements instead of relabeling old evidence after synchronization.
- Before starting or reporting a POC/baseline experiment, fetch the remote branches and verify that every accepted decision relevant to the run is present on `origin/codex/v4167-poc`. Record any intentional unmerged comparison snapshot or rejected branch in `EXPERIMENT.md`.

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
- Commit the notebook with the related code or config change. For runtime-only results, use a notebook-only commit. An experiment is not complete until its notebook entry is committed and pushed to GitHub.
- Never stage or commit unrelated dirty-worktree changes while publishing an experiment record.
