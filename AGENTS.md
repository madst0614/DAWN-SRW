# Local agent instructions

- Do not provide debug scripts or tests to the user.
- Use `C:\Users\MADST\Desktop\dawn-spatial\.venv-jax062\Scripts\python.exe` for local DAWN-SRW tests and validation.
- If validation requires a missing Python package, install it into that virtual environment and continue validation.
- Keep every TPU launcher and watcher on tmux session `train` and remote log `~/train.log`. Do not add per-task session/log names or `--session`/`--log` override options. Clear and start `~/train.log` before remote repo/dependency setup, capture setup failures there, and append the tmux task stream with combined stdout/stderr so `tail -f ~/train.log` never shows a stale prior run.

## TPU access policy

- Use these connection defaults; they are connection settings, not authorization to choose a TPU:
  - GCP project: `dawn-486218`
  - GCP zone: `us-central2-b`
  - Remote SSH user: `madst0614`
  - Remote repository: `/home/madst0614/DAWN-SRW`
  - Bundled gcloud Python: `C:\Users\MADST\gcloud\google-cloud-sdk\platform\bundledpython\python.exe`
  - gcloud entry point: `C:\Users\MADST\gcloud\google-cloud-sdk\lib\gcloud.py`
- For local TPU SSH, default to the bundled invocation:
  `& '<bundled-gcloud-python>' '<gcloud.py>' compute tpus tpu-vm ssh 'madst0614@<EXPLICIT_TPU_NAME>' --zone='us-central2-b' --project='dawn-486218' ...`
- There is no default TPU target. A TPU becomes authorized when the user explicitly names that existing TPU resource in the current task or its follow-up conversation. That authorization persists for the rest of the task and its follow-up turns until the user changes or revokes it; do not ask the user to repeat the exact resource name on every turn. Do not infer authorization from memory, logs, configs, launcher defaults, or currently running jobs.
- If no exact TPU resource has been named anywhere in the current task conversation, or the intended target is ambiguous, stop and ask the user which existing TPU to use before running any TPU gcloud, SSH, launcher, watcher, or remote process command.
- If the user names multiple TPU resources, use only those exact resources and do not discover, select, contact, or modify any others.
- TPU creation and resource allocation are prohibited. Never create, queue, reserve, resize, recreate, provision, or change TPU capacity/topology, and never invoke a keeper or launcher mode that performs those actions. This includes `tpu-vm create`, queued-resource creation, reservations, and equivalent API or script operations.
- On an explicitly named existing TPU, perform the in-scope connection, inspection, training, benchmark, launcher, and log-watching actions needed for the task. Existing placeholder or keep-alive training may be stopped and replaced without asking again unless the user explicitly says to preserve that run.
