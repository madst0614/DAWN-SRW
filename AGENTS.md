# Local agent instructions

- Do not provide debug scripts or tests to the user.
- Use `C:\Users\MADST\Desktop\dawn-spatial\.venv-jax062\Scripts\python.exe` for local DAWN-SRW tests and validation.
- If validation requires a missing Python package, install it into that virtual environment and continue validation.
- Keep every TPU launcher and watcher on tmux session `train` and remote log `~/train.log`. Do not add per-task session/log names or `--session`/`--log` override options; combine stdout and stderr with `2>&1 | tee ~/train.log`.
