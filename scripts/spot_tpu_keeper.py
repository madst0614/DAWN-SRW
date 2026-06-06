#!/usr/bin/env python3
"""Event-based Spot TPU queued-resource keeper.

Strict behavior:
  - Before QR is ACTIVE: only read queued-resource state. No SSH probing.
  - QR MISSING: create Spot queued-resource.
  - QR WAITING/PROVISIONING/etc.: wait quietly; log only state changes.
  - QR ACTIVE: start SSH readiness loop.
  - ACTIVE + SSH_READY: call scripts/launch_tpu_pod.sh exactly once for this keeper process.
  - ACTIVE + SSH_NOT_READY longer than timeout: delete QR/node and recreate.
  - QR FAILED/SUSPENDED/etc.: delete QR/node and recreate.
  - All events are written to log.txt by default.
  - No repeated same-state spam unless --verbose is used.

This is not an experiment manager. It does not inspect tmux/training process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


PENDING_QR_STATES = {
    "QUEUED",
    "ACCEPTED",
    "WAITING",
    "WAITING_FOR_RESOURCES",
    "PROVISIONING",
    "CREATING",
    "SCHEDULED",
}

ACTIVE_QR_STATES = {"ACTIVE", "READY"}

TERMINAL_QR_STATES = {
    "FAILED",
    "SUSPENDING",
    "SUSPENDED",
    "DELETING",
    "DELETED",
    "CANCELLED",
    "CANCELED",
    "EXPIRED",
}

NOT_FOUND_MARKERS = (
    "not found",
    "not_found",
    "does not exist",
    "was not found",
    "could not fetch resource",
    "resource was not found",
)

DESCRIBE_ERROR = object()


def default_local_repo() -> str:
    p = Path(__file__).resolve()
    if p.parent.name == "scripts":
        return str(p.parent.parent)
    return "."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-based Spot TPU QR keeper.")
    p.add_argument("--node-id", required=True)
    p.add_argument("--accelerator-type", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--branch", default="main")
    p.add_argument("--zone", default="us-central2-b")
    p.add_argument("--project", default="dawn-486218")
    p.add_argument("--runtime-version", default="tpu-ubuntu2204-base")
    p.add_argument("--queued-resource-id", default=None)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument(
        "--ssh-ready-timeout-seconds",
        type=int,
        default=300,
        help="After QR ACTIVE, recreate if SSH is not ready within this many seconds.",
    )
    p.add_argument(
        "--local-repo",
        default=default_local_repo(),
        help="Local repo path containing scripts/launch_tpu_pod.sh.",
    )
    p.add_argument(
        "--log-file",
        default="log.txt",
        help="Event log file. Relative path is under --local-repo. Default: log.txt",
    )
    p.add_argument("--token", default=None)
    p.add_argument("--verbose", action="store_true", help="Print every poll/state check.")
    args = p.parse_args()

    if args.poll_seconds <= 0:
        p.error("--poll-seconds must be positive")
    if args.ssh_ready_timeout_seconds <= 0:
        p.error("--ssh-ready-timeout-seconds must be positive")
    if not args.queued_resource_id:
        args.queued_resource_id = args.node_id

    args.local_repo = str(Path(args.local_repo).expanduser().resolve())
    log_path = Path(args.log_file).expanduser()
    if not log_path.is_absolute():
        log_path = Path(args.local_repo) / log_path
    args.log_file = str(log_path)

    args._last_qr_state = None
    args._active_since = None
    args._launch_called = False
    args._ssh_wait_logged = False
    return args


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(args: argparse.Namespace, msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    try:
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as exc:
        print(f"[{ts()}] WARN cannot write {args.log_file}: {exc}", flush=True)


def state_event(args: argparse.Namespace, state: str) -> None:
    if args.verbose or state != args._last_qr_state:
        log(args, f"STATE qr={state}")
        args._last_qr_state = state


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return None


def run_stream(args: argparse.Namespace, cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str] | None:
    log(args, "CMD " + " ".join(cmd))
    try:
        return subprocess.run(cmd, text=True, cwd=cwd, check=False)
    except FileNotFoundError as exc:
        log(args, f"ERROR command not found: {cmd[0]} ({exc})")
        return None


def is_not_found(stderr: str) -> bool:
    s = (stderr or "").lower()
    return any(m in s for m in NOT_FOUND_MARKERS)


def describe_json(args: argparse.Namespace, cmd: list[str], label: str) -> dict[str, Any] | None | object:
    proc = run_capture(cmd)
    if proc is None:
        log(args, f"ERROR command not found: {cmd[0]}")
        return DESCRIBE_ERROR

    if proc.returncode == 0:
        out = (proc.stdout or "").strip()
        if not out:
            return {}
        try:
            obj = json.loads(out)
        except json.JSONDecodeError as exc:
            log(args, f"ERROR cannot parse {label} JSON: {exc}")
            return DESCRIBE_ERROR
        if isinstance(obj, dict):
            return obj
        log(args, f"ERROR unexpected {label} JSON type: {type(obj).__name__}")
        return DESCRIBE_ERROR

    if is_not_found(proc.stderr):
        return None

    log(args, f"ERROR describe {label} failed: {(proc.stderr or '').strip()}")
    return DESCRIBE_ERROR


def describe_qr(args: argparse.Namespace) -> dict[str, Any] | None | object:
    return describe_json(
        args,
        [
            "gcloud", "compute", "tpus", "queued-resources", "describe",
            args.queued_resource_id,
            "--zone", args.zone,
            "--project", args.project,
            "--format=json",
        ],
        f"queued-resource {args.queued_resource_id}",
    )


def normalize_state(x: Any) -> str | None:
    if not isinstance(x, str) or not x.strip():
        return None
    return x.strip().split("/")[-1].upper()


def extract_state(x: Any) -> str | None:
    if isinstance(x, str):
        return normalize_state(x)
    if isinstance(x, dict):
        for k in ("state", "value", "name", "status", "phase"):
            v = extract_state(x.get(k))
            if v:
                return v
    return None


def qr_state(qr: dict[str, Any] | None) -> str:
    if qr is None:
        return "MISSING"

    for k in ("state", "queuedResourceState", "status", "phase"):
        v = extract_state(qr.get(k))
        if v:
            return v

    known = PENDING_QR_STATES | ACTIVE_QR_STATES | TERMINAL_QR_STATES
    for v in qr.values():
        s = extract_state(v)
        if s in known:
            return s

    return "UNKNOWN"


def create_qr(args: argparse.Namespace) -> bool:
    log(
        args,
        f"EVENT create_qr qr={args.queued_resource_id} node={args.node_id} "
        f"accel={args.accelerator_type} runtime={args.runtime_version} spot=true",
    )
    proc = run_stream(
        args,
        [
            "gcloud", "compute", "tpus", "queued-resources", "create",
            args.queued_resource_id,
            "--node-id", args.node_id,
            "--zone", args.zone,
            "--project", args.project,
            "--accelerator-type", args.accelerator_type,
            "--runtime-version", args.runtime_version,
            "--spot",
        ],
    )
    return bool(proc and proc.returncode == 0)


def delete_qr(args: argparse.Namespace) -> bool:
    log(args, f"EVENT delete_qr qr={args.queued_resource_id}")
    proc = run_stream(
        args,
        [
            "gcloud", "compute", "tpus", "queued-resources", "delete",
            args.queued_resource_id,
            "--zone", args.zone,
            "--project", args.project,
            "--quiet",
        ],
    )
    return bool(proc and proc.returncode == 0)


def delete_node(args: argparse.Namespace) -> bool:
    # This is best-effort. QR deletion may already remove the node.
    log(args, f"EVENT delete_node node={args.node_id} best_effort=true")
    proc = run_stream(
        args,
        [
            "gcloud", "compute", "tpus", "tpu-vm", "delete",
            args.node_id,
            "--zone", args.zone,
            "--project", args.project,
            "--quiet",
        ],
    )
    return bool(proc and proc.returncode == 0)


def recreate(args: argparse.Namespace, reason: str, qr_exists: bool = True) -> None:
    log(args, f"EVENT stale_recreate reason={reason}")
    if qr_exists:
        delete_qr(args)
    delete_node(args)
    create_qr(args)
    args._active_since = None
    args._launch_called = False
    args._ssh_wait_logged = False
    args._last_qr_state = None


def ssh_ready(args: argparse.Namespace) -> bool:
    proc = run_capture(
        [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh",
            args.node_id,
            "--zone", args.zone,
            "--project", args.project,
            "--worker=all",
            "--command=echo keeper_ssh_ok",
        ]
    )
    return bool(proc and proc.returncode == 0 and "keeper_ssh_ok" in (proc.stdout or ""))


def launch(args: argparse.Namespace) -> None:
    local_repo = Path(args.local_repo)
    launcher = local_repo / "scripts" / "launch_tpu_pod.sh"
    log(args, f"EVENT launch_start repo={local_repo} config={args.config}")

    if not launcher.exists():
        log(args, f"ERROR launcher_missing path={launcher}")
        return

    pull = run_stream(args, ["git", "pull", "--ff-only"], cwd=str(local_repo))
    if pull is None or pull.returncode != 0:
        log(args, "WARN git_pull_failed continuing=true")

    cmd = [
        "bash", "scripts/launch_tpu_pod.sh",
        "--tpu", args.node_id,
        "--zone", args.zone,
        "--project", args.project,
        "--branch", args.branch,
        "--config", args.config,
    ]
    if args.token:
        cmd.extend(["--token", args.token])

    proc = run_stream(args, cmd, cwd=str(local_repo))
    if proc is None or proc.returncode != 0:
        log(args, "EVENT launch_failed will_not_retry_until_qr_changes=true")
    else:
        log(args, "EVENT launch_done")


def reset_for_new_qr_cycle(args: argparse.Namespace) -> None:
    args._active_since = None
    args._launch_called = False
    args._ssh_wait_logged = False


def main() -> int:
    args = parse_args()
    log(
        args,
        f"EVENT keeper_start qr={args.queued_resource_id} node={args.node_id} "
        f"project={args.project} zone={args.zone} accel={args.accelerator_type} "
        f"config={args.config} log_file={args.log_file}",
    )

    while True:
        qr = describe_qr(args)
        if qr is DESCRIBE_ERROR:
            time.sleep(args.poll_seconds)
            continue

        state = qr_state(qr)
        state_event(args, state)

        if state == "MISSING":
            recreate(args, reason="qr_missing", qr_exists=False)
            time.sleep(args.poll_seconds)
            continue

        if state in PENDING_QR_STATES:
            # No SSH here. QR state only until ACTIVE.
            reset_for_new_qr_cycle(args)
            time.sleep(args.poll_seconds)
            continue

        if state in TERMINAL_QR_STATES:
            recreate(args, reason=f"qr_terminal_{state}", qr_exists=True)
            time.sleep(args.poll_seconds)
            continue

        if state in ACTIVE_QR_STATES:
            now = time.time()
            if args._active_since is None:
                args._active_since = now
                args._ssh_wait_logged = False
                log(args, "EVENT active_reached begin_ssh_wait=true")

            if ssh_ready(args):
                if not args._launch_called:
                    log(args, "EVENT ssh_ready")
                    launch(args)
                    args._launch_called = True
                # After launch, keep monitoring QR state only. Do not inspect/restart tmux.
                time.sleep(args.poll_seconds)
                continue

            elapsed = int(now - args._active_since)
            if not args._ssh_wait_logged:
                log(
                    args,
                    f"EVENT ssh_wait active_elapsed={elapsed}s "
                    f"timeout={args.ssh_ready_timeout_seconds}s",
                )
                args._ssh_wait_logged = True
            elif args.verbose:
                log(
                    args,
                    f"DEBUG ssh_wait active_elapsed={elapsed}s "
                    f"timeout={args.ssh_ready_timeout_seconds}s",
                )

            if elapsed >= args.ssh_ready_timeout_seconds:
                recreate(
                    args,
                    reason=f"active_but_ssh_not_ready_{elapsed}s",
                    qr_exists=True,
                )

            time.sleep(args.poll_seconds)
            continue

        # Unknown state: don't duplicate-create. Log once via state_event and wait.
        if args.verbose:
            log(args, f"DEBUG unknown_qr_state={state}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f"[{ts()}] EVENT interrupted", flush=True)
        raise SystemExit(130)
