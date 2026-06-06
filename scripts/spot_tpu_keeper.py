#!/usr/bin/env python3
"""Minimal Spot TPU queued-resource keeper.

This script watches one user-specified TPU node-id / queued-resource. It creates
or recreates the Spot queued-resource only when it is missing or terminal, waits
for the TPU VM to become SSH-ready, then calls scripts/launch_tpu_pod.sh once
for that newly available VM.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


PENDING_STATES = {
    "QUEUED",
    "ACCEPTED",
    "WAITING",
    "WAITING_FOR_RESOURCES",
    "PROVISIONING",
    "CREATING",
    "SCHEDULED",
}
TERMINAL_STATES = {
    "FAILED",
    "DELETING",
    "DELETED",
    "CANCELLED",
    "CANCELED",
    "EXPIRED",
}
ACTIVE_STATES = {"ACTIVE"}
NOT_FOUND_MARKERS = (
    "not found",
    "not_found",
    "does not exist",
    "was not found",
    "could not fetch resource",
)
DESCRIBE_ERROR = object()


def default_local_repo() -> str:
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "scripts":
        return str(script_path.parent.parent)
    return "."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal Spot TPU queued-resource keeper. Creates or recreates one "
            "queued-resource, waits for SSH readiness, and calls launch_tpu_pod.sh once."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/spot_tpu_keeper.py \\\n"
            "    --node-id spatial-analysis1 \\\n"
            "    --accelerator-type v4-8 \\\n"
            "    --config configs/train_config_v4162_40M_c4_5B_qinitB_tighter_init.yaml\n\n"
            "  python3 scripts/spot_tpu_keeper.py \\\n"
            "    --node-id spatial-1B \\\n"
            "    --accelerator-type v4-64 \\\n"
            "    --config configs/train_config_v4162_400M_c4_40B_v4_64.yaml"
        ),
    )
    parser.add_argument("--node-id", required=True, help="TPU VM node id to watch/create.")
    parser.add_argument(
        "--accelerator-type",
        required=True,
        help="TPU accelerator type, for example v4-8 or v4-64.",
    )
    parser.add_argument("--config", required=True, help="Training config passed to launch_tpu_pod.sh.")
    parser.add_argument("--branch", default="main", help="Git branch passed to launch_tpu_pod.sh.")
    parser.add_argument("--zone", default="us-central2-b", help="GCP zone.")
    parser.add_argument("--project", default="dawn-486218", help="GCP project.")
    parser.add_argument(
        "--runtime-version",
        default="tpu-ubuntu2204-base",
        help="TPU runtime version for queued-resource creation.",
    )
    parser.add_argument(
        "--queued-resource-id",
        default=None,
        help="Queued-resource id. Defaults to the node id.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval in seconds.")
    parser.add_argument(
        "--local-repo",
        default=default_local_repo(),
        help="Local repository path containing scripts/launch_tpu_pod.sh.",
    )
    parser.add_argument("--token", default=None, help="Optional GitHub token forwarded to launch_tpu_pod.sh.")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if not args.queued_resource_id:
        args.queued_resource_id = args.node_id
    args.local_repo = str(Path(args.local_repo).expanduser().resolve())
    return args


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def is_not_found(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in NOT_FOUND_MARKERS)


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        log(f"Command not found: {cmd[0]} ({exc})")
        return None


def run_streaming(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, text=True, cwd=cwd, check=False)
    except FileNotFoundError as exc:
        log(f"Command not found: {cmd[0]} ({exc})")
        return None


def describe_json(cmd: list[str], label: str) -> dict[str, Any] | None | object:
    proc = run_capture(cmd)
    if proc is None:
        return DESCRIBE_ERROR
    if proc.returncode == 0:
        output = proc.stdout.strip()
        if not output:
            return {}
        try:
            loaded = json.loads(output)
        except json.JSONDecodeError as exc:
            log(f"Could not parse {label} JSON: {exc}")
            return DESCRIBE_ERROR
        if isinstance(loaded, dict):
            return loaded
        log(f"Unexpected {label} JSON type: {type(loaded).__name__}")
        return DESCRIBE_ERROR
    if is_not_found(proc.stderr):
        return None
    log(f"Failed to describe {label}; will retry. stderr: {proc.stderr.strip()}")
    return DESCRIBE_ERROR


def describe_queued_resource(args: argparse.Namespace) -> dict[str, Any] | None | object:
    return describe_json(
        [
            "gcloud",
            "compute",
            "tpus",
            "queued-resources",
            "describe",
            args.queued_resource_id,
            "--zone",
            args.zone,
            "--project",
            args.project,
            "--format=json",
        ],
        f"queued-resource {args.queued_resource_id}",
    )


def describe_tpu_vm(args: argparse.Namespace) -> dict[str, Any] | None | object:
    return describe_json(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            args.node_id,
            "--zone",
            args.zone,
            "--project",
            args.project,
            "--format=json",
        ],
        f"TPU VM {args.node_id}",
    )


def normalize_state(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().split("/")[-1].upper()


def queued_resource_state(qr: dict[str, Any] | None) -> str:
    if qr is None:
        return "MISSING"
    candidates: list[Any] = []
    raw_state = qr.get("state")
    if isinstance(raw_state, dict):
        candidates.extend(
            raw_state.get(key)
            for key in ("state", "value", "name", "status", "phase")
        )
    else:
        candidates.append(raw_state)
    candidates.extend(qr.get(key) for key in ("status", "phase"))
    for candidate in candidates:
        state = normalize_state(candidate)
        if state:
            return state
    return "UNKNOWN"


def tpu_vm_state(node: dict[str, Any] | None) -> str:
    if node is None:
        return "MISSING"
    return normalize_state(node.get("state")) or normalize_state(node.get("status")) or "EXISTS"


def ssh_ready(args: argparse.Namespace) -> bool:
    proc = run_capture(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            args.node_id,
            "--zone",
            args.zone,
            "--project",
            args.project,
            "--worker=all",
            "--command=echo keeper_ssh_ok",
        ]
    )
    return bool(proc and proc.returncode == 0)


def create_spot_queued_resource(args: argparse.Namespace) -> bool:
    log(
        "Creating Spot queued-resource "
        f"{args.queued_resource_id} for node {args.node_id} ({args.accelerator_type})."
    )
    proc = run_streaming(
        [
            "gcloud",
            "compute",
            "tpus",
            "queued-resources",
            "create",
            args.queued_resource_id,
            "--node-id",
            args.node_id,
            "--zone",
            args.zone,
            "--project",
            args.project,
            "--accelerator-type",
            args.accelerator_type,
            "--runtime-version",
            args.runtime_version,
            "--spot",
        ]
    )
    return bool(proc and proc.returncode == 0)


def delete_queued_resource(args: argparse.Namespace) -> bool:
    log(f"Deleting stale queued-resource {args.queued_resource_id}.")
    proc = run_streaming(
        [
            "gcloud",
            "compute",
            "tpus",
            "queued-resources",
            "delete",
            args.queued_resource_id,
            "--zone",
            args.zone,
            "--project",
            args.project,
            "--quiet",
        ]
    )
    return bool(proc and proc.returncode == 0)


def recreate_spot_queued_resource(args: argparse.Namespace, qr_exists: bool) -> None:
    if qr_exists:
        delete_queued_resource(args)
    create_spot_queued_resource(args)


def call_launch_once(args: argparse.Namespace) -> None:
    local_repo = Path(args.local_repo)
    launcher = local_repo / "scripts" / "launch_tpu_pod.sh"
    log(f"Calling launch_tpu_pod.sh once from {local_repo}.")
    if not launcher.exists():
        log(f"Launcher not found at {launcher}; not retrying while this TPU VM remains alive.")
        return

    pull_proc = run_streaming(["git", "pull", "--ff-only"], cwd=str(local_repo))
    if pull_proc is None or pull_proc.returncode != 0:
        log("git pull --ff-only failed; continuing to launch, matching '|| true' behavior.")

    cmd = [
        "bash",
        "scripts/launch_tpu_pod.sh",
        "--tpu",
        args.node_id,
        "--zone",
        args.zone,
        "--project",
        args.project,
        "--branch",
        args.branch,
        "--config",
        args.config,
    ]
    if args.token:
        cmd.extend(["--token", args.token])
    proc = run_streaming(cmd, cwd=str(local_repo))
    if proc is None or proc.returncode != 0:
        log("launch_tpu_pod.sh failed; keeper will not retry while this TPU VM remains alive.")
    else:
        log("launch_tpu_pod.sh completed.")


def state_label(qr_state: str, node_state: str, ssh_is_ready: bool) -> str:
    ssh = "SSH_READY" if ssh_is_ready else "SSH_NOT_READY"
    return f"queued-resource={qr_state}, node={node_state}, {ssh}"


def sleep_poll(args: argparse.Namespace) -> None:
    time.sleep(args.poll_seconds)


def main() -> int:
    args = parse_args()
    log(
        "Watching node "
        f"{args.node_id} / queued-resource {args.queued_resource_id} "
        f"in {args.project}/{args.zone}."
    )

    startup_checked = False
    previously_live = False
    launch_called_for_current_vm = False
    last_reported_state: str | None = None

    while True:
        qr = describe_queued_resource(args)
        node = describe_tpu_vm(args)
        if qr is DESCRIBE_ERROR or node is DESCRIBE_ERROR:
            sleep_poll(args)
            continue

        qr_state = queued_resource_state(qr)
        node_state = tpu_vm_state(node)
        node_exists = node is not None
        should_test_ssh = node_exists or qr_state in ACTIVE_STATES
        ssh_is_ready = ssh_ready(args) if should_test_ssh else False
        current_state = state_label(qr_state, node_state, ssh_is_ready)
        if current_state != last_reported_state:
            log(current_state)
            last_reported_state = current_state

        if not startup_checked:
            startup_checked = True
            if ssh_is_ready:
                log("TPU VM is already SSH-ready at keeper startup; monitoring only.")
                previously_live = True
                launch_called_for_current_vm = True
                sleep_poll(args)
                continue
            log("No SSH-ready TPU VM at keeper startup; will launch after readiness.")

        if ssh_is_ready:
            if not previously_live:
                log("TPU VM is SSH-ready.")
            previously_live = True
            if not launch_called_for_current_vm:
                call_launch_once(args)
                launch_called_for_current_vm = True
            sleep_poll(args)
            continue

        if previously_live and not node_exists:
            log("Previously live TPU VM is gone; treating this as Spot preemption/recovery.")
            previously_live = False
            launch_called_for_current_vm = False
            if qr is None:
                log("Queued-resource is also missing; recreating Spot queued-resource.")
                create_spot_queued_resource(args)
                sleep_poll(args)
                continue
            if qr_state in TERMINAL_STATES or qr_state in ACTIVE_STATES:
                log(f"Queued-resource is stale after node loss ({qr_state}); recreating.")
                recreate_spot_queued_resource(args, qr_exists=True)
                sleep_poll(args)
                continue
            if qr_state in PENDING_STATES:
                log(f"Queued-resource is already {qr_state}; waiting without creating a duplicate.")
                sleep_poll(args)
                continue

        if node_exists:
            log("TPU VM exists but SSH is not ready; waiting without relaunching.")
            sleep_poll(args)
            continue

        if qr is not None and qr_state in TERMINAL_STATES:
            log(f"Queued-resource is terminal ({qr_state}); recreating.")
            recreate_spot_queued_resource(args, qr_exists=True)
            sleep_poll(args)
            continue

        if qr is None:
            log("Queued-resource and TPU VM are missing; creating Spot queued-resource.")
            create_spot_queued_resource(args)
            sleep_poll(args)
            continue

        if qr_state in PENDING_STATES or qr_state in ACTIVE_STATES or qr_state == "UNKNOWN":
            log(f"Queued-resource exists in {qr_state}; waiting without creating a duplicate.")
            sleep_poll(args)
            continue

        log(f"Queued-resource exists in unrecognized non-terminal state {qr_state}; waiting.")
        sleep_poll(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted; exiting.")
        raise SystemExit(130)
