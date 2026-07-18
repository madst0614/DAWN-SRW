"""Target/runtime resolution for ``train_analysis_pool``.

Targets own checkpoint and model identity. Runtime profiles own physical device
topology. The effective mesh is the validated application of a runtime to a
target: the target fixes the model axis and the runtime supplies total devices,
so the data axis is derived exactly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "configs" / "train_analysis_pool.yaml"


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    model_version: str
    scale: str
    checkpoint: str
    config: str | None
    mesh_model: int
    compatible_runtimes: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeSpec:
    runtime_id: str
    accelerator_type: str
    global_device_count: int
    worker_count: int


@dataclass(frozen=True)
class MeshResolution:
    runtime_id: str
    accelerator_type: str
    global_device_count: int
    worker_count: int
    mesh_data: int
    mesh_model: int


@dataclass(frozen=True)
class ExecutionSelection:
    target_id: str | None
    checkpoint: str
    config: str | None
    expected_model_version: str | None
    scale: str | None
    mesh: MeshResolution

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mesh"] = asdict(self.mesh)
        return value


def _read_registry(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path or DEFAULT_REGISTRY_PATH)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid train_analysis_pool registry: {source}")
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError(
            "unsupported train_analysis_pool registry schema_version="
            f"{value.get('schema_version')!r}")
    if not isinstance(value.get("targets"), Mapping):
        raise ValueError("train_analysis_pool registry lacks targets")
    if not isinstance(value.get("runtimes"), Mapping):
        raise ValueError("train_analysis_pool registry lacks runtimes")
    return dict(value)


def _positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{field} must be positive, got {result}")
    return result


def target_spec(target_id: str, *, registry_path: str | Path | None = None) -> TargetSpec:
    registry = _read_registry(registry_path)
    requested = str(target_id).strip()
    matches = [
        str(registered)
        for registered in registry["targets"]
        if str(registered).casefold() == requested.casefold()
    ]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous case-insensitive analysis target {target_id!r}: "
            f"{','.join(matches)}")
    key = matches[0] if matches else requested
    raw = registry["targets"].get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"unknown analysis target {target_id!r}; "
            f"known={','.join(registry['targets'])}")
    compatible = tuple(str(value) for value in raw.get("compatible_runtimes", ()))
    if not compatible:
        raise ValueError(f"target {key} has no compatible_runtimes")
    checkpoint = str(raw.get("checkpoint") or "").strip()
    model_version = str(raw.get("model_version") or "").strip()
    if not checkpoint or not model_version:
        raise ValueError(f"target {key} lacks checkpoint or model_version")
    return TargetSpec(
        target_id=key,
        model_version=model_version,
        scale=str(raw.get("scale") or "").strip(),
        checkpoint=checkpoint,
        config=(str(raw["config"]) if raw.get("config") else None),
        mesh_model=_positive_int(raw.get("mesh_model"), f"target {key}.mesh_model"),
        compatible_runtimes=compatible,
    )


def runtime_spec(
        runtime_id: str | None = None, *,
        registry_path: str | Path | None = None) -> RuntimeSpec:
    registry = _read_registry(registry_path)
    key = str(runtime_id or registry.get("default_runtime") or "").strip().lower()
    raw = registry["runtimes"].get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"unknown analysis runtime {key!r}; "
            f"known={','.join(registry['runtimes'])}")
    accelerator_type = str(raw.get("accelerator_type") or "").strip()
    if not accelerator_type:
        raise ValueError(f"runtime {key} lacks accelerator_type")
    return RuntimeSpec(
        runtime_id=key,
        accelerator_type=accelerator_type,
        global_device_count=_positive_int(
            raw.get("global_device_count"),
            f"runtime {key}.global_device_count"),
        worker_count=_positive_int(
            raw.get("worker_count"), f"runtime {key}.worker_count"),
    )


def default_runtime_id(*, registry_path: str | Path | None = None) -> str:
    return runtime_spec(None, registry_path=registry_path).runtime_id


def resolve_execution_selection(
        *, target_id: str | None, checkpoint: str | None,
        config: str | None, runtime_id: str | None,
        registry_path: str | Path | None = None,
        mesh_model_override: int | None = None,
        mesh_data_override: int | None = None) -> ExecutionSelection:
    if bool(target_id) == bool(checkpoint):
        raise ValueError("exactly one of --target or --checkpoint is required")
    runtime = runtime_spec(runtime_id, registry_path=registry_path)
    target = (
        target_spec(str(target_id), registry_path=registry_path)
        if target_id else None)

    if target is not None:
        if runtime.runtime_id not in target.compatible_runtimes:
            raise ValueError(
                f"target {target.target_id} does not support runtime "
                f"{runtime.runtime_id}; compatible="
                f"{','.join(target.compatible_runtimes)}")
        if config is not None:
            raise ValueError(
                "--config cannot override a registered target config")
        if (mesh_model_override is not None
                and int(mesh_model_override) != target.mesh_model):
            raise ValueError(
                f"--mesh-model={mesh_model_override} conflicts with target "
                f"{target.target_id} mesh_model={target.mesh_model}")
        mesh_model = target.mesh_model
        selected_checkpoint = target.checkpoint
        selected_config = target.config
        expected_model_version = target.model_version
        scale = target.scale
    else:
        mesh_model = _positive_int(
            1 if mesh_model_override is None else mesh_model_override,
            "ad-hoc mesh_model")
        selected_checkpoint = str(checkpoint)
        selected_config = config
        expected_model_version = None
        scale = None

    if runtime.global_device_count % mesh_model != 0:
        raise ValueError(
            f"runtime {runtime.runtime_id} devices={runtime.global_device_count} "
            f"is not divisible by mesh_model={mesh_model}")
    mesh_data = runtime.global_device_count // mesh_model
    if (mesh_data_override is not None
            and int(mesh_data_override) != mesh_data):
        raise ValueError(
            f"--mesh-data={mesh_data_override} conflicts with resolved "
            f"target/runtime mesh_data={mesh_data}")
    mesh = MeshResolution(
        runtime_id=runtime.runtime_id,
        accelerator_type=runtime.accelerator_type,
        global_device_count=runtime.global_device_count,
        worker_count=runtime.worker_count,
        mesh_data=mesh_data,
        mesh_model=mesh_model,
    )
    return ExecutionSelection(
        target_id=target.target_id if target is not None else None,
        checkpoint=selected_checkpoint,
        config=selected_config,
        expected_model_version=expected_model_version,
        scale=scale,
        mesh=mesh,
    )


def apply_execution_mesh(
        config: Mapping[str, Any], selection: ExecutionSelection,
        *, visible_device_count: int,
        visible_process_count: int | None = None) -> dict[str, Any]:
    if int(visible_device_count) != selection.mesh.global_device_count:
        raise ValueError(
            f"runtime {selection.mesh.runtime_id} requires "
            f"{selection.mesh.global_device_count} visible JAX devices, got "
            f"{visible_device_count}")
    if (visible_process_count is not None
            and int(visible_process_count) != selection.mesh.worker_count):
        raise ValueError(
            f"runtime {selection.mesh.runtime_id} requires "
            f"{selection.mesh.worker_count} JAX processes/workers, got "
            f"{visible_process_count}")
    value = deepcopy(dict(config))
    model = value.get("model")
    training = value.get("training")
    if not isinstance(model, Mapping) or not isinstance(training, Mapping):
        raise ValueError("checkpoint full_config must contain model and training maps")
    actual_version = str(model.get("model_version") or "")
    if (selection.expected_model_version is not None
            and actual_version != selection.expected_model_version):
        raise ValueError(
            f"target {selection.target_id} expects model_version="
            f"{selection.expected_model_version}, checkpoint declares "
            f"{actual_version!r}")
    checkpoint_mesh_model = _positive_int(
        training.get("mesh_model", selection.mesh.mesh_model),
        "checkpoint training.mesh_model")
    if checkpoint_mesh_model != selection.mesh.mesh_model:
        raise ValueError(
            f"target/runtime mesh_model={selection.mesh.mesh_model} differs "
            f"from checkpoint training.mesh_model={checkpoint_mesh_model}")
    value["training"] = dict(training)
    value["training"]["mesh_model"] = selection.mesh.mesh_model
    value["training"]["mesh_data"] = selection.mesh.mesh_data
    return value


def validate_target_checkpoint_config(
        checkpoint_config: Mapping[str, Any], target: TargetSpec) -> None:
    """Verify that a target label resolves to its declared model identity."""
    if target.config is None:
        raise ValueError(f"target {target.target_id} lacks a canonical config")
    config_path = Path(target.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        declared = yaml.safe_load(handle)
    if not isinstance(declared, Mapping):
        raise ValueError(
            f"target {target.target_id} canonical config is invalid")
    declared_model = declared.get("model")
    checkpoint_model = checkpoint_config.get("model")
    if not isinstance(declared_model, Mapping) or not isinstance(
            checkpoint_model, Mapping):
        raise ValueError("target/checkpoint model config must be mappings")
    differences = [
        key for key, expected in declared_model.items()
        if checkpoint_model.get(key) != expected
    ]
    if differences:
        raise ValueError(
            f"target {target.target_id} model contract differs from "
            "checkpoint full_config: " + ",".join(sorted(differences)))
    declared_mesh_model = _positive_int(
        (declared.get("training") or {}).get("mesh_model"),
        f"target {target.target_id} config training.mesh_model")
    if declared_mesh_model != target.mesh_model:
        raise ValueError(
            f"target {target.target_id} registry mesh_model="
            f"{target.mesh_model} differs from canonical config "
            f"mesh_model={declared_mesh_model}")


def target_runtime_catalog(
        *, registry_path: str | Path | None = None) -> dict[str, Any]:
    registry = _read_registry(registry_path)
    return {
        "schema_version": registry["schema_version"],
        "default_runtime": registry["default_runtime"],
        "targets": deepcopy(dict(registry["targets"])),
        "runtimes": deepcopy(dict(registry["runtimes"])),
    }
