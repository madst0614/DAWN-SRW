"""Dynamic answer-position RW operator programs and immutable artifacts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.dawn_analysis_storage import read_npz, write_npz_atomic
from analysis.operator_interpretability.artifacts import sha256_path
from analysis.operator_interpretability.benchmark_schema import (
    BenchmarkExample,
    canonical_hash,
)
from analysis.operator_interpretability.protocol import (
    NATIVE_PROGRAM_CLAIM_LADDER,
    ProtocolConfig,
)
from analysis.operator_interpretability.units import OperatorSpaceShape


PROGRAM_ARTIFACT_SCHEMA_VERSION = 1
PROGRAM_ALGORITHM_VERSION = "answer_position_mass_prefix_v1"
PROGRAM_ROUTES = ("q", "k", "v", "rst")
PROGRAM_MODES = {
    "production": 0,
    "own_id_replay": 1,
    "own_id_ablation": 2,
    "source_id_replay": 3,
    "source_contribution_transplant": 4,
}


def _pool_size(shape: OperatorSpaceShape, route: str) -> int:
    return int(shape.pool_size(route))


def _total_operator_sites(shape: OperatorSpaceShape) -> int:
    return int(shape.n_layers) * (
        2 * int(shape.n_qk) + int(shape.n_v) + int(shape.n_rst))


def _program_digest(
        *, example_id: str, prompt_side: str, program_mass: float,
        ids: Mapping[str, np.ndarray], valid: Mapping[str, np.ndarray],
        example_index: int) -> str:
    digest = hashlib.sha256()
    digest.update(PROGRAM_ALGORITHM_VERSION.encode("ascii"))
    digest.update(str(example_id).encode("utf-8"))
    digest.update(str(prompt_side).encode("utf-8"))
    digest.update(np.asarray([program_mass], dtype="<f8").tobytes())
    for route in PROGRAM_ROUTES:
        route_ids = np.asarray(ids[route][:, example_index], dtype="<i4")
        route_valid = np.asarray(
            valid[route][:, example_index], dtype=np.bool_)
        digest.update(route.encode("ascii"))
        for layer in range(route_ids.shape[0]):
            selected = route_ids[layer, route_valid[layer]]
            digest.update(np.asarray([layer, selected.size], dtype="<i4").tobytes())
            digest.update(np.asarray(selected, dtype="<i4").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class OperatorProgramSchedule:
    """Fixed-width, batch-specific multilayer operator ID schedule."""

    example_ids: tuple[str, ...]
    prompt_side: str
    program_mass: float
    ids: Mapping[str, np.ndarray]
    valid: Mapping[str, np.ndarray]
    records: tuple[Mapping[str, Any], ...]
    schedule_hash: str

    @property
    def batch_size(self) -> int:
        return len(self.example_ids)

    @property
    def widths(self) -> dict[str, int]:
        return {
            route: int(np.asarray(self.ids[route]).shape[-1])
            for route in PROGRAM_ROUTES
        }

    def validate(self, shape: OperatorSpaceShape) -> "OperatorProgramSchedule":
        if not self.example_ids:
            raise ValueError("operator program schedule is empty")
        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("operator program example ids must be unique")
        if not 0.0 < float(self.program_mass) <= 1.0:
            raise ValueError("operator program mass must be in (0, 1]")
        if set(self.ids) != set(PROGRAM_ROUTES):
            raise ValueError("operator program ids must contain q,k,v,rst")
        if set(self.valid) != set(PROGRAM_ROUTES):
            raise ValueError("operator program validity must contain q,k,v,rst")
        for route in PROGRAM_ROUTES:
            ids = np.asarray(self.ids[route])
            valid = np.asarray(self.valid[route])
            if ids.dtype != np.int32:
                raise TypeError(f"{route} program ids must have int32 dtype")
            if valid.dtype != np.bool_:
                raise TypeError(f"{route} program validity must have bool dtype")
            if ids.ndim != 3 or ids.shape[:2] != (
                    int(shape.n_layers), self.batch_size) or ids.shape[2] <= 0:
                raise ValueError(
                    f"{route} program ids must have shape [layers,batch,width], "
                    f"got {ids.shape}")
            if valid.shape != ids.shape:
                raise ValueError(f"{route} program validity shape mismatch")
            if np.any(ids[~valid] != 0):
                raise ValueError(
                    f"{route} invalid padded program ids must be exactly zero")
            selected = ids[valid]
            if selected.size and (
                    int(selected.min()) < 0
                    or int(selected.max()) >= _pool_size(shape, route)):
                raise ValueError(f"{route} program contains an out-of-range id")
            for layer in range(ids.shape[0]):
                for example_index in range(ids.shape[1]):
                    row = ids[layer, example_index, valid[layer, example_index]]
                    if len(set(int(value) for value in row)) != row.size:
                        raise ValueError(
                            f"{route} program contains duplicate ids at "
                            f"layer={layer} example={example_index}")
        if len(self.records) != self.batch_size:
            raise ValueError("operator program record count mismatch")
        expected_hash = _schedule_hash(
            self.example_ids, self.prompt_side, self.program_mass,
            self.ids, self.valid)
        if str(self.schedule_hash) != expected_hash:
            raise ValueError("operator program schedule hash mismatch")
        return self


def _schedule_hash(
        example_ids: Sequence[str], prompt_side: str, program_mass: float,
        ids: Mapping[str, np.ndarray], valid: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(PROGRAM_ALGORITHM_VERSION.encode("ascii"))
    digest.update(str(prompt_side).encode("utf-8"))
    digest.update(np.asarray([program_mass], dtype="<f8").tobytes())
    for example_id in example_ids:
        encoded = str(example_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    for route in PROGRAM_ROUTES:
        digest.update(route.encode("ascii"))
        digest.update(np.asarray(ids[route], dtype="<i4").tobytes())
        digest.update(np.asarray(valid[route], dtype=np.bool_).tobytes())
    return digest.hexdigest()


def capture_schedule_widths(
        captures: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return one static route width shared by every mass candidate."""
    widths = {route: 1 for route in PROGRAM_ROUTES}
    for capture in captures:
        for row in capture.get("rows") or ():
            route = str(row["route"])
            if route in widths:
                widths[route] = max(
                    widths[route], len(row.get("operator_ids") or ()))
    qk_width = max(widths["q"], widths["k"])
    widths["q"] = qk_width
    widths["k"] = qk_width
    return widths


def _selected_prefix(
        row: Mapping[str, Any], program_mass: float) -> np.ndarray:
    captured_mass = float(row["captured_mass"])
    if not math.isfinite(captured_mass) or not 0.0 < captured_mass <= 1.000001:
        raise ValueError("program capture has invalid captured_mass")
    if float(program_mass) > captured_mass + 1.0e-7:
        raise RuntimeError(
            "program capture cannot reach the preregistered mass: "
            f"program_mass={program_mass} captured_mass={captured_mass} "
            f"example={row.get('example_id')} layer={row.get('layer')} "
            f"route={row.get('route')}")
    operator_ids = np.asarray(row["operator_ids"], dtype=np.int32)
    weights = np.asarray(row["weights"], dtype=np.float64)
    if operator_ids.ndim != 1 or weights.shape != operator_ids.shape:
        raise ValueError("program capture ids and weights must be aligned vectors")
    if (operator_ids.size == 0 or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)):
        raise ValueError("program capture weights must be finite and nonnegative")
    if np.any(weights[1:] > weights[:-1] + 1.0e-12):
        raise ValueError("program capture weights are not descending")
    captured_weight_sum = float(weights.sum())
    if not captured_weight_sum > 0.0:
        raise RuntimeError("program capture has zero captured contribution mass")
    estimated_total_mass = captured_weight_sum / captured_mass
    required_mass = float(program_mass) * estimated_total_mass
    cumulative = np.cumsum(weights, dtype=np.float64)
    if cumulative[-1] + 1.0e-10 * max(1.0, required_mass) < required_mass:
        raise RuntimeError("captured prefix does not reach required program mass")
    count = int(np.searchsorted(cumulative, required_mass, side="left")) + 1
    return operator_ids[:count]


def _records_from_arrays(
        *, example_ids: Sequence[str], prompt_side: str,
        program_mass: float, ids: Mapping[str, np.ndarray],
        valid: Mapping[str, np.ndarray], shape: OperatorSpaceShape,
        captured_mass: np.ndarray | None = None,
        extra: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    total_sites = _total_operator_sites(shape)
    records = []
    for example_index, example_id in enumerate(example_ids):
        per_layer_route: dict[str, dict[str, int]] = {}
        per_layer_mass: dict[str, dict[str, float | None]] = {}
        route_counts = {route: 0 for route in PROGRAM_ROUTES}
        for layer in range(int(shape.n_layers)):
            layer_counts = {}
            layer_mass = {}
            for route_index, route in enumerate(PROGRAM_ROUTES):
                count = int(np.sum(valid[route][layer, example_index]))
                layer_counts[route] = count
                route_counts[route] += count
                layer_mass[route] = (
                    float(captured_mass[layer, example_index, route_index])
                    if captured_mass is not None else None)
            per_layer_route[str(layer)] = layer_counts
            per_layer_mass[str(layer)] = layer_mass
        site_count = int(sum(route_counts.values()))
        record = {
            "example_id": str(example_id),
            "prompt_side": str(prompt_side),
            "program_mass": float(program_mass),
            "site_count": site_count,
            "site_fraction_of_total_operator_sites": (
                site_count / max(total_sites, 1)),
            "per_layer_route_site_count": per_layer_route,
            "per_layer_route_captured_mass": per_layer_mass,
            "per_route_site_count": route_counts,
            "program_hash": _program_digest(
                example_id=str(example_id), prompt_side=prompt_side,
                program_mass=program_mass, ids=ids, valid=valid,
                example_index=example_index),
        }
        if extra is not None:
            record.update(dict(extra[example_index]))
        records.append(record)
    return tuple(records)


def build_program_schedule(
        capture: Mapping[str, Any], examples: Sequence[BenchmarkExample], *,
        shape: OperatorSpaceShape, program_mass: float, prompt_side: str,
        widths: Mapping[str, int] | None = None) -> OperatorProgramSchedule:
    """Select captured-mass-corrected prefixes for every layer and route."""
    if capture.get("status") != "ready":
        raise ValueError("operator program capture is not ready")
    if prompt_side not in {"base", "source"}:
        raise ValueError("program prompt_side must be base or source")
    example_ids = tuple(str(example.example_id) for example in examples)
    if not example_ids:
        raise ValueError("operator program examples are empty")
    rows = list(capture.get("rows") or ())
    by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["example_id"]), int(row["layer"]), str(row["route"]))
        if key in by_key:
            raise ValueError(f"duplicate program capture row: {key}")
        by_key[key] = row
    expected_keys = {
        (example_id, layer, route)
        for example_id in example_ids
        for layer in range(int(shape.n_layers))
        for route in PROGRAM_ROUTES
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        extra = sorted(set(by_key) - expected_keys)
        raise ValueError(
            "program capture row contract mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}")
    selected: dict[tuple[str, int, str], np.ndarray] = {}
    required_widths = {route: 1 for route in PROGRAM_ROUTES}
    captured_mass = np.zeros(
        (int(shape.n_layers), len(example_ids), len(PROGRAM_ROUTES)),
        dtype=np.float64)
    for example_index, example_id in enumerate(example_ids):
        for layer in range(int(shape.n_layers)):
            for route_index, route in enumerate(PROGRAM_ROUTES):
                row = by_key[(example_id, layer, route)]
                prefix = _selected_prefix(row, float(program_mass))
                if len(set(int(value) for value in prefix)) != prefix.size:
                    raise ValueError("program prefix contains duplicate operator ids")
                selected[(example_id, layer, route)] = prefix
                required_widths[route] = max(
                    required_widths[route], int(prefix.size))
                captured_mass[layer, example_index, route_index] = float(
                    row["captured_mass"])
    fixed_widths = {
        route: int((widths or required_widths)[route])
        for route in PROGRAM_ROUTES
    }
    for route in PROGRAM_ROUTES:
        if fixed_widths[route] < required_widths[route]:
            raise ValueError(
                f"fixed {route} program width is smaller than selected prefix")
    ids = {
        route: np.zeros(
            (int(shape.n_layers), len(example_ids), fixed_widths[route]),
            dtype=np.int32)
        for route in PROGRAM_ROUTES
    }
    valid = {route: np.zeros_like(ids[route], dtype=np.bool_)
             for route in PROGRAM_ROUTES}
    for example_index, example_id in enumerate(example_ids):
        for layer in range(int(shape.n_layers)):
            for route in PROGRAM_ROUTES:
                prefix = selected[(example_id, layer, route)]
                ids[route][layer, example_index, :prefix.size] = prefix
                valid[route][layer, example_index, :prefix.size] = True
    records = _records_from_arrays(
        example_ids=example_ids, prompt_side=prompt_side,
        program_mass=float(program_mass), ids=ids, valid=valid, shape=shape,
        captured_mass=captured_mass)
    schedule = OperatorProgramSchedule(
        example_ids=example_ids,
        prompt_side=prompt_side,
        program_mass=float(program_mass),
        ids=ids,
        valid=valid,
        records=records,
        schedule_hash=_schedule_hash(
            example_ids, prompt_side, float(program_mass), ids, valid),
    )
    return schedule.validate(shape)


def reindex_program_schedule(
        schedule: OperatorProgramSchedule, donor_indices: Sequence[int], *,
        recipient_example_ids: Sequence[str], prompt_side: str,
        shape: OperatorSpaceShape) -> OperatorProgramSchedule:
    indices = np.asarray(donor_indices, dtype=np.int64)
    if indices.shape != (len(recipient_example_ids),):
        raise ValueError("program donor index shape mismatch")
    if np.any(indices < 0) or np.any(indices >= schedule.batch_size):
        raise ValueError("program donor index is out of range")
    ids = {route: np.asarray(schedule.ids[route])[:, indices].copy()
           for route in PROGRAM_ROUTES}
    valid = {route: np.asarray(schedule.valid[route])[:, indices].copy()
             for route in PROGRAM_ROUTES}
    donor_ids = [schedule.example_ids[int(index)] for index in indices]
    records = _records_from_arrays(
        example_ids=recipient_example_ids, prompt_side=prompt_side,
        program_mass=schedule.program_mass, ids=ids, valid=valid,
        shape=shape,
        extra=[{"donor_example_id": donor_id}
               for donor_id in donor_ids])
    result = OperatorProgramSchedule(
        example_ids=tuple(map(str, recipient_example_ids)),
        prompt_side=prompt_side,
        program_mass=schedule.program_mass,
        ids=ids,
        valid=valid,
        records=records,
        schedule_hash=_schedule_hash(
            tuple(map(str, recipient_example_ids)), prompt_side,
            schedule.program_mass, ids, valid),
    )
    return result.validate(shape)


def deterministic_mismatch_mapping(
        examples: Sequence[BenchmarkExample],
        schedule: OperatorProgramSchedule, *, seed: int) -> dict[str, Any]:
    if len(examples) != schedule.batch_size or len(examples) < 2:
        raise ValueError("mismatched programs require at least two aligned rows")
    counts = [int(record["site_count"]) for record in schedule.records]
    templates = [str(example.metadata.get("template")) for example in examples]
    donor_indices = []
    rows = []
    for index, example in enumerate(examples):
        eligible = [candidate for candidate in range(len(examples))
                    if candidate != index]
        same_template = [candidate for candidate in eligible
                         if templates[candidate] == templates[index]]
        candidates = same_template or eligible
        donor = min(candidates, key=lambda candidate: (
            abs(counts[candidate] - counts[index]),
            canonical_hash({
                "seed": int(seed),
                "recipient_example_id": example.example_id,
                "donor_example_id": examples[candidate].example_id,
            }),
            str(examples[candidate].example_id),
        ))
        donor_indices.append(donor)
        rows.append({
            "recipient_example_id": str(example.example_id),
            "donor_example_id": str(examples[donor].example_id),
            "same_template": templates[donor] == templates[index],
            "recipient_site_count": counts[index],
            "donor_site_count": counts[donor],
            "site_count_distance": abs(counts[donor] - counts[index]),
        })
    return {
        "rule": "same_template_nearest_site_count_seeded",
        "seed": int(seed),
        "donor_indices": donor_indices,
        "rows": rows,
        "mapping_hash": canonical_hash(rows),
    }


def random_program_schedule(
        schedule: OperatorProgramSchedule, *, shape: OperatorSpaceShape,
        seed: int) -> OperatorProgramSchedule:
    ids = {route: np.zeros_like(schedule.ids[route], dtype=np.int32)
           for route in PROGRAM_ROUTES}
    valid = {route: np.asarray(schedule.valid[route]).copy()
             for route in PROGRAM_ROUTES}
    for route in PROGRAM_ROUTES:
        pool_size = _pool_size(shape, route)
        for layer in range(int(shape.n_layers)):
            for example_index, example_id in enumerate(schedule.example_ids):
                count = int(np.sum(valid[route][layer, example_index]))
                if count > pool_size:
                    raise ValueError("random program count exceeds route pool")
                row_seed = int(canonical_hash({
                    "seed": int(seed),
                    "example_id": example_id,
                    "layer": layer,
                    "route": route,
                })[:16], 16)
                rng = np.random.default_rng(row_seed)
                ids[route][layer, example_index, :count] = rng.choice(
                    pool_size, size=count, replace=False).astype(np.int32)
    records = _records_from_arrays(
        example_ids=schedule.example_ids, prompt_side="random_id",
        program_mass=schedule.program_mass, ids=ids, valid=valid,
        shape=shape,
        extra=[{"random_seed": int(seed)} for _ in schedule.example_ids])
    result = OperatorProgramSchedule(
        example_ids=schedule.example_ids,
        prompt_side="random_id",
        program_mass=schedule.program_mass,
        ids=ids,
        valid=valid,
        records=records,
        schedule_hash=_schedule_hash(
            schedule.example_ids, "random_id", schedule.program_mass,
            ids, valid),
    )
    return result.validate(shape)


def compactness_metrics(
        schedule: OperatorProgramSchedule, *, shape: OperatorSpaceShape,
        paired_schedule: OperatorProgramSchedule | None = None,
        mismatched_schedule: OperatorProgramSchedule | None = None,
) -> dict[str, Any]:
    fractions = np.asarray([
        float(record["site_fraction_of_total_operator_sites"])
        for record in schedule.records], dtype=np.float64)
    per_route = {}
    for route in PROGRAM_ROUTES:
        counts = np.asarray(schedule.valid[route]).sum(axis=(0, 2))
        denominator = int(shape.n_layers) * _pool_size(shape, route)
        values = counts.astype(np.float64) / max(denominator, 1)
        per_route[route] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
        }
    canonical_order = sorted(
        range(schedule.batch_size), key=lambda index: schedule.example_ids[index])
    union: set[tuple[int, str, int]] = set()
    union_curve = []
    total_sites = _total_operator_sites(shape)
    for example_index in canonical_order:
        for route in PROGRAM_ROUTES:
            route_ids = schedule.ids[route][:, example_index]
            route_valid = schedule.valid[route][:, example_index]
            for layer in range(int(shape.n_layers)):
                union.update(
                    (layer, route, int(operator_id))
                    for operator_id in route_ids[layer, route_valid[layer]])
        union_curve.append({
            "example_count": len(union_curve) + 1,
            "union_fraction": len(union) / max(total_sites, 1),
        })

    def overlaps(other: OperatorProgramSchedule | None):
        if other is None:
            return None
        if other.batch_size != schedule.batch_size:
            raise ValueError("program overlap schedules are not aligned")
        output = {}
        for route in PROGRAM_ROUTES:
            values = []
            for example_index in range(schedule.batch_size):
                left = {
                    (layer, int(operator_id))
                    for layer in range(int(shape.n_layers))
                    for operator_id in schedule.ids[route][
                        layer, example_index,
                        schedule.valid[route][layer, example_index]]
                }
                right = {
                    (layer, int(operator_id))
                    for layer in range(int(shape.n_layers))
                    for operator_id in other.ids[route][
                        layer, example_index,
                        other.valid[route][layer, example_index]]
                }
                union_ids = left | right
                values.append(
                    len(left & right) / len(union_ids) if union_ids else 1.0)
            output[route] = float(np.mean(values))
        return output

    return {
        "site_fraction": fractions.tolist(),
        "median_site_fraction": float(np.median(fractions)),
        "mean_site_fraction": float(fractions.mean()),
        "per_route_site_fraction": per_route,
        "union_fraction_vs_example_count": union_curve,
        "same_pair_route_overlap": overlaps(paired_schedule),
        "mismatched_route_overlap": overlaps(mismatched_schedule),
    }


def select_validation_program(
        candidates: Sequence[Mapping[str, Any]], *,
        config: ProtocolConfig) -> dict[str, Any]:
    """Freeze the smallest preregistered passing mass without test access."""
    allowed = tuple(float(value) for value in config.program_mass_candidates)
    ordered = sorted(candidates, key=lambda row: float(row["program_mass"]))
    if tuple(float(row["program_mass"]) for row in ordered) != allowed:
        raise ValueError("validation program candidates do not match protocol")
    passing = []
    for row in ordered:
        checks = {
            "replay_faithfulness_ci": float(
                row["replay"]["faithfulness_ci"]["ci_low"])
                >= config.program_replay_faithfulness_min,
            "replay_answer_agreement": float(
                row["replay"]["answer_agreement_with_full"])
                >= config.program_replay_agreement_min,
            "own_ablation_margin_drop_ci": float(
                row["ablation"]["own_program"]["margin_drop_ci"]["ci_low"])
                > 0.0,
            "paired_over_mismatch_ci": float(
                row["transplant"]["paired_vs_mismatch"]["effect_ci"]["ci_low"])
                > 0.0,
            "compactness": float(
                row["compactness"]["median_site_fraction"])
                <= config.program_compact_fraction_max,
        }
        row["validation_selection_checks"] = checks
        if all(checks.values()):
            passing.append(row)
    if not passing:
        return {
            "status": "no_compact_validation_program",
            "selected_program_mass": None,
            "selection_phase": "validation",
            "test_consulted": False,
            "candidate_count": len(ordered),
        }
    selected = passing[0]
    return {
        "status": "selected",
        "selected_program_mass": float(selected["program_mass"]),
        "selection_phase": "validation",
        "test_consulted": False,
        "selection_rule": "smallest_mass_passing_all_preregistered_gates",
        "program_algorithm_version": PROGRAM_ALGORITHM_VERSION,
        "selected_validation_metrics_hash": canonical_hash({
            "program_mass": selected["program_mass"],
            "checks": selected["validation_selection_checks"],
            "replay": selected["replay"],
            "ablation": selected["ablation"],
            "transplant": selected["transplant"],
            "compactness": selected["compactness"],
        }),
    }


def evaluate_native_program_claims(
        test_result: Mapping[str, Any], *, config: ProtocolConfig) -> dict[str, Any]:
    compact = (
        float(test_result["compactness"]["median_site_fraction"])
        <= config.program_compact_fraction_max)
    sufficiency = (
        float(test_result["replay"]["faithfulness_ci"]["ci_low"])
        >= config.program_replay_faithfulness_min
        and float(test_result["replay"]["answer_agreement_with_full"])
        >= config.program_replay_agreement_min)
    necessity = (
        float(test_result["ablation"]["own_program"][
            "margin_drop_ci"]["ci_low"]) > 0.0
        and float(test_result["ablation"]["own_program"][
            "permutation"]["p_value_two_sided"]) < config.alpha)
    transfer = (
        float(test_result["transplant"]["paired_vs_mismatch"][
            "effect_ci"]["ci_low"]) > 0.0
        and float(test_result["transplant"]["paired_vs_mismatch"][
            "permutation"]["p_value_two_sided"]) < config.alpha
        and float(test_result["transplant"][
            "bidirectional_answer_flip_fraction"])
        >= config.program_transplant_flip_min)
    passed = {
        "descriptive_program": True,
        "compact_dynamic_sufficiency": compact and sufficiency,
        "causal_dynamic_program": compact and sufficiency and necessity,
        "counterfactual_program_transplant": (
            compact and sufficiency and necessity and transfer),
    }
    strongest = None
    for claim in NATIVE_PROGRAM_CLAIM_LADDER:
        if passed[claim]:
            strongest = claim
        else:
            break
    return {
        "status": "ready",
        "passed": bool(passed["counterfactual_program_transplant"]),
        "claims": {
            claim: {"passed": bool(passed[claim])}
            for claim in NATIVE_PROGRAM_CLAIM_LADDER
        },
        "strongest_supported_claim": strongest,
        "checkpoint_specific": True,
        "scientific_claims_primary_modified": False,
    }


def write_program_schedule_artifact(
        store: Any, relative_path: str, schedule: OperatorProgramSchedule, *,
        shape: OperatorSpaceShape, protocol: Mapping[str, Any]) -> dict[str, Any] | None:
    """Persist raw schedules outside item JSON and bind them to the protocol."""
    schedule.validate(shape)
    if not bool(store.is_primary):
        return None
    path = store.path(*relative_path.replace("\\", "/").split("/"))
    counts = np.stack([
        np.asarray(schedule.valid[route]).sum(axis=-1).T
        for route in PROGRAM_ROUTES], axis=-1).astype(np.int32)
    captured_mass = np.full(counts.shape, np.nan, dtype=np.float64)
    for example_index, record in enumerate(schedule.records):
        mass_by_layer = record.get("per_layer_route_captured_mass") or {}
        for layer in range(int(shape.n_layers)):
            route_values = mass_by_layer.get(str(layer)) or {}
            for route_index, route in enumerate(PROGRAM_ROUTES):
                value = route_values.get(route)
                if value is not None:
                    captured_mass[example_index, layer, route_index] = float(
                        value)
    site_counts = np.asarray(
        [record["site_count"] for record in schedule.records],
        dtype=np.int32)
    site_fractions = np.asarray([
        record["site_fraction_of_total_operator_sites"]
        for record in schedule.records], dtype=np.float64)
    arrays: dict[str, Any] = {
        "artifact_schema_version": np.asarray(
            PROGRAM_ARTIFACT_SCHEMA_VERSION, dtype=np.int32),
        "program_algorithm_version": np.asarray(PROGRAM_ALGORITHM_VERSION),
        "protocol_hash": np.asarray(canonical_hash(protocol)),
        "schedule_hash": np.asarray(schedule.schedule_hash),
        "prompt_side": np.asarray(schedule.prompt_side),
        "program_mass": np.asarray(schedule.program_mass, dtype=np.float64),
        "example_ids": np.asarray(schedule.example_ids),
        "program_hashes": np.asarray([
            record["program_hash"] for record in schedule.records]),
        "site_count": site_counts,
        "site_fraction": site_fractions,
        "per_layer_route_site_count": counts,
        "per_layer_route_captured_mass": captured_mass,
    }
    for route in PROGRAM_ROUTES:
        arrays[f"selected_ids_{route}"] = np.asarray(
            schedule.ids[route], dtype=np.int32)
        arrays[f"selected_valid_{route}"] = np.asarray(
            schedule.valid[route], dtype=np.bool_)
    write_npz_atomic(path, **arrays)
    digest = sha256_path(path)
    return {
        "path": path,
        "sha256": digest,
        "artifact_schema_version": PROGRAM_ARTIFACT_SCHEMA_VERSION,
        "program_algorithm_version": PROGRAM_ALGORITHM_VERSION,
        "schedule_hash": schedule.schedule_hash,
        "program_mass": schedule.program_mass,
        "prompt_side": schedule.prompt_side,
        "example_count": schedule.batch_size,
        "widths": schedule.widths,
        "raw_ids_embedded_in_item_json": False,
    }


def load_program_schedule_artifact(
        path: str, *, expected_sha256: str, shape: OperatorSpaceShape,
        protocol: Mapping[str, Any]) -> OperatorProgramSchedule:
    """Load a schedule only when binary, protocol, and checkpoint agree."""
    if sha256_path(path) != str(expected_sha256):
        raise ValueError("operator program binary artifact hash mismatch")
    arrays = read_npz(path)
    if int(np.asarray(arrays["artifact_schema_version"]).item()) != (
            PROGRAM_ARTIFACT_SCHEMA_VERSION):
        raise ValueError("operator program artifact schema mismatch")
    if str(np.asarray(arrays["program_algorithm_version"]).item()) != (
            PROGRAM_ALGORITHM_VERSION):
        raise ValueError("operator program construction algorithm mismatch")
    if str(np.asarray(arrays["protocol_hash"]).item()) != canonical_hash(protocol):
        raise ValueError(
            "operator program protocol/config/checkpoint mismatch")
    example_ids = tuple(str(value) for value in arrays["example_ids"].tolist())
    prompt_side = str(np.asarray(arrays["prompt_side"]).item())
    program_mass = float(np.asarray(arrays["program_mass"]).item())
    ids = {route: np.asarray(arrays[f"selected_ids_{route}"], dtype=np.int32)
           for route in PROGRAM_ROUTES}
    valid = {
        route: np.asarray(arrays[f"selected_valid_{route}"], dtype=np.bool_)
        for route in PROGRAM_ROUTES
    }
    records = _records_from_arrays(
        example_ids=example_ids, prompt_side=prompt_side,
        program_mass=program_mass, ids=ids, valid=valid, shape=shape)
    schedule = OperatorProgramSchedule(
        example_ids=example_ids,
        prompt_side=prompt_side,
        program_mass=program_mass,
        ids=ids,
        valid=valid,
        records=records,
        schedule_hash=str(np.asarray(arrays["schedule_hash"]).item()),
    )
    return schedule.validate(shape)
