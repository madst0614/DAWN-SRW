"""Production-backed sparse capture and discovery-only candidate ranking."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis.dawn_analysis_common import materialize_global_tree
from analysis.dawn_analysis_trace import topk_trace_forward
from analysis.operator_interpretability.benchmark_schema import (
    BenchmarkExample,
    canonical_hash,
)
from analysis.operator_interpretability.statistics import spearman_rank
from analysis.operator_interpretability.units import OperatorSite, RankedSite


ROUTES = ("q", "k", "v", "rst")


def _pad_examples(examples: Sequence[BenchmarkExample], *, pad_token_id: int,
                  multiple: int) -> tuple[np.ndarray, np.ndarray, int]:
    if not examples:
        raise ValueError("capture batch is empty")
    length = max(len(example.input_ids_base) for example in examples)
    batch_size = ((len(examples) + multiple - 1) // multiple) * multiple
    input_ids = np.full((batch_size, length), int(pad_token_id), dtype=np.int32)
    positions = np.zeros((batch_size,), dtype=np.int32)
    for index, example in enumerate(examples):
        ids = np.asarray(example.input_ids_base, dtype=np.int32)
        input_ids[index, :len(ids)] = ids
        positions[index] = int(example.trace_position_base)
    for index in range(len(examples), batch_size):
        input_ids[index] = input_ids[0]
        positions[index] = positions[0]
    return input_ids, positions, len(examples)


def _capture_tiers(initial: tuple[int, int, int], maxima: tuple[int, int, int]):
    current = tuple(min(int(value), int(limit)) for value, limit in zip(initial, maxima))
    yield current
    while current != maxima:
        current = tuple(
            min(limit, max(value + 1, value * 2))
            for value, limit in zip(current, maxima))
        yield current


def _capture_operator_paths(
        ctx: Any, examples: Sequence[BenchmarkExample], *,
        pad_token_id: int, topk_qk: int, topk_v: int, topk_rst: int,
        max_topk_qk: int, max_topk_v: int, max_topk_rst: int,
        capture_threshold: float, seed: int,
        required_phase: str, rank_candidates: bool,
        max_examples: int | None = None) -> dict[str, Any]:
    selected = list(examples[:max_examples] if max_examples else examples)
    if not selected or any(
            example.phase != required_phase for example in selected):
        raise ValueError(
            f"capture requires only phase={required_phase!r} examples")
    if rank_candidates and required_phase != "discovery":
        raise ValueError("operator candidates may only be ranked on discovery")
    data_replicas = max(1, int(ctx.mesh.shape["data"]))
    input_np, position_np, real_count = _pad_examples(
        selected, pad_token_id=pad_token_id, multiple=data_replicas)
    input_ids = jax.device_put(
        jnp.asarray(input_np), NamedSharding(ctx.mesh, P("data", None)))
    positions = jax.device_put(
        jnp.asarray(position_np), NamedSharding(ctx.mesh, P("data")))
    n_qk = int(ctx.model_cfg["n_qk"])
    n_v = int(ctx.model_cfg["n_v"])
    n_rst = int(ctx.model_cfg.get("n_rst", ctx.model_cfg.get("n_know")))
    requested_initial = (int(topk_qk), int(topk_v), int(topk_rst))
    maxima = (
        min(n_qk, int(max_topk_qk)),
        min(n_v, int(max_topk_v)),
        min(n_rst, int(max_topk_rst)),
    )
    initial = tuple(
        min(value, limit) for value, limit in zip(requested_initial, maxima))
    compiled: dict[tuple[int, int, int], Any] = {}

    def executable(widths: tuple[int, int, int]):
        if widths not in compiled:
            qk_width, v_width, rst_width = widths

            @jax.jit
            def trace_step(params, ids, target):
                return topk_trace_forward(
                    params, ctx.model_cfg, ids,
                    topk_qk=qk_width, topk_v=v_width, topk_rst=rst_width,
                    target_positions=target,
                    production_srw_fns=ctx.sharded_fns,
                )

            compiled[widths] = trace_step
        return compiled[widths]

    trace = None
    final_widths = initial
    retries = 0
    capture_history: list[dict[str, Any]] = []
    for widths in _capture_tiers(initial, maxima):
        candidate = materialize_global_tree(executable(widths)(
            ctx.params, input_ids, positions))
        qualified = {}
        capture_row = {"widths": list(widths), "routes": {}}
        for route in ROUTES:
            mass = np.asarray(candidate[f"{route}_captured_mass"])[
                :, :real_count]
            route_ok = bool(np.all(mass >= float(capture_threshold)))
            qualified[route] = route_ok
            capture_row["routes"][route] = {
                "minimum": float(np.min(mass)),
                "mean": float(np.mean(mass)),
                "qualified": route_ok,
            }
        capture_history.append(capture_row)
        trace = candidate
        final_widths = widths
        if all(qualified.values()):
            break
        retries += 1
    if trace is None:
        raise RuntimeError("capture produced no trace")

    aggregate: dict[OperatorSite, list[float]] = defaultdict(list)
    split_aggregate = (
        defaultdict(list), defaultdict(list))
    qualified_denominator: dict[tuple[str, int], int] = defaultdict(int)
    split_denominator = (defaultdict(int), defaultdict(int))
    qualified_rows = 0
    total_rows = 0
    attrition: dict[str, dict[int, int]] = {
        route: defaultdict(int) for route in ROUTES}
    row_capture: list[dict[str, Any]] = []
    split_assignment = [
        int(canonical_hash({
            "seed": int(seed),
            "benchmark_id": example.benchmark_id,
            "example_id": example.example_id,
        })[:16], 16) % 2
        for example in selected
    ]
    for route in ROUTES:
        ids = np.asarray(trace[f"{route}_top_idx"])[:, :real_count]
        contribution_norms = np.asarray(
            trace[f"{route}_top_val"], dtype=np.float64
        )[:, :real_count]
        captured = np.asarray(
            trace[f"{route}_captured_mass"], dtype=np.float64
        )[:, :real_count]
        for layer in range(ids.shape[0]):
            for example_index in range(real_count):
                total_rows += 1
                mass = float(captured[layer, example_index])
                is_qualified = mass >= float(capture_threshold)
                row_capture.append({
                    "benchmark_id": selected[example_index].benchmark_id,
                    "example_id": selected[example_index].example_id,
                    "pair_group_id": selected[example_index].metadata.get(
                        "pair_group_id", selected[example_index].example_id),
                    "phase": selected[example_index].phase,
                    "causal_variable": selected[example_index].causal_variable,
                    "pair_type": selected[example_index].pair_type,
                    "discovery_split": split_assignment[example_index],
                    "layer": layer,
                    "route": route,
                    "captured_mass": mass,
                    "qualified": is_qualified,
                    "operator_ids": [
                        int(value) for value in ids[layer, example_index]],
                    "weights": [
                        float(value) for value in
                        contribution_norms[layer, example_index]],
                })
                if not is_qualified:
                    attrition[route][layer] += 1
                    continue
                qualified_rows += 1
                qualified_denominator[(route, layer)] += 1
                split_index = split_assignment[example_index]
                split_denominator[split_index][(route, layer)] += 1
                for operator_id, contribution_norm in zip(
                        ids[layer, example_index],
                        contribution_norms[layer, example_index]):
                    site = OperatorSite(
                        layer=layer, route=route,
                        operator_id=int(operator_id))
                    value = float(contribution_norm)
                    aggregate[site].append(value)
                    split_aggregate[split_index][site].append(value)

    ranked = [
        RankedSite(
            site=site,
            importance=float(
                np.sum(values)
                / qualified_denominator[(site.route, site.layer)]),
            discovery_count=len(values),
            captured_mass_mean=float(np.mean([
                row["captured_mass"] for row in row_capture
                if row["layer"] == site.layer and row["route"] == site.route
                and row["qualified"]
            ])),
        )
        for site, values in aggregate.items()
    ]
    ranked.sort(key=lambda row: (-row.importance, row.site))
    common_sites = sorted(set(split_aggregate[0]) | set(split_aggregate[1]))
    rank_stability = spearman_rank(
        [
            np.sum(split_aggregate[0][site])
            / max(split_denominator[0][(site.route, site.layer)], 1)
            for site in common_sites
        ],
        [
            np.sum(split_aggregate[1][site])
            / max(split_denominator[1][(site.route, site.layer)], 1)
            for site in common_sites
        ],
    )
    if not ranked:
        raise RuntimeError("no captured-mass-qualified discovery candidates")
    return {
        "status": "ready",
        "ranked_sites": [
            {
                "layer": row.site.layer,
                "route": row.site.route,
                "operator_id": row.site.operator_id,
                "importance": row.importance,
                "discovery_count": row.discovery_count,
                "captured_mass_mean": row.captured_mass_mean,
            }
            for row in ranked
        ],
        "candidate_count": len(ranked),
        "qualified_row_count": qualified_rows,
        "total_row_count": total_rows,
        "qualified_fraction": qualified_rows / max(total_rows, 1),
        "rank_stability": rank_stability,
        "rank_stability_common_site_count": len(common_sites),
        "rank_stability_split_rule": "seeded_hash_of_benchmark_and_example_id",
        "rank_stability_split_example_counts": [
            split_assignment.count(0), split_assignment.count(1)],
        "capture_threshold": float(capture_threshold),
        "capture_mass_definition": (
            "sum_pre_cancellation_production_precision_operator_vector_norms"),
        "topk_selection_metric": "absolute_contribution_mass",
        "requested_initial_widths": list(requested_initial),
        "initial_widths": list(initial),
        "final_widths": list(final_widths),
        "retry_count": retries,
        "capture_history": capture_history,
        "attrition_by_route_layer": {
            route: {str(layer): count for layer, count in layers.items()}
            for route, layers in attrition.items()
        },
        "rows": row_capture,
        "phase": required_phase,
        "ranking_eligible": bool(rank_candidates),
    }


def capture_discovery_candidates(
        ctx: Any, examples: Sequence[BenchmarkExample], *,
        pad_token_id: int, topk_qk: int, topk_v: int, topk_rst: int,
        max_topk_qk: int, max_topk_v: int, max_topk_rst: int,
        capture_threshold: float, seed: int,
        max_examples: int | None = None) -> dict[str, Any]:
    """Rank operator sites from discovery traces only."""
    return _capture_operator_paths(
        ctx, examples,
        pad_token_id=pad_token_id,
        topk_qk=topk_qk, topk_v=topk_v, topk_rst=topk_rst,
        max_topk_qk=max_topk_qk,
        max_topk_v=max_topk_v,
        max_topk_rst=max_topk_rst,
        capture_threshold=capture_threshold,
        seed=seed,
        required_phase="discovery",
        rank_candidates=True,
        max_examples=max_examples,
    )


def capture_held_out_paths(
        ctx: Any, examples: Sequence[BenchmarkExample], *, phase: str,
        pad_token_id: int, topk_qk: int, topk_v: int, topk_rst: int,
        max_topk_qk: int, max_topk_v: int, max_topk_rst: int,
        capture_threshold: float, seed: int,
        max_examples: int | None = None) -> dict[str, Any]:
    """Capture frozen sparse paths without allowing held-out candidate ranking."""
    if phase not in {"validation", "test"}:
        raise ValueError("held-out path capture requires validation or test")
    result = _capture_operator_paths(
        ctx, examples,
        pad_token_id=pad_token_id,
        topk_qk=topk_qk, topk_v=topk_v, topk_rst=topk_rst,
        max_topk_qk=max_topk_qk,
        max_topk_v=max_topk_v,
        max_topk_rst=max_topk_rst,
        capture_threshold=capture_threshold,
        seed=seed,
        required_phase=phase,
        rank_candidates=False,
        max_examples=max_examples,
    )
    result.pop("ranked_sites", None)
    result.pop("candidate_count", None)
    result.pop("rank_stability", None)
    result.pop("rank_stability_common_site_count", None)
    return result


def ranked_site_objects(result: Mapping[str, Any]) -> list[RankedSite]:
    if result.get("status") != "ready":
        raise ValueError("capture result is not ready")
    return [
        RankedSite(
            site=OperatorSite(
                int(row["layer"]), str(row["route"]),
                int(row["operator_id"])),
            importance=float(row["importance"]),
            discovery_count=int(row["discovery_count"]),
            captured_mass_mean=float(row["captured_mass_mean"]),
        )
        for row in result["ranked_sites"]
    ]
