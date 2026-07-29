"""Production-backed sparse capture and discovery-only candidate ranking."""

from __future__ import annotations

import hashlib
import json
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
    RAVEL_SOURCE_COLUMNS,
    RAVEL_VARIABLES,
    canonical_hash,
)
from analysis.operator_interpretability.statistics import spearman_rank
from analysis.operator_interpretability.units import OperatorSite, RankedSite


ROUTES = ("q", "k", "v", "rst")
DISCOVERY_OVERLAP_PREFIX_COUNTS = (
    32, 64, 128, 256, 512, 1024, 2048, 4096,
)


def _digest_block(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    digest.update(payload)


def _pad_examples(examples: Sequence[BenchmarkExample], *, pad_token_id: int,
                  multiple: int, prompt_side: str = "base") -> tuple[
                      np.ndarray, np.ndarray, int]:
    if not examples:
        raise ValueError("capture batch is empty")
    if prompt_side not in {"base", "source"}:
        raise ValueError("capture prompt_side must be base or source")
    prompts = [
        example.input_ids_base if prompt_side == "base"
        else example.input_ids_source
        for example in examples
    ]
    length = max(len(prompt) for prompt in prompts)
    batch_size = ((len(examples) + multiple - 1) // multiple) * multiple
    input_ids = np.full((batch_size, length), int(pad_token_id), dtype=np.int32)
    positions = np.zeros((batch_size,), dtype=np.int32)
    for index, (example, prompt) in enumerate(zip(examples, prompts)):
        ids = np.asarray(prompt, dtype=np.int32)
        input_ids[index, :len(ids)] = ids
        positions[index] = int(
            example.trace_position_base if prompt_side == "base"
            else example.trace_position_source)
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


def _discovery_split_assignments(
        examples: Sequence[BenchmarkExample], *, seed: int) -> tuple[
            list[int], str, dict[str, dict[str, list[int]]], list[int]]:
    benchmark_ids = {str(example.benchmark_id) for example in examples}
    if len(benchmark_ids) != 1:
        raise ValueError("capture split requires exactly one benchmark")
    benchmark_id = next(iter(benchmark_ids))
    if benchmark_id != "ravel":
        assignments = [
            int(canonical_hash({
                "seed": int(seed),
                "benchmark_id": example.benchmark_id,
                "example_id": example.example_id,
            })[:16], 16) % 2
            for example in examples
        ]
        return (
            assignments,
            "seeded_hash_of_benchmark_and_example_id",
            {},
            [assignments.count(0), assignments.count(1)],
        )

    group_strata: dict[str, tuple[str, str]] = {}
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for example in examples:
        group_id = str(example.metadata.get("pair_group_id") or "")
        source_column = str(example.metadata.get(
            "official_counterfactual_column") or "")
        variable = str(example.causal_variable)
        if not group_id:
            raise ValueError("RAVEL discovery split requires pair_group_id")
        if variable not in RAVEL_VARIABLES:
            raise ValueError(
                f"RAVEL discovery split has invalid variable={variable!r}")
        if source_column not in RAVEL_SOURCE_COLUMNS:
            raise ValueError(
                "RAVEL discovery split has invalid official source column")
        stratum = (variable, source_column)
        previous = group_strata.setdefault(group_id, stratum)
        if previous != stratum:
            raise ValueError(
                "one RAVEL pair_group_id spans multiple localization strata")
    for group_id, stratum in group_strata.items():
        grouped[stratum].append(group_id)

    assignment_by_group: dict[str, int] = {}
    group_totals = [0, 0]
    stratum_counts: dict[str, dict[str, list[int]]] = {
        variable: {
            source_column: [0, 0]
            for source_column in RAVEL_SOURCE_COLUMNS
        }
        for variable in RAVEL_VARIABLES
    }
    for stratum in (
            (variable, source_column)
            for variable in RAVEL_VARIABLES
            for source_column in RAVEL_SOURCE_COLUMNS):
        group_ids = sorted(
            grouped.get(stratum, ()),
            key=lambda group_id: (
                canonical_hash({
                    "seed": int(seed),
                    "causal_variable": stratum[0],
                    "official_counterfactual_column": stratum[1],
                    "pair_group_id": group_id,
                }),
                group_id,
            ))
        if group_totals[0] < group_totals[1]:
            first_split = 0
        elif group_totals[1] < group_totals[0]:
            first_split = 1
        else:
            first_split = int(canonical_hash({
                "seed": int(seed),
                "stratum": list(stratum),
            })[:16], 16) % 2
        counts = stratum_counts[stratum[0]][stratum[1]]
        for index, group_id in enumerate(group_ids):
            split = (first_split + index) % 2
            assignment_by_group[group_id] = split
            counts[split] += 1
            group_totals[split] += 1

    assignments = [
        assignment_by_group[str(example.metadata["pair_group_id"])]
        for example in examples
    ]
    return (
        assignments,
        "seeded_balanced_pair_group_split_within_causal_variable_and_"
        "official_counterfactual_column",
        stratum_counts,
        group_totals,
    )


def _split_rank_stability(
        split_aggregate: tuple[Mapping[OperatorSite, Sequence[float]],
                               Mapping[OperatorSite, Sequence[float]]],
        split_denominator: tuple[Mapping[tuple[str, int], int],
                                 Mapping[tuple[str, int], int]],
) -> tuple[float | None, int]:
    common_sites = sorted(set(split_aggregate[0]) | set(split_aggregate[1]))
    stability = spearman_rank(
        [
            np.sum(split_aggregate[0].get(site, ()))
            / max(split_denominator[0].get((site.route, site.layer), 0), 1)
            for site in common_sites
        ],
        [
            np.sum(split_aggregate[1].get(site, ()))
            / max(split_denominator[1].get((site.route, site.layer), 0), 1)
            for site in common_sites
        ],
    )
    if stability is not None and not np.isfinite(stability):
        stability = None
    return stability, len(common_sites)


def _split_topk_overlap(
        split_aggregate: tuple[
            Mapping[OperatorSite, Sequence[float]],
            Mapping[OperatorSite, Sequence[float]]],
        split_denominator: tuple[
            Mapping[tuple[str, int], int],
            Mapping[tuple[str, int], int]],
) -> list[dict[str, Any]]:
    rankings = []
    for split in (0, 1):
        scored = [
            (
                abs(float(np.sum(values))) / max(
                    int(split_denominator[split].get(
                        (site.route, site.layer), 0)),
                    1),
                site,
            )
            for site, values in split_aggregate[split].items()
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        rankings.append([site for _, site in scored])
    if not rankings[0] or not rankings[1]:
        return []
    rows = []
    for requested_count in DISCOVERY_OVERLAP_PREFIX_COUNTS:
        count = min(
            int(requested_count), len(rankings[0]), len(rankings[1]))
        if count <= 0:
            continue
        split_sets = (
            set(rankings[0][:count]),
            set(rankings[1][:count]),
        )
        shared = split_sets[0] & split_sets[1]
        union = split_sets[0] | split_sets[1]
        rows.append({
            "requested_prefix_count": int(requested_count),
            "evaluated_prefix_count": int(count),
            "shared_site_count": len(shared),
            "overlap_fraction_of_each_prefix": len(shared) / count,
            "jaccard": len(shared) / len(union),
            "shared_site_count_by_route": {
                route: sum(site.route == route for site in shared)
                for route in ROUTES
            },
            "split_0_site_count_by_route": {
                route: sum(
                    site.route == route for site in split_sets[0])
                for route in ROUTES
            },
            "split_1_site_count_by_route": {
                route: sum(
                    site.route == route for site in split_sets[1])
                for route in ROUTES
            },
        })
    return rows


def _capture_operator_paths(
        ctx: Any, examples: Sequence[BenchmarkExample], *,
        pad_token_id: int, topk_qk: int, topk_v: int, topk_rst: int,
        max_topk_qk: int, max_topk_v: int, max_topk_rst: int,
        capture_threshold: float, seed: int,
        required_phase: str, rank_candidates: bool,
        retain_rows: bool,
        max_examples: int | None = None,
        prompt_side: str = "base",
        require_all_qualified: bool = False,
        bind_prompt_side_in_digest: bool = False) -> dict[str, Any]:
    selected = list(examples[:max_examples] if max_examples else examples)
    if not selected or any(
            example.phase != required_phase for example in selected):
        raise ValueError(
            f"capture requires only phase={required_phase!r} examples")
    if rank_candidates and required_phase != "discovery":
        raise ValueError("operator candidates may only be ranked on discovery")
    data_replicas = max(1, int(ctx.mesh.shape["data"]))
    input_np, position_np, real_count = _pad_examples(
        selected, pad_token_id=pad_token_id, multiple=data_replicas,
        prompt_side=prompt_side)
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
    final_qualified = {route: False for route in ROUTES}
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
        final_qualified = qualified
        if all(qualified.values()):
            break
        retries += 1
    if trace is None:
        raise RuntimeError("capture produced no trace")
    if require_all_qualified and not all(final_qualified.values()):
        failed = ",".join(
            route for route in ROUTES if not final_qualified[route])
        raise RuntimeError(
            "native operator program capture failed closed at maximum width: "
            f"phase={required_phase} prompt_side={prompt_side} "
            f"capture_threshold={capture_threshold} failed_routes={failed} "
            f"final_widths={final_widths}")

    aggregate: dict[OperatorSite, list[float]] = defaultdict(list)
    split_aggregate = (defaultdict(list), defaultdict(list))
    qualified_denominator: dict[tuple[str, int], int] = defaultdict(int)
    qualified_mass_total: dict[tuple[str, int], float] = defaultdict(float)
    split_denominator = (defaultdict(int), defaultdict(int))
    is_ravel = selected[0].benchmark_id == "ravel"
    variable_split_aggregate = {
        variable: (defaultdict(list), defaultdict(list))
        for variable in RAVEL_VARIABLES
    } if is_ravel else {}
    variable_split_denominator = {
        variable: (defaultdict(int), defaultdict(int))
        for variable in RAVEL_VARIABLES
    } if is_ravel else {}
    qualified_rows = 0
    total_rows = 0
    raw_operator_value_count = 0
    raw_capture_digest = hashlib.sha256()
    attrition: dict[str, dict[int, int]] = {
        route: defaultdict(int) for route in ROUTES}
    row_capture: list[dict[str, Any]] = []
    (split_assignment, split_rule, split_stratum_group_counts,
     split_group_counts) = _discovery_split_assignments(
         selected, seed=seed)
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
                operator_ids = np.asarray(
                    ids[layer, example_index], dtype="<i4")
                weights = np.asarray(
                    contribution_norms[layer, example_index], dtype="<f8")
                header = json.dumps({
                    "benchmark_id": selected[example_index].benchmark_id,
                    "example_id": selected[example_index].example_id,
                    "pair_group_id": selected[example_index].metadata.get(
                        "pair_group_id", selected[example_index].example_id),
                    "phase": selected[example_index].phase,
                    "causal_variable": selected[example_index].causal_variable,
                    "pair_type": selected[example_index].pair_type,
                    "official_counterfactual_column": (
                        selected[example_index].metadata.get(
                            "official_counterfactual_column")),
                    "discovery_split": split_assignment[example_index],
                    "layer": layer,
                    "route": route,
                    "captured_mass": mass,
                    "qualified": is_qualified,
                    **({"prompt_side": prompt_side}
                       if bind_prompt_side_in_digest else {}),
                }, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False).encode("utf-8")
                _digest_block(raw_capture_digest, header)
                _digest_block(raw_capture_digest, operator_ids.tobytes())
                _digest_block(raw_capture_digest, weights.tobytes())
                raw_operator_value_count += int(operator_ids.size)
                if retain_rows:
                    row_capture.append({
                        "benchmark_id": selected[example_index].benchmark_id,
                        "example_id": selected[example_index].example_id,
                        "pair_group_id": selected[example_index].metadata.get(
                            "pair_group_id", selected[example_index].example_id),
                        "phase": selected[example_index].phase,
                        "causal_variable": selected[example_index].causal_variable,
                        "pair_type": selected[example_index].pair_type,
                        "official_counterfactual_column": (
                            selected[example_index].metadata.get(
                                "official_counterfactual_column")),
                        "discovery_split": split_assignment[example_index],
                        "layer": layer,
                        "route": route,
                        "captured_mass": mass,
                        "qualified": is_qualified,
                        **({"prompt_side": prompt_side}
                           if bind_prompt_side_in_digest else {}),
                        "operator_ids": operator_ids.tolist(),
                        "weights": weights.tolist(),
                    })
                if not is_qualified:
                    attrition[route][layer] += 1
                    continue
                qualified_rows += 1
                qualified_denominator[(route, layer)] += 1
                qualified_mass_total[(route, layer)] += mass
                split_index = split_assignment[example_index]
                split_denominator[split_index][(route, layer)] += 1
                variable = str(selected[example_index].causal_variable)
                if is_ravel:
                    variable_split_denominator[variable][split_index][
                        (route, layer)] += 1
                for operator_id, contribution_norm in zip(
                        ids[layer, example_index],
                        contribution_norms[layer, example_index]):
                    site = OperatorSite(
                        layer=layer, route=route,
                        operator_id=int(operator_id))
                    value = float(contribution_norm)
                    aggregate[site].append(value)
                    split_aggregate[split_index][site].append(value)
                    if is_ravel:
                        variable_split_aggregate[variable][split_index][
                            site].append(value)

    ranked = [
        RankedSite(
            site=site,
            importance=float(
                np.sum(values)
                / qualified_denominator[(site.route, site.layer)]),
            discovery_count=len(values),
            captured_mass_mean=float(
                qualified_mass_total[(site.route, site.layer)]
                / qualified_denominator[(site.route, site.layer)]),
        )
        for site, values in aggregate.items()
    ]
    ranked.sort(key=lambda row: (-row.importance, row.site))
    pooled_rank_stability, pooled_common_site_count = _split_rank_stability(
        split_aggregate, split_denominator)
    discovery_split_topk_overlap = _split_topk_overlap(
        split_aggregate, split_denominator)
    rank_stability = pooled_rank_stability
    rank_stability_common_site_count = pooled_common_site_count
    rank_stability_by_causal_variable: dict[str, dict[str, Any]] = {}
    rank_stability_min_variable: str | None = None
    if is_ravel:
        for variable in RAVEL_VARIABLES:
            stability, common_site_count = _split_rank_stability(
                variable_split_aggregate[variable],
                variable_split_denominator[variable])
            source_split_counts = split_stratum_group_counts[variable]
            split_counts = [
                sum(source_split_counts[source_column][split]
                    for source_column in RAVEL_SOURCE_COLUMNS)
                for split in (0, 1)]
            all_source_strata_represented = all(
                min(source_split_counts[source_column]) > 0
                for source_column in RAVEL_SOURCE_COLUMNS)
            rank_stability_by_causal_variable[variable] = {
                "status": (
                    "ready" if stability is not None
                    and all_source_strata_represented
                    else "insufficient_stratified_split"),
                "rank_stability": stability,
                "common_site_count": common_site_count,
                "split_independent_group_counts": split_counts,
                "split_source_group_counts": source_split_counts,
            }
        ready_variable_stabilities = {
            variable: float(row["rank_stability"])
            for variable, row in rank_stability_by_causal_variable.items()
            if row["status"] == "ready"
        }
        if len(ready_variable_stabilities) == len(RAVEL_VARIABLES):
            rank_stability_min_variable = min(
                ready_variable_stabilities,
                key=lambda variable: (
                    ready_variable_stabilities[variable], variable))
            rank_stability = ready_variable_stabilities[
                rank_stability_min_variable]
            rank_stability_common_site_count = int(
                rank_stability_by_causal_variable[
                    rank_stability_min_variable]["common_site_count"])
        else:
            rank_stability = None
            rank_stability_common_site_count = 0
    if not ranked:
        raise RuntimeError("no captured-mass-qualified discovery candidates")
    result = {
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
        "rank_stability_common_site_count": rank_stability_common_site_count,
        "rank_stability_split_rule": split_rule,
        "rank_stability_split_example_counts": [
            split_assignment.count(0), split_assignment.count(1)],
        "rank_stability_split_independent_group_counts": split_group_counts,
        "discovery_split_topk_overlap": discovery_split_topk_overlap,
        "discovery_split_topk_overlap_selection_metric": (
            "absolute_split_mean_contribution_importance"),
        "discovery_split_topk_overlap_test_used": False,
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
        "raw_capture_digest": raw_capture_digest.hexdigest(),
        "raw_capture_digest_algorithm": (
            "sha256_len_prefixed_canonical_row_header_le_i32_ids_le_f64_weights"),
        "raw_capture_operator_value_count": raw_operator_value_count,
        "raw_rows_materialized_for_runtime": bool(retain_rows),
        "phase": required_phase,
        "ranking_eligible": bool(rank_candidates),
        **({"prompt_side": prompt_side}
           if bind_prompt_side_in_digest else {}),
    }
    if is_ravel:
        result.update({
            "rank_stability_aggregation": (
                "minimum_across_preregistered_causal_variables"),
            "rank_stability_min_variable": rank_stability_min_variable,
            "rank_stability_by_causal_variable": (
                rank_stability_by_causal_variable),
            "rank_stability_split_strata_group_counts": (
                split_stratum_group_counts),
            "pooled_rank_stability": pooled_rank_stability,
            "pooled_rank_stability_common_site_count": (
                pooled_common_site_count),
            "pooled_rank_stability_used_for_gate": False,
        })
    if retain_rows:
        result["rows"] = row_capture
    return result


def capture_discovery_candidates(
        ctx: Any, examples: Sequence[BenchmarkExample], *,
        pad_token_id: int, topk_qk: int, topk_v: int, topk_rst: int,
        max_topk_qk: int, max_topk_v: int, max_topk_rst: int,
        capture_threshold: float, seed: int,
        retain_rows: bool = True,
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
        retain_rows=retain_rows,
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
        retain_rows=True,
        max_examples=max_examples,
    )
    result.pop("ranked_sites", None)
    result.pop("candidate_count", None)
    for key in tuple(result):
        if (key.startswith("rank_stability")
                or key.startswith("pooled_rank_stability")):
            result.pop(key, None)
    return result


def capture_program_paths(
        ctx: Any, examples: Sequence[BenchmarkExample], *, phase: str,
        prompt_side: str, pad_token_id: int,
        topk_qk: int, topk_v: int, topk_rst: int,
        max_topk_qk: int, max_topk_v: int, max_topk_rst: int,
        capture_threshold: float, seed: int,
        max_examples: int | None = None) -> dict[str, Any]:
    """Capture one native program side with fail-closed adaptive coverage."""
    if phase not in {"discovery", "validation", "test"}:
        raise ValueError("operator program capture has an invalid phase")
    return _capture_operator_paths(
        ctx, examples,
        pad_token_id=pad_token_id,
        topk_qk=topk_qk,
        topk_v=topk_v,
        topk_rst=topk_rst,
        max_topk_qk=max_topk_qk,
        max_topk_v=max_topk_v,
        max_topk_rst=max_topk_rst,
        capture_threshold=capture_threshold,
        seed=seed,
        required_phase=phase,
        rank_candidates=False,
        retain_rows=True,
        max_examples=max_examples,
        prompt_side=prompt_side,
        require_all_qualified=True,
        bind_prompt_side_in_digest=True,
    )


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
