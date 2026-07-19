"""CPU semantic contracts for the IOI paired operator trajectory."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import re

import jax
import jax.numpy as jnp
import numpy as np

import analysis.operator_interpretability.paired_trajectory as trajectory_module
import analysis.operator_interpretability.runner as runner_module
from analysis.operator_interpretability.benchmark_schema import BenchmarkExample
from analysis.operator_interpretability.benchmarks.mib_ioi import (
    _render_semantic_prompt,
)
from analysis.operator_interpretability.paired_trajectory import (
    _batched_intervention_rows,
    _expand_trajectory_widths,
    _operator_group_capture_rows,
    _operator_group_patch_rows,
    _path_intervention_rows,
    _summarize_operator_group_direction,
    _validate_complete_trace,
    build_divergence_atlas,
    deduplicate_residual_candidates,
    evaluate_cumulative_path,
    freeze_chronological_path,
    ioi_semantic_record,
    merge_staged_coarse_patch_results,
    summarize_frozen_path_uncertainty,
)
from analysis.operator_interpretability.protocol import ProtocolConfig
from analysis.operator_interpretability.runner import (
    OperatorInterpretabilityRunner,
)
from models.dawn_srw_v4171 import (
    TRAJECTORY_PATCH_STAGES,
    _analysis_apply_trajectory_patch,
    make_sharded_srw_paired_trajectory_minimal,
    make_sharded_srw_trajectory_minimal,
)
from test_operator_interpretability_cpu import (
    _assert_exact,
    _build_mesh_and_kernels,
    _make_model_harness,
)


ROUTES = ("q", "k", "v", "rst")


class _Encoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class _TwoSubtokenNameTokenizer:
    """Small offset tokenizer whose person names occupy two token slots."""

    def __init__(self, names):
        self.names = set(names)
        self._vocabulary = {}

    def _pieces(self, text):
        output = []
        for match in re.finditer(r"\w+|[^\w\s]", text):
            token = match.group(0)
            start, end = match.span()
            if token in self.names:
                middle = start + (end - start) // 2
                output.extend(((text[start:middle], start, middle),
                               (text[middle:end], middle, end)))
            else:
                output.append((token, start, end))
        return output

    def __call__(self, text, *, add_special_tokens=False,
                 return_offsets_mapping=False):
        assert add_special_tokens is False
        pieces = self._pieces(str(text))
        ids = []
        for token, _, _ in pieces:
            if token not in self._vocabulary:
                self._vocabulary[token] = len(self._vocabulary) + 1
            ids.append(self._vocabulary[token])
        result = _Encoding(input_ids=ids)
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (start, end) for _, start, end in pieces]
        return result


def _semantic_example():
    subject = "Caroline"
    indirect_object = "Marianna"
    template = (
        "When {name_A} and {name_B} went to the {place}, "
        "{name_C} gave the {object} to")
    metadata = {
        "subject": subject,
        "indirect_object": indirect_object,
        "place": "market",
        "object": "letter",
        "template": template,
    }
    base_prompt, base_spans = _render_semantic_prompt(
        template, metadata, s2_name=subject)
    source_prompt, source_spans = _render_semantic_prompt(
        template, metadata, s2_name=indirect_object)
    tokenizer = _TwoSubtokenNameTokenizer((subject, indirect_object))
    base_ids = tuple(tokenizer(base_prompt, add_special_tokens=False).input_ids)
    source_ids = tuple(
        tokenizer(source_prompt, add_special_tokens=False).input_ids)
    metadata["semantic_char_spans"] = {
        "base": {
            "first_name_a": base_spans["name_A"],
            "first_name_b": base_spans["name_B"],
            "s2_counterfactual": base_spans["name_C"],
        },
        "source": {
            "first_name_a": source_spans["name_A"],
            "first_name_b": source_spans["name_B"],
            "s2_counterfactual": source_spans["name_C"],
        },
    }
    example = BenchmarkExample(
        benchmark_id="mib_ioi",
        example_id="trajectory-semantic-0",
        phase="discovery",
        base_prompt=base_prompt,
        source_prompt=source_prompt,
        positive_answer=" Marianna",
        negative_answer=" Caroline",
        intervention_positive_answer=" Caroline",
        intervention_negative_answer=" Marianna",
        causal_variable="ioi_task_output",
        pair_type="s2_io_flip_counterfactual",
        source_behavior_required=True,
        trace_position_base=len(base_ids) - 1,
        trace_position_source=len(source_ids) - 1,
        input_ids_base=base_ids,
        input_ids_source=source_ids,
        positive_ids=(29,),
        negative_ids=(30,),
        source_positive_ids=(30,),
        source_negative_ids=(29,),
        intervention_positive_ids=(30,),
        intervention_negative_ids=(29,),
        metadata=metadata,
    )
    return example, tokenizer


def test_s2_prefix_identity_and_multisubtoken_alignment() -> None:
    example, tokenizer = _semantic_example()
    record = ioi_semantic_record(example, tokenizer)
    assert record.s2_end - record.s2_start == 2
    assert record.position_roles[:2] == ("s2", "s2_subtoken_1")
    assert example.input_ids_base[:record.s2_start] == (
        example.input_ids_source[:record.s2_start])
    assert example.input_ids_base[record.s2_start:record.s2_end] != (
        example.input_ids_source[record.s2_start:record.s2_end])
    assert example.input_ids_base[record.s2_end:] == (
        example.input_ids_source[record.s2_end:])
    assert record.positions == tuple(
        range(record.s2_start, record.answer_position + 1))
    print("TRAJECTORY_S2_SEMANTIC_ALIGNMENT_OK")


def _make_trajectory_harness():
    (mesh, production_single, suppression_single,
     production_paired, suppression_paired) = _build_mesh_and_kernels()
    harness = _make_model_harness(
        mesh, production_single, suppression_single,
        production_paired, suppression_paired)
    trajectory_single = make_sharded_srw_trajectory_minimal(
        mesh, max_chunk_size=2, trajectory_capture_width=harness.dim)
    trajectory_paired = make_sharded_srw_paired_trajectory_minimal(
        mesh, max_chunk_size=2, trajectory_capture_width=harness.dim)
    sharded_fns = dict(harness.kwargs["sharded_fns"])
    sharded_fns.update({
        "attn_v_single_minimal": (
            trajectory_single._v4171_production_wrapper),
        "rst_single_minimal": (
            trajectory_single._v4171_production_wrapper),
        "attn_qk_paired_minimal": (
            trajectory_paired._v4171_production_wrapper),
        "attn_v_single_trajectory_minimal": trajectory_single,
        "rst_single_trajectory_minimal": trajectory_single,
        "attn_qk_paired_trajectory_minimal": trajectory_paired,
        "_v4171_kernel_profile": "trajectory",
    })
    kwargs = dict(harness.kwargs)
    kwargs["sharded_fns"] = sharded_fns
    positions = jnp.broadcast_to(
        jnp.arange(harness.seq, dtype=jnp.int32)[None, :],
        (harness.batch, harness.seq))
    position_valid = jnp.ones_like(positions, dtype=jnp.bool_)
    route_shape = (
        harness.layers, harness.batch, harness.seq, harness.dim)
    empty_ids = {
        route: jnp.zeros(route_shape, dtype=jnp.int32) for route in ROUTES}
    empty_valid = {
        route: jnp.zeros(route_shape, dtype=jnp.bool_) for route in ROUTES}
    patch_shape = (harness.batch, 2)
    no_patch = {
        "layers": jnp.zeros(patch_shape, dtype=jnp.int32),
        "positions": jnp.zeros(patch_shape, dtype=jnp.int32),
        "stages": jnp.zeros(patch_shape, dtype=jnp.int32),
        "enabled": jnp.zeros(patch_shape, dtype=jnp.bool_),
        "values": jnp.zeros(
            patch_shape + (harness.dim,), dtype=jnp.float32),
    }

    @jax.jit
    def trajectory(
            params, input_ids, ids_q, ids_k, ids_v, ids_rst,
            valid_q, valid_k, valid_v, valid_rst, replay,
            patch_layers, patch_positions, patch_stages,
            patch_enabled, patch_values):
        return harness.model.apply(
            params, input_ids,
            trajectory_positions=positions,
            trajectory_position_valid=position_valid,
            selected_ids_q=ids_q,
            selected_ids_k=ids_k,
            selected_ids_v=ids_v,
            selected_ids_rst=ids_rst,
            selected_valid_q=valid_q,
            selected_valid_k=valid_k,
            selected_valid_v=valid_v,
            selected_valid_rst=valid_rst,
            replay_full_active=replay,
            patch_layers=patch_layers,
            patch_positions=patch_positions,
            patch_stages=patch_stages,
            patch_enabled=patch_enabled,
            patch_values=patch_values,
            labels=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            return_residual=True,
            method=(
                harness.model.analysis_forward_with_paired_operator_trajectory),
            **kwargs)

    @jax.jit
    def production(params, input_ids):
        return harness.model.apply(
            params, input_ids, labels=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            minimal_train=True, analysis_return_residual=True, **kwargs)

    aligned_base = jnp.asarray([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ], dtype=jnp.int32)
    aligned_source = jnp.asarray([
        [1, 2, 13, 4, 5],
        [6, 7, 18, 9, 10],
    ], dtype=jnp.int32)
    return SimpleNamespace(
        **vars(harness), trajectory=trajectory, production=production,
        positions=positions, position_valid=position_valid,
        empty_ids=empty_ids, empty_valid=empty_valid, no_patch=no_patch,
        aligned_base=aligned_base, aligned_source=aligned_source)


def _run_trajectory(harness, input_ids, *, ids=None, valid=None,
                    replay=False, patch=None):
    ids = harness.empty_ids if ids is None else ids
    valid = harness.empty_valid if valid is None else valid
    patch = harness.no_patch if patch is None else patch
    return harness.trajectory(
        harness.variables, input_ids,
        ids["q"], ids["k"], ids["v"], ids["rst"],
        valid["q"], valid["k"], valid["v"], valid["rst"],
        jnp.asarray(replay, dtype=jnp.bool_),
        patch["layers"], patch["positions"], patch["stages"],
        patch["enabled"], patch["values"])


def _outcome(result):
    return {
        "per_token_ce": result["per_token_ce"],
        "final_residual": result["final_residual"],
    }


def _patch(harness, *, layers, positions, stages, enabled, values):
    def rows(values_, dtype):
        return jnp.asarray(
            [values_ for _ in range(harness.batch)], dtype=dtype)

    patch_values = np.asarray(values, dtype=np.float32)
    if patch_values.ndim == 2:
        patch_values = np.broadcast_to(
            patch_values[None, :, :],
            (harness.batch,) + patch_values.shape).copy()
    return {
        "layers": rows(layers, jnp.int32),
        "positions": rows(positions, jnp.int32),
        "stages": rows(stages, jnp.int32),
        "enabled": rows(enabled, jnp.bool_),
        "values": jnp.asarray(patch_values, dtype=jnp.float32),
    }


def test_full_active_capture_and_replay(harness):
    capture = _run_trajectory(harness, harness.aligned_base)
    production = harness.production(harness.variables, harness.aligned_base)
    _assert_exact(
        _outcome(production), _outcome(capture),
        "trajectory capture production parity")
    trace = capture["operator_trajectory_trace"]
    summary = _validate_complete_trace(trace, real_count=harness.batch)
    assert all(value["omitted_active_count"] == 0
               for value in summary.values())
    assert all(value["captured_active_count"] >= 0
               for value in summary.values())
    selected = {
        route: trace["routes"][route]["operator_id"] for route in ROUTES}
    selected_valid = {
        route: trace["routes"][route]["operator_valid"] for route in ROUTES}
    replay = _run_trajectory(
        harness, harness.aligned_base, ids=selected,
        valid=selected_valid, replay=True)
    replay_trace = replay["operator_trajectory_trace"]
    for route in ROUTES:
        np.testing.assert_allclose(
            np.asarray(replay_trace["routes"][route]["production_output"]),
            np.asarray(
                replay_trace["routes"][route]["selected_replay_output"]),
            rtol=5.0e-4, atol=5.0e-4)
    np.testing.assert_allclose(
        np.asarray(capture["final_residual"]),
        np.asarray(replay["final_residual"]), rtol=5.0e-4, atol=5.0e-4)
    np.testing.assert_allclose(
        np.asarray(capture["per_token_ce"]),
        np.asarray(replay["per_token_ce"]), rtol=5.0e-4, atol=5.0e-4)
    print("TRAJECTORY_FULL_ACTIVE_REPLAY_OK")
    return capture


def test_prefix_state_identity(harness):
    paired = jnp.stack(
        (harness.aligned_base[0], harness.aligned_source[0]), axis=0)
    result = _run_trajectory(harness, paired)
    states = result["operator_trajectory_trace"]["states"]
    for stage in ("residual_input", "post_attention", "post_rst"):
        np.testing.assert_array_equal(
            np.asarray(states[stage])[:, 0, :2],
            np.asarray(states[stage])[:, 1, :2])
    print("TRAJECTORY_PREFIX_STATE_IDENTITY_OK")


def test_self_route_residual_and_disabled_patches(harness, capture):
    target_layer = 0
    target_position = 2
    slot = target_position
    debug = capture["parity_debug"]
    for route in ROUTES:
        self_value = np.asarray(debug[route])[target_layer, :, slot, :]
        values = np.zeros((harness.batch, 2, harness.dim), dtype=np.float32)
        values[:, 0, :] = self_value
        patch = _patch(
            harness, layers=(target_layer, 0),
            positions=(target_position, 0),
            stages=(TRAJECTORY_PATCH_STAGES[route], 0),
            enabled=(True, False), values=values)
        result = _run_trajectory(harness, harness.aligned_base, patch=patch)
        _assert_exact(
            _outcome(capture), _outcome(result),
            f"self {route} trajectory patch")

    residual_value = np.asarray(
        capture["operator_trajectory_trace"]["states"]["residual_input"]
    )[target_layer, :, slot, :]
    residual_values = np.zeros(
        (harness.batch, 2, harness.dim), dtype=np.float32)
    residual_values[:, 0, :] = residual_value
    residual_patch = _patch(
        harness, layers=(target_layer, 0),
        positions=(target_position, 0),
        stages=(TRAJECTORY_PATCH_STAGES["residual_input"], 0),
        enabled=(True, False), values=residual_values)
    residual_result = _run_trajectory(
        harness, harness.aligned_base, patch=residual_patch)
    _assert_exact(
        _outcome(capture), _outcome(residual_result),
        "self residual trajectory patch")

    arbitrary = np.full(
        (harness.batch, 2, harness.dim), 123.5, dtype=np.float32)
    disabled = _patch(
        harness, layers=(0, 1), positions=(2, 3),
        stages=(TRAJECTORY_PATCH_STAGES["q"],
                TRAJECTORY_PATCH_STAGES["post_rst"]),
        enabled=(False, False), values=arbitrary)
    disabled_result = _run_trajectory(
        harness, harness.aligned_base, patch=disabled)
    _assert_exact(
        _outcome(capture), _outcome(disabled_result),
        "disabled trajectory patch")
    print("TRAJECTORY_SELF_AND_DISABLED_PATCH_OK")


def test_multisite_layer_order_and_sibling_route_isolation(harness, capture):
    debug = capture["parity_debug"]
    layer0_q = np.asarray(debug["q"])[0, :, 2, :] + 0.25
    layer1_rst = np.asarray(debug["rst"])[1, :, 3, :] - 0.25
    values_a = np.zeros((harness.batch, 2, harness.dim), dtype=np.float32)
    values_a[:, 0, :] = layer1_rst
    values_a[:, 1, :] = layer0_q
    values_b = values_a[:, ::-1, :].copy()
    schedule_a = _patch(
        harness, layers=(1, 0), positions=(3, 2),
        stages=(TRAJECTORY_PATCH_STAGES["rst"],
                TRAJECTORY_PATCH_STAGES["q"]),
        enabled=(True, True), values=values_a)
    schedule_b = _patch(
        harness, layers=(0, 1), positions=(2, 3),
        stages=(TRAJECTORY_PATCH_STAGES["q"],
                TRAJECTORY_PATCH_STAGES["rst"]),
        enabled=(True, True), values=values_b)
    result_a = _run_trajectory(
        harness, harness.aligned_base, patch=schedule_a)
    result_b = _run_trajectory(
        harness, harness.aligned_base, patch=schedule_b)
    _assert_exact(
        _outcome(result_a), _outcome(result_b),
        "trajectory layer order independent of schedule slot order")
    assert not np.array_equal(
        np.asarray(capture["final_residual"]),
        np.asarray(result_a["final_residual"]))

    direct_siblings = {
        "q": ("k", "v"),
        "k": ("q", "v"),
        "v": ("q", "k"),
        "rst": ("q", "k", "v"),
    }
    for route, siblings in direct_siblings.items():
        values = np.zeros((harness.batch, 2, harness.dim), dtype=np.float32)
        values[:, 0, :] = np.asarray(debug[route])[0, :, 2, :] + 0.5
        patch = _patch(
            harness, layers=(0, 0), positions=(2, 0),
            stages=(TRAJECTORY_PATCH_STAGES[route], 0),
            enabled=(True, False), values=values)
        result = _run_trajectory(harness, harness.aligned_base, patch=patch)
        assert not np.array_equal(
            np.asarray(debug[route])[0, :, 2, :],
            np.asarray(result["parity_debug"][route])[0, :, 2, :])
        for sibling in siblings:
            np.testing.assert_array_equal(
                np.asarray(debug[sibling])[0],
                np.asarray(result["parity_debug"][sibling])[0])
    print("TRAJECTORY_LAYER_ORDER_AND_SIBLING_ISOLATION_OK")


def test_padded_duplicate_row_isolation(harness):
    real = harness.aligned_base[0]
    mixed = jnp.stack((real, harness.aligned_base[1]), axis=0)
    padded = jnp.stack((real, real), axis=0)
    mixed_result = _run_trajectory(harness, mixed)
    padded_result = _run_trajectory(harness, padded)
    np.testing.assert_array_equal(
        np.asarray(mixed_result["final_residual"])[0],
        np.asarray(padded_result["final_residual"])[0])
    np.testing.assert_array_equal(
        np.asarray(mixed_result["per_token_ce"])[0],
        np.asarray(padded_result["per_token_ce"])[0])
    for route in ROUTES:
        for field in ("production_output", "operator_id", "operator_valid"):
            np.testing.assert_array_equal(
                np.asarray(mixed_result["operator_trajectory_trace"]
                           ["routes"][route][field])[:, 0],
                np.asarray(padded_result["operator_trajectory_trace"]
                           ["routes"][route][field])[:, 0])
    print("TRAJECTORY_PADDED_DUPLICATE_ISOLATION_OK")


def test_width_growth_preserves_unaffected_pools() -> None:
    widths = {"qk": 2, "v": 4, "rst": 6}
    expanded, changed = _expand_trajectory_widths(
        widths, {"qk": 16, "v": 16, "rst": 16}, {"qk": 5})
    assert expanded == {"qk": 5, "v": 4, "rst": 6}
    assert tuple(changed) == ("qk",)
    assert changed["qk"] == {
        "previous_width": 2,
        "observed_active_count": 5,
        "updated_width": 5,
    }
    inherited, second_changed = _expand_trajectory_widths(
        expanded, {"qk": 16, "v": 16, "rst": 16}, {"v": 7})
    assert inherited == {"qk": 5, "v": 8, "rst": 6}
    assert tuple(second_changed) == ("v",)
    print("TRAJECTORY_WIDTH_GROWTH_AND_INHERITANCE_OK")


def test_operator_support_set_decomposition() -> None:
    example, tokenizer = _semantic_example()
    source_record = ioi_semantic_record(example, tokenizer)
    position = source_record.s2_start
    record = replace(
        source_record, s2_end=position + 1, answer_position=position,
        positions=(position,), position_roles=("s2",))
    ids = np.full((1, 4, 1, 3), -1, dtype=np.int32)
    valid = np.zeros_like(ids, dtype=np.bool_)
    ids[0, 0:2, 0, :2] = np.asarray([0, 1], dtype=np.int32)
    ids[0, 2:4, 0, :2] = np.asarray([1, 2], dtype=np.int32)
    valid[..., :2] = True
    route_template = {
        "operator_id": ids,
        "operator_valid": valid,
        "prewrite_amplitude_bf16_bits": np.zeros_like(ids, dtype=np.uint16),
        "read_scalar_bf16_bits": np.zeros_like(ids, dtype=np.uint16),
        "execution_weight": np.ones_like(ids, dtype=np.float32),
        "query": np.ones((1, 4, 1, 2), dtype=np.float32),
        "tau": np.ones((1, 4, 1, 1), dtype=np.float32),
        "denominator": np.ones((1, 4, 1, 1), dtype=np.float32),
        "scale": np.ones((1,), dtype=np.float32),
        "production_output": np.zeros((1, 4, 1, 2), dtype=np.float32),
    }
    trace = {
        "states": {
            stage: np.zeros((1, 4, 1, 2), dtype=np.float32)
            for stage in ("residual_input", "post_attention", "post_rst")
        },
        "routes": {
            route: {name: np.array(value, copy=True)
                    for name, value in route_template.items()}
            for route in ROUTES
        },
    }
    atlas = build_divergence_atlas(
        {"capture_trace": trace}, (example,),
        {example.example_id: record},
        operator_keys={
            "qk": np.eye(3, 2, dtype=np.float32),
            "v": np.eye(3, 2, dtype=np.float32),
            "rst": np.eye(3, 2, dtype=np.float32),
        },
        epsilon=1.0e-8)
    row = atlas["_per_example_sites"][example.example_id][
        "route:0:s2:q"]
    np.testing.assert_array_equal(row["common_ids"], np.asarray([1]))
    np.testing.assert_array_equal(row["base_only_ids"], np.asarray([0]))
    np.testing.assert_array_equal(row["source_only_ids"], np.asarray([2]))
    print("TRAJECTORY_OPERATOR_SUPPORT_SET_DECOMPOSITION_OK")


def test_batched_patch_matches_scalar_reference() -> None:
    value = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    patch_layers = np.zeros((2, 2), dtype=np.int32)
    patch_positions = np.asarray([[1, 2], [0, 2]], dtype=np.int32)
    patch_stages = np.asarray([
        [TRAJECTORY_PATCH_STAGES["q"], TRAJECTORY_PATCH_STAGES["k"]],
        [TRAJECTORY_PATCH_STAGES["q"], TRAJECTORY_PATCH_STAGES["q"]],
    ], dtype=np.int32)
    patch_enabled = np.ones((2, 2), dtype=np.bool_)
    patch_values = np.asarray([
        [[101.0, 102.0], [103.0, 104.0]],
        [[201.0, 202.0], [203.0, 204.0]],
    ], dtype=np.float32)
    batched = _analysis_apply_trajectory_patch(
        jnp.asarray(value), layer_index=0,
        stage_code=TRAJECTORY_PATCH_STAGES["q"],
        patch_layers=jnp.asarray(patch_layers),
        patch_positions=jnp.asarray(patch_positions),
        patch_stages=jnp.asarray(patch_stages),
        patch_enabled=jnp.asarray(patch_enabled),
        patch_values=jnp.asarray(patch_values))
    scalar = value.copy()
    for batch_index in range(2):
        for slot in range(2):
            if (patch_enabled[batch_index, slot]
                    and patch_layers[batch_index, slot] == 0
                    and patch_stages[batch_index, slot]
                    == TRAJECTORY_PATCH_STAGES["q"]):
                scalar[batch_index, patch_positions[batch_index, slot]] = (
                    patch_values[batch_index, slot])
    np.testing.assert_array_equal(np.asarray(batched), scalar)
    print("TRAJECTORY_BATCHED_PATCH_SCALAR_PARITY_OK")


def test_margin_orientation_donor_group_and_path_gate() -> None:
    first, tokenizer = _semantic_example()
    first_record = ioi_semantic_record(first, tokenizer)
    second = replace(
        first, example_id="trajectory-semantic-1",
        positive_ids=(25,), negative_ids=(26,),
        source_positive_ids=(26,), source_negative_ids=(25,),
        intervention_positive_ids=(26,), intervention_negative_ids=(25,))
    second_record = replace(
        first_record, example_id=second.example_id)
    examples = (first, second)
    semantic = {
        first.example_id: first_record,
        second.example_id: second_record,
    }
    task = {
        "paired_values": np.ones((2, 8), dtype=np.float32),
        "mismatch_values": np.full((2, 8), 2.0, dtype=np.float32),
        "self_values": np.full((2, 8), 3.0, dtype=np.float32),
        "positions": np.asarray(
            [first_record.s2_start, second_record.s2_start], np.int32),
        "layer": 0,
        "stage": TRAJECTORY_PATCH_STAGES["q"],
    }
    source_to_base = _batched_intervention_rows(
        (first,), direction="source_to_base", sequence_length=(
            len(first.input_ids_source) + 1), pad_token_id=0,
        data_multiple=1, patch_slots=2, d_model=8,
        tasks=({**task,
                "paired_values": task["paired_values"][:1],
                "mismatch_values": task["mismatch_values"][:1],
                "self_values": task["self_values"][:1],
                "positions": task["positions"][:1]},))
    base_to_source = _batched_intervention_rows(
        (first,), direction="base_to_source", sequence_length=(
            len(first.input_ids_base) + 1), pad_token_id=0,
        data_multiple=1, patch_slots=2, d_model=8,
        tasks=({**task,
                "paired_values": task["paired_values"][:1],
                "mismatch_values": task["mismatch_values"][:1],
                "self_values": task["self_values"][:1],
                "positions": task["positions"][:1]},))
    prompt_length = len(first.input_ids_base)
    assert source_to_base["labels"][0, prompt_length] == first.positive_ids[0]
    assert source_to_base["labels"][1, prompt_length] == first.negative_ids[0]
    assert base_to_source["labels"][0, prompt_length] == first.negative_ids[0]
    assert base_to_source["labels"][1, prompt_length] == first.positive_ids[0]

    groups = (np.asarray([1, 3], np.int32), np.asarray([6], np.int32))
    mismatch = np.asarray([1, 0], dtype=np.int32)
    site = {"semantic_role": "s2"}
    capture_rows = _operator_group_capture_rows(
        examples, semantic, direction="source_to_base", site=site,
        groups=groups, mismatch_indices=mismatch, group_width=2,
        pad_token_id=0, data_multiple=1)
    np.testing.assert_array_equal(
        capture_rows["selected_ids"][1], np.asarray([1, 3]))
    np.testing.assert_array_equal(
        capture_rows["selected_ids"][2], np.asarray([6, -1]))
    np.testing.assert_array_equal(
        capture_rows["selected_ids"][5], np.asarray([6, -1]))
    np.testing.assert_array_equal(
        capture_rows["selected_ids"][6], np.asarray([1, 3]))
    patch_rows = _operator_group_patch_rows(
        examples, semantic, direction="source_to_base", site=site,
        groups=groups, mismatch_indices=mismatch,
        contributions=np.zeros((2, 4, 8), dtype=np.float32),
        group_width=2,
        sequence_length=max(len(first.input_ids_source),
                            len(second.input_ids_source)) + 1,
        pad_token_id=0, data_multiple=1)
    np.testing.assert_array_equal(
        patch_rows["selected_ids"][2], np.asarray([1, 3]))
    np.testing.assert_array_equal(
        patch_rows["selected_ids"][4], np.asarray([6, -1]))
    np.testing.assert_array_equal(
        patch_rows["selected_ids"][12], np.asarray([6, -1]))
    np.testing.assert_array_equal(
        patch_rows["selected_ids"][14], np.asarray([1, 3]))

    def direction(paired, specific, *, self_passed=True):
        return {
            "paired_margin_shift_mean": paired,
            "paired_minus_mismatched_effect_mean": specific,
            "self_reconstruction_passed": True,
            "disabled_noop_passed": True,
        } if self_passed else {
            "paired_margin_shift_mean": paired,
            "paired_minus_mismatched_effect_mean": specific,
            "self_reconstruction_passed": False,
            "disabled_noop_passed": True,
        }
    summaries = [
        {
            "candidate_index": 0, "patch_kind": "route",
            "bidirectional_paired_effect_mean": 0.75,
            "bidirectional_specific_effect_mean": 0.75,
            "directions": {
                "source_to_base": direction(2.0, 2.0),
                "base_to_source": direction(-0.5, -0.5),
            },
        },
        {
            "candidate_index": 1, "patch_kind": "route",
            "bidirectional_paired_effect_mean": 0.5,
            "bidirectional_specific_effect_mean": 0.2,
            "directions": {
                "source_to_base": direction(0.6, 0.3),
                "base_to_source": direction(0.4, 0.1),
            },
        },
        {
            "candidate_index": 2, "patch_kind": "route",
            "bidirectional_paired_effect_mean": 0.8,
            "bidirectional_specific_effect_mean": 0.4,
            "directions": {
                "source_to_base": direction(0.8, 0.4),
                "base_to_source": direction(
                    0.8, 0.4, self_passed=False),
            },
        },
    ]
    candidates = [
        {"layer": 0, "semantic_role": "s2", "route": "q"},
        {"layer": 1, "semantic_role": "answer_position", "route": "v"},
        {"layer": 0, "semantic_role": "post_s2", "route": "rst"},
    ]
    frozen = freeze_chronological_path(
        {"site_summaries": summaries}, candidates,
        config=ProtocolConfig())
    assert [site["candidate_index"] for site in frozen["sites"]] == [1]
    assert frozen["per_direction_gate_required"] is True

    no_path = freeze_chronological_path(
        {"site_summaries": summaries[:1]}, candidates[:1],
        config=ProtocolConfig())
    assert no_path["status"] == "no_causal_path"
    assert no_path["path_length"] == 0
    no_path_result = evaluate_cumulative_path(
        None, (), no_path, {}, {}, production_atlas={}, pad_token_id=0,
        config=ProtocolConfig(), phase="validation",
        evaluate_prefix_curve=False)
    assert no_path_result["status"] == "no_causal_path"
    assert no_path_result["validation_path_evaluated"] is False
    assert no_path_result["causal_path_supported"] is False

    group_summary, group_vectors = _summarize_operator_group_direction(
        np.asarray([
            [1.0, 0.4, 0.1, 0.0, -0.2],
            [1.2, 0.5, 0.2, 0.0, -0.1],
        ], dtype=np.float64),
        np.zeros((2, 5, 3), dtype=np.float64),
        before=np.zeros((2,), dtype=np.float64), self_atol=1.0e-6)
    assert group_summary["mismatched_margin_shift_mean"] == (
        group_summary[
            "donor_own_group_mismatched_program_margin_shift_mean"])
    np.testing.assert_array_equal(
        group_vectors["mismatched"],
        group_vectors["donor_own_group_mismatched_program"])
    direction_vectors = {
        direction: {
            "paired_effect": np.full((16,), 1.0, dtype=np.float64),
            "mismatched_effect": np.full((16,), 0.2, dtype=np.float64),
        }
        for direction in ("source_to_base", "base_to_source")
    }
    uncertainty = summarize_frozen_path_uncertainty(
        {"path_length": 1, "directions": direction_vectors},
        config=ProtocolConfig(), seed=4172)
    assert uncertainty["paired_effect_ci"]["ci_low"] > 0.0
    assert uncertainty[
        "paired_minus_mismatched_effect_ci"]["ci_low"] > 0.0
    assert uncertainty["paired_effect_paired_permutation"][
        "p_value_two_sided"] < ProtocolConfig().alpha
    assert uncertainty["paired_minus_mismatched_paired_permutation"][
        "p_value_two_sided"] < ProtocolConfig().alpha
    assert all(
        uncertainty["per_direction"][direction][
            "paired_and_specific_validation_passed"]
        for direction in ("source_to_base", "base_to_source"))
    assert uncertainty[
        "direction_averaged_causal_pair_specific_validation_passed"] is True
    assert uncertainty["validation_bidirectional_path_supported"] is True
    assert uncertainty["causal_pair_specific_validation_passed"] is True

    asymmetric_vectors = {
        "source_to_base": {
            "paired_effect": np.full((16,), 2.0, dtype=np.float64),
            "mismatched_effect": np.full((16,), 0.2, dtype=np.float64),
        },
        "base_to_source": {
            "paired_effect": np.full((16,), -0.2, dtype=np.float64),
            "mismatched_effect": np.full((16,), -0.05, dtype=np.float64),
        },
    }
    asymmetric = summarize_frozen_path_uncertainty(
        {"path_length": 1, "directions": asymmetric_vectors},
        config=ProtocolConfig(), seed=4173)
    assert asymmetric[
        "direction_averaged_causal_pair_specific_validation_passed"] is True
    assert asymmetric["direction_averaged"][
        "paired_and_specific_validation_passed"] is True
    assert asymmetric["per_direction"]["source_to_base"][
        "paired_and_specific_validation_passed"] is True
    assert asymmetric["per_direction"]["base_to_source"][
        "paired_and_specific_validation_passed"] is False
    assert asymmetric["validation_bidirectional_path_supported"] is False
    assert asymmetric["causal_pair_specific_validation_passed"] is False
    print("TRAJECTORY_ORIENTATION_DONOR_GROUP_AND_PATH_GATE_OK")


def _passing_direction() -> dict[str, object]:
    return {
        "paired_margin_shift_mean": 1.0,
        "paired_minus_mismatched_effect_mean": 0.5,
        "self_reconstruction_passed": True,
        "disabled_noop_passed": True,
    }


def _passing_patch_summary(
        candidate_index: int, patch_kind: str, route: str
) -> dict[str, object]:
    return {
        "candidate_index": candidate_index,
        "layer": 0,
        "semantic_role": "s2",
        "route": route,
        "patch_kind": patch_kind,
        "directions": {
            "source_to_base": _passing_direction(),
            "base_to_source": _passing_direction(),
        },
        "bidirectional_paired_effect_mean": 1.0,
        "bidirectional_specific_effect_mean": 0.5,
    }


def test_native_stage_order_and_staged_residual_deduplication() -> None:
    assert [TRAJECTORY_PATCH_STAGES[route] for route in ROUTES] == [0, 1, 2, 3]
    candidates = [
        {"candidate_index": 0, "layer": 0,
         "semantic_role": "s2", "route": "rst"},
        {"candidate_index": 1, "layer": 0,
         "semantic_role": "s2", "route": "v"},
        {"candidate_index": 2, "layer": 0,
         "semantic_role": "s2", "route": "q"},
        {"candidate_index": 3, "layer": 0,
         "semantic_role": "s2", "route": "k"},
    ]
    route_summaries = [
        _passing_patch_summary(index, "route", candidate["route"])
        for index, candidate in enumerate(candidates)
    ]
    frozen = freeze_chronological_path(
        {"site_summaries": route_summaries}, candidates,
        config=ProtocolConfig())
    assert [site["route"] for site in frozen["sites"]] == list(ROUTES)

    residual_candidates = deduplicate_residual_candidates(candidates)
    assert len(residual_candidates) == 1
    assert residual_candidates[0]["candidate_index"] == 0
    route_result = {
        "status": "ready", "phase": "discovery",
        "patch_kinds_evaluated": ["route"],
        "candidate_count": 4, "evaluated_patch_count": 4,
        "forward_call_count": 4,
        "effective_intervention_batch_size": 4,
        "resource_retry_count": 0, "resource_retries": [],
        "mismatch_mapping_hash": "mapping",
        "site_summaries": route_summaries,
        "_vectors": [
            {"candidate_index": index, "patch_kind": "route"}
            for index in range(4)
        ],
    }
    residual_summary = _passing_patch_summary(0, "residual", "rst")
    residual_result = {
        "status": "ready", "phase": "discovery",
        "patch_kinds_evaluated": ["residual"],
        "candidate_count": 1, "evaluated_patch_count": 1,
        "forward_call_count": 1,
        "effective_intervention_batch_size": 4,
        "resource_retry_count": 0, "resource_retries": [],
        "mismatch_mapping_hash": "mapping",
        "site_summaries": [residual_summary],
        "_vectors": [{"candidate_index": 0, "patch_kind": "residual"}],
    }
    merged = merge_staged_coarse_patch_results(
        route_result, residual_result)
    assert merged["evaluated_patch_count"] == 5
    assert merged["positive_sites"]["route_candidate_indices"] == [0, 1, 2, 3]
    assert merged["positive_sites"]["residual_layer_roles"] == [{
        "representative_candidate_index": 0,
        "layer": 0,
        "semantic_role": "s2",
    }]
    assert merged["positive_sites"]["operator_followup_basis"] == (
        "route_positive_only")
    print("TRAJECTORY_NATIVE_STAGE_ORDER_AND_RESIDUAL_DEDUP_OK")


def test_prefix_fusion_and_oom_fallback() -> None:
    first, tokenizer = _semantic_example()
    record = ioi_semantic_record(first, tokenizer)
    second = replace(
        first, example_id="trajectory-prefix-1",
        positive_ids=(25,), negative_ids=(26,),
        source_positive_ids=(26,), source_negative_ids=(25,))
    examples = (first, second)
    sites = [
        {"candidate_index": 0, "layer": 0,
         "semantic_role": "s2", "route": "q"},
        {"candidate_index": 1, "layer": 0,
         "semantic_role": "answer_position", "route": "v"},
    ]
    positions = np.asarray([
        [record.s2_start, record.answer_position],
        [record.s2_start, record.answer_position],
    ], dtype=np.int32)
    site_values = {
        "sequence_length": max(
            len(example.input_ids_source) + 1 for example in examples),
        "base": {
            "position": positions,
            "route": np.arange(32, dtype=np.float32).reshape(2, 2, 8),
        },
        "source": {
            "position": positions,
            "route": np.arange(32, 64, dtype=np.float32).reshape(2, 2, 8),
        },
    }
    row_kwargs = {
        "examples": examples,
        "direction": "source_to_base",
        "global_indices": np.asarray([0, 1], dtype=np.int32),
        "sequence_length": site_values["sequence_length"],
        "pad_token_id": 0,
        "data_multiple": 1,
        "patch_slots": 4,
        "d_model": 8,
        "site_values": site_values,
        "donor_indices": np.asarray([1, 0], dtype=np.int32),
    }
    fused = _path_intervention_rows(
        **row_kwargs, path_prefixes=((sites[0],), tuple(sites)))
    scalar_first = _path_intervention_rows(
        **row_kwargs, path_prefixes=((sites[0],),))
    scalar_second = _path_intervention_rows(
        **row_kwargs, path_prefixes=(tuple(sites),))
    scalar_count = int(scalar_first["real_count"])
    assert fused["real_count"] == 2 * scalar_count
    for name in (
            "input_ids", "labels", "target_positions", "patch_layers",
            "patch_positions", "patch_stages", "patch_enabled",
            "patch_values"):
        np.testing.assert_array_equal(
            fused[name][:scalar_count], scalar_first[name][:scalar_count])
        np.testing.assert_array_equal(
            fused[name][scalar_count:2 * scalar_count],
            scalar_second[name][:scalar_count])

    original = trajectory_module._evaluate_cumulative_path_once
    attempted = []

    def fake_once(*args, prefix_batch_size, **kwargs):
        attempted.append(prefix_batch_size)
        if prefix_batch_size > 1:
            raise MemoryError("synthetic prefix batch pressure")
        return {
            "status": "ready", "phase": "discovery",
            "path_record_hash": "path", "forward_call_count": 1,
            "prefixes": [], "_vectors": [],
        }

    trajectory_module._evaluate_cumulative_path_once = fake_once
    try:
        fallback = evaluate_cumulative_path(
            None, (),
            {"sites": [dict(sites[0]) for _ in range(4)],
             "path_record_hash": "path"},
            {}, {}, production_atlas={}, pad_token_id=0,
            config=ProtocolConfig(trajectory_path_prefix_batch_size=4),
            phase="discovery", evaluate_prefix_curve=True)
    finally:
        trajectory_module._evaluate_cumulative_path_once = original
    assert attempted == [4, 2, 1]
    assert fallback["initial_prefix_batch_size"] == 4
    assert fallback["effective_prefix_batch_size"] == 1
    assert fallback["resource_retry_count"] == 2
    assert all(
        not retry["scientific_path_changed"]
        for retry in fallback["resource_retries"])
    print("TRAJECTORY_PREFIX_FUSION_AND_OOM_FALLBACK_OK")


def test_trajectory_request_and_test_accessor_isolation() -> None:
    mixed = object.__new__(OperatorInterpretabilityRunner)
    try:
        OperatorInterpretabilityRunner.run(mixed, (
            "mib_ioi.paired_operator_trajectory",
            "mib_ioi.native_operator_program",
        ))
    except ValueError as exc:
        assert "held-out test access remains isolated" in str(exc)
    else:
        raise AssertionError("mixed trajectory request was not rejected")

    isolated = object.__new__(OperatorInterpretabilityRunner)
    isolated._paired_trajectory_test_isolated = True
    isolated.config = ProtocolConfig(minimum_known_correct=1)
    isolated.ctx = SimpleNamespace()
    isolated.tokenizer = SimpleNamespace(pad_token_id=0)
    isolated._scope = lambda kind: ("mib_ioi",)
    accessed_phases = []

    def load_phase(_benchmark_id, phase):
        accessed_phases.append(phase)
        return [SimpleNamespace(example_id=f"{phase}-0")]

    isolated._load_phase_examples = load_phase
    original = runner_module.evaluate_behavior
    runner_module.evaluate_behavior = lambda *args, **kwargs: {
        "known_correct": [True]}
    try:
        result = OperatorInterpretabilityRunner._run_behavioral_eligibility(
            isolated)
    finally:
        runner_module.evaluate_behavior = original
    assert accessed_phases == ["discovery", "validation"]
    test_phase = result["benchmarks"]["mib_ioi"]["phases"]["test"]
    assert test_phase["test_evaluated"] is False
    assert test_phase["test_data_accessor_called"] is False
    print("TRAJECTORY_REQUEST_AND_TEST_ACCESSOR_ISOLATION_OK")


def main() -> None:
    test_s2_prefix_identity_and_multisubtoken_alignment()
    harness = _make_trajectory_harness()
    capture = test_full_active_capture_and_replay(harness)
    test_prefix_state_identity(harness)
    test_self_route_residual_and_disabled_patches(harness, capture)
    test_multisite_layer_order_and_sibling_route_isolation(harness, capture)
    test_padded_duplicate_row_isolation(harness)
    test_width_growth_preserves_unaffected_pools()
    test_operator_support_set_decomposition()
    test_batched_patch_matches_scalar_reference()
    test_margin_orientation_donor_group_and_path_gate()
    test_native_stage_order_and_staged_residual_deduplication()
    test_prefix_fusion_and_oom_fallback()
    test_trajectory_request_and_test_accessor_isolation()
    print("IOI_PAIRED_OPERATOR_TRAJECTORY_CPU_OK")


if __name__ == "__main__":
    main()
