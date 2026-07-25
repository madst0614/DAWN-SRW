import ast
import importlib
import inspect

import jax.numpy as jnp

from scripts import train_jax


def _main_tree():
    return ast.parse(inspect.getsource(train_jax.main))


def _call_name(call):
    return call.func.id if isinstance(call.func, ast.Name) else None


def test_training_loop_calls_only_train_step_model_executable():
    tree = _main_tree()
    epoch_loop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "epoch")
    step_loop = next(
        node for node in ast.walk(epoch_loop)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and any(
            isinstance(element, ast.Name) and element.id == "local_step"
            for element in node.target.elts))
    calls = [
        node for node in ast.walk(step_loop)
        if isinstance(node, ast.Call)]
    direct_calls = [_call_name(call) for call in calls]

    assert direct_calls.count("train_step_fn") == 1
    assert "production_diagnostic_step_fn" not in direct_calls
    assert "analysis_step_fn" not in direct_calls
    assert "geometry_step_fn" not in direct_calls
    assert "eval_step_fn" not in direct_calls
    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "apply"
        for call in calls)

    evaluate_calls = [
        call for call in ast.walk(epoch_loop)
        if isinstance(call, ast.Call) and _call_name(call) == "evaluate"]
    assert len(evaluate_calls) == 2
    assert all(
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "eval_step_fn"
        for call in evaluate_calls)


def test_main_constructs_no_diagnostic_or_analysis_kernels():
    source = inspect.getsource(train_jax.main)
    assert "_sharded_fns_diagnostics" not in source
    assert "_sharded_fns_analysis" not in source
    assert "create_production_diagnostic_step(" not in source
    assert "create_analysis_step(" not in source
    assert "create_geometry_step(" not in source
    assert "kernel_profile='production_diagnostics'" not in source
    assert "kernel_profile=\"production_diagnostics\"" not in source
    assert (
        "_sharded_fns_eval\n"
        "        if _sharded_fns_eval is not None else _sharded_fns"
    ) in source

    build_calls = [
        node for node in ast.walk(_main_tree())
        if isinstance(node, ast.Call)
        and _call_name(node) == "build_canonical_sharded_fns"]
    assert build_calls
    for call in build_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "kernel_profile" not in keywords
        assert not (
            "analysis" in keywords
            and isinstance(keywords["analysis"], ast.Constant)
            and keywords["analysis"].value is True)


def test_v417x_train_model_call_contract_is_explicit():
    source = inspect.getsource(train_jax.create_train_step)
    assert "extra_kw['analysis'] = False" in source
    assert "extra_kw['minimal_train'] = True" in source
    assert "extra_kw['minimal_runtime_profile'] = 'training'" in source
    assert "extra_kw['compute_accuracy'] = _train_compute_accuracy" in source
    assert "extra_kw['collect_train_metrics'] = jnp.asarray(" in source
    assert "static_argnames" not in source
    assert "v417x training requires production kernels" in source


def test_v4174_dynamic_metric_source_contract():
    from models import dawn_srw_v4174 as v4174

    model_source = inspect.getsource(v4174.DAWN_SRW_V4174.__call__)
    output_source = inspect.getsource(v4174._dense_rw_output_sharded)
    den_source = inspect.getsource(
        v4174._global_dense_rw_den_sharded)
    representation_reduce_source = inspect.getsource(
        v4174._psum_dense_rw_representation_sharded)
    metric_scan_source = inspect.getsource(
        v4174._dense_rw_metric_stats_sharded)
    metric_vector_source = inspect.getsource(
        v4174._collect_dense_rw_metric_vector_sharded)
    metric_cond_source = inspect.getsource(v4174._metric_only_cond)
    metric_cond_fwd_source = inspect.getsource(
        v4174._metric_only_cond_fwd)
    metric_cond_bwd_source = inspect.getsource(
        v4174._metric_only_cond_bwd)
    attention_source = inspect.getsource(
        v4174._make_sharded_attention_space_dense)
    rst_source = inspect.getsource(v4174._make_sharded_rst_space_dense)
    attention_minimal_source = inspect.getsource(
        v4174.make_sharded_attention_space_dense_minimal)
    attention_fp32_collective_source = inspect.getsource(
        v4174.make_sharded_attention_space_dense_fp32_collective_reference)
    rst_minimal_source = inspect.getsource(
        v4174.make_sharded_rst_space_dense_minimal)
    rst_fp32_collective_source = inspect.getsource(
        v4174.make_sharded_rst_space_dense_fp32_collective_reference)
    trainer_source = inspect.getsource(train_jax.create_train_step)
    builder_source = inspect.getsource(train_jax.build_canonical_sharded_fns)
    main_source = inspect.getsource(train_jax.main)

    assert "collect_train_metrics=True" in model_source
    assert "collect_regular_metrics" in model_source
    assert "jax.lax.cond(" in model_source
    assert "collect_metrics" not in output_source
    assert "jax.lax.cond(" not in output_source
    assert "raw_out, gate_mass = carry_value" in output_source
    assert "jnp.square" not in output_source
    assert "gate.max" not in output_source
    assert "margin >" not in output_source
    assert "depth.sum" not in output_source
    assert "jax.checkpoint(production_step, prevent_cse=False)" in output_source
    assert "use_chunk_remat" not in output_source
    assert "remat_chunks" not in output_source
    production_step_source = output_source.split(
        "def production_step", 1)[1].split("scan_step =", 1)[0]
    for required in (
            "read_value", "rho", "gate", "valid", "execution_weight"):
        assert required in production_step_source

    assert "jax.lax.cond(" not in den_source
    assert "gate_mass.astype(jnp.float32)" in den_source
    assert '"model"' in den_source
    assert "_shared_composition_den(" in den_source
    assert "return global_gate_mass, gate_den" in den_source
    assert "raw_out" not in den_source
    assert "jnp.bfloat16 if collective_bf16 else jnp.float32" in (
        representation_reduce_source)
    assert "local_output.astype(collective_dtype)" in (
        representation_reduce_source)
    assert '"model").astype(jnp.float32)' in representation_reduce_source
    assert "write_vectors" not in metric_scan_source
    assert "\"amtn,amnr->amtr\"" not in metric_scan_source
    assert "raw_out" not in metric_scan_source
    assert "chunk_out" not in metric_scan_source
    assert metric_scan_source.count("jax.lax.stop_gradient(") >= 4
    assert "gate_mass, gate_sq, gate_max, active_count, depth_sum" in (
        metric_scan_source)
    assert "jnp.square" in metric_scan_source
    assert "gate.max" in metric_scan_source
    assert "margin >" in metric_scan_source
    assert "depth.sum" in metric_scan_source
    assert metric_vector_source.count("jax.lax.psum(") == 2
    assert metric_vector_source.count("jax.lax.pmax(") == 1
    assert "packed_metrics" in metric_vector_source
    assert "jax.nn.one_hot" in metric_vector_source
    assert "return jax.lax.stop_gradient(jnp.stack(metric_values))" in (
        metric_vector_source)
    assert "return jax.lax.cond(" in metric_cond_source
    assert "jnp.zeros((metric_count,), dtype=jnp.float32)" in (
        metric_cond_source)
    assert "None)" in metric_cond_fwd_source
    assert "return None, None" in metric_cond_bwd_source

    for executor_source in (attention_source, rst_source):
        assert "metric_vector = _metric_only_cond(" in executor_source
        assert "len(metric_names)" in executor_source
        assert "collect_metrics" in executor_source
    attention_production_source = attention_source.split(
        "q_per_space =", 1)[0]
    rst_production_source = rst_source.split("n_per_space =", 1)[0]
    assert "grouped_raw_out" in attention_production_source
    assert "grouped_gate_mass" in attention_production_source
    assert "local_grouped_space_results" in attention_production_source
    assert "local_grouped_output" in attention_production_source
    assert "grouped_output" in attention_source
    assert attention_production_source.count(
        "_global_dense_rw_den_sharded(") == 1
    assert attention_production_source.count(
        "_psum_dense_rw_representation_sharded(") == 1
    assert "jax.lax.psum(" not in attention_production_source
    assert attention_production_source.index(
        'writeback_dot(\n            "amtr,mrd->atd"') < (
            attention_production_source.index(
                "_psum_dense_rw_representation_sharded("))
    assert "local_space_result" in rst_production_source
    assert "local_update" in rst_production_source
    assert rst_production_source.count(
        "_global_dense_rw_den_sharded(") == 1
    assert rst_production_source.count(
        "_psum_dense_rw_representation_sharded(") == 1
    assert "jax.lax.psum(" not in rst_production_source
    assert rst_production_source.index(
        'writeback_dot(\n            "mtr,mrd->td"') < (
            rst_production_source.index(
                "_psum_dense_rw_representation_sharded("))
    assert "_reduce_dense_rw_output_sharded" not in attention_source
    assert "_reduce_dense_rw_output_sharded" not in rst_source
    assert "stop_gradient" not in attention_production_source
    assert "stop_gradient" not in rst_production_source
    assert "output_collective_bf16=True" in attention_minimal_source
    assert "output_collective_bf16=True" in rst_minimal_source
    assert "output_collective_bf16=False" in (
        attention_fp32_collective_source)
    assert "output_collective_bf16=False" in rst_fp32_collective_source
    assert "remat_chunks" not in attention_source
    assert "remat_chunks" not in rst_source
    assert "remat_chunks" not in builder_source
    assert '_v4174_chunk_remat_policy = "always"' in attention_source
    assert '_v4174_chunk_remat_policy = "always"' in rst_source
    assert "@partial(jax.jit, donate_argnums=(0, 1))" in trainer_source
    assert "static_argnames" not in trainer_source
    assert "collect_train_metrics=True" in trainer_source
    assert "step_after_update in (1, 5, 10, 20, 50)" in main_source
    assert "_upcoming_is_regular" in main_source
    assert "train_metrics_collected != 1.0" in main_source
    assert "del variables" in main_source
    assert "del dummy_input" in main_source
    assert "Scratch init cleanup:" in main_source
    assert "def _clear_startup_only_jax_caches():" in main_source
    assert "lowered_train_step = train_step_fn.lower(" in main_source
    assert "train_step_fn = lowered_train_step.compile()" in main_source
    compile_only_source = main_source.split(
        "# train_step donates params/opt_state.", 1)[1].split(
            "def _train_step_cache_size", 1)[0]
    assert "train_step_fn(" not in compile_only_source
    assert "donated state, compile-only" in compile_only_source
    assert "startup does not consume donated params/opt_state" in (
        compile_only_source)
    assert "if str(model_version) != V4174_MODEL_VERSION:" in main_source


def test_regular_record_uses_train_step_scalars_without_diagnostics():
    metrics = {
        "grad_norm": jnp.float32(1.25),
        "tau_update_qk_max_abs": jnp.float32(0.01),
        "tau_update_v_max_abs": jnp.float32(0.02),
        "tau_update_rst_max_abs": jnp.float32(0.03),
        **{
            key: jnp.float32(0.5)
            for key in (
                *train_jax.LINEAR_DIRECT_TAU_REGULAR_REQUIRED_METRIC_NAMES,
                *train_jax.V4174_COMPOSITION_REGULAR_METRIC_NAMES,
                *train_jax.V4174_SELECTOR_METRIC_NAMES)
        },
        "diagnostic_loss": jnp.float32(99.0),
    }
    win_avgs = {
        "loss": 2.0,
        "ce": 1.75,
        "aux": 0.25,
        "tau_reg": 0.1,
        "orth": 0.2,
        "div": 0.3,
        "acc": None,
    }
    ctx = {
        "current_lr": 1.0e-3,
        "model_version": train_jax.V4174_MODEL_VERSION,
        "total_micro_steps": 100,
        "progress": 5.0,
        "total_elapsed": 10.0,
        "eta": None,
    }

    record = train_jax._build_minimal_pretraining_record(
        metrics, win_avgs, ctx, 5, 1,
        sec_per_it=0.5, tokens_per_sec=2048.0)

    allowed = {
        *train_jax.MINIMAL_PRETRAINING_REQUIRED_LOG_KEYS,
        *train_jax.MINIMAL_PRETRAINING_OPTIONAL_LOG_KEYS,
        "timestamp",
    }
    assert set(record) <= allowed
    assert set(train_jax.MINIMAL_PRETRAINING_REQUIRED_LOG_KEYS) <= set(record)
    assert "diagnostic_loss" not in record
    assert "attention_space_gate_mass_mean" in record
    assert not any(
        key in record
        for key in train_jax.V4174_DIRECT_RW_GRADIENT_METRIC_NAMES)
    assert record["loss"] == 2.0
    assert record["tokens_per_sec"] == 2048.0
    assert all(
        not hasattr(value, "shape") or value.shape == ()
        for key, value in record.items()
        if key != "timestamp")


def test_public_analysis_imports_remain_available():
    importlib.import_module("analysis.dawn_analysis_common")
    importlib.import_module("analysis.operator_interpretability")
