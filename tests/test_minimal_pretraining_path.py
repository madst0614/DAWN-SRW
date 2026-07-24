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
    assert "v417x training requires production kernels" in source


def test_regular_record_uses_train_step_scalars_without_diagnostics():
    metrics = {
        "grad_norm": jnp.float32(1.25),
        "tau_update_qk_max_abs": jnp.float32(0.01),
        "tau_update_v_max_abs": jnp.float32(0.02),
        "tau_update_rst_max_abs": jnp.float32(0.03),
        **{
            key: jnp.float32(index + 1)
            for index, key in enumerate(
                train_jax.V4174_DIRECT_RW_GRADIENT_METRIC_NAMES)
        },
        "diagnostic_loss": jnp.float32(99.0),
        "attention_space_gate_mass_mean": jnp.float32(0.5),
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
    assert "attention_space_gate_mass_mean" not in record
    assert record["loss"] == 2.0
    assert record["tokens_per_sec"] == 2048.0
    assert all(
        not hasattr(value, "shape") or value.shape == ()
        for key, value in record.items()
        if key != "timestamp")


def test_public_analysis_imports_remain_available():
    importlib.import_module("analysis.dawn_analysis_common")
    importlib.import_module("analysis.operator_interpretability")
