"""Complete static report generation for DAWN-SRW analysis artifacts."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from analysis.dawn_analysis_common import AnalysisContext
from analysis.dawn_analysis_storage import (
    basename,
    dirname,
    exists,
    join_path,
    list_paths,
    makedirs,
    open_path,
    read_npz,
    write_bytes_atomic,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


CLAIM_NAMES = {
    "performance_retained": "Performance retained",
    "sparse_execution_works": "Sparse execution works",
    "operators_non_collapsed": "Operators are non-collapsed",
    "usage_is_structured": "Operators have structured usage",
    "operators_are_causal": "Operators are causal",
    "token_traces_available": "Token-level traces are available",
    "patching_steering_available": "Patching/steering available",
}

PERFORMANCE_LOSS_GAP_THRESHOLD = 0.05
SMALL_PRUNE_DELTA_THRESHOLD = 0.02
ABLATION_DELTA_THRESHOLD = 1.0e-4

PERFORMANCE_FIELDS = [
    "model",
    "checkpoint_step",
    "val_loss",
    "delta_vs_baseline",
    "ppl",
    "accuracy",
    "valid_tokens",
    "tok_per_sec",
]
PRUNE_FIELDS = [
    "eps",
    "val_loss",
    "delta_loss",
    "ppl",
    "accuracy",
    "valid_tokens",
    "compute_frac",
    "gate_mass_retained",
    "estimated_compute_frac",
]
GEOMETRY_FIELDS = [
    "pool",
    "N",
    "effective_rank",
    "mean_cosine",
    "max_cosine",
    "op_key_norm_mean",
    "op_key_norm_std",
    "read_norm_mean",
    "write_norm_mean",
    "dead_count",
]
USAGE_FIELDS = [
    "pool",
    "active_ops",
    "total_ops",
    "active_frac",
    "total_topk_mass",
    "tokens_observed",
    "mean_mass_per_active_op",
]
ABLATION_FIELDS = [
    "pool",
    "strategy",
    "k",
    "seed",
    "base_loss",
    "ablated_loss",
    "delta_loss",
    "base_acc",
    "ablated_acc",
    "delta_acc",
    "num_examples",
    "operator_ids",
]
OPERATOR_CARD_FIELDS = [
    "pool",
    "operator_id",
    "usage_rank",
    "total_mass",
    "activation_frequency",
    "top_context_1",
    "top_context_2",
    "top_context_3",
    "ablation_delta_loss_if_available",
]
TRACE_FIELDS = [
    "prompt_idx",
    "prompt_id",
    "length",
    "text",
    "token_ids",
    "heatmap_figure",
    "json_artifact",
    "topk_npz",
    "heatmap_csv",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "missing"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        value_f = float(value)
        if not math.isfinite(value_f):
            return "missing"
        if 0.0 < abs(value_f) < 0.001:
            return f"{value_f:.1e}"
        if abs(value_f) >= 1000:
            return f"{value_f:,.0f}"
        if abs(value_f) >= 10:
            return f"{value_f:,.2f}"
        return f"{value_f:,.{digits}f}"
    return str(value)


def _compact_path(path: Optional[str]) -> str:
    if not path:
        return ""
    return str(path).replace("\\", "/")


def _strip_output(path: Optional[str], output_dir: str) -> str:
    if not path:
        return ""
    path_s = _compact_path(path).rstrip("/")
    out_s = _compact_path(output_dir).rstrip("/")
    if path_s == out_s:
        return ""
    if path_s.startswith(out_s + "/"):
        return path_s[len(out_s) + 1 :]
    return path_s


def _href(path: Optional[str], output_dir: str, *, from_page: bool = False) -> str:
    """Return a relative link from report/index.html or report/pages/*.html."""
    rel = _strip_output(path, output_dir)
    if not rel:
        return "#"
    if rel.startswith("report/"):
        rel = rel[len("report/") :]
        return "../" + rel if from_page else rel
    return ("../../" if from_page else "../") + rel


def _artifact_link(path: Optional[str], output_dir: str, *, from_page: bool = False) -> str:
    if not path:
        return "missing"
    label = html.escape(_strip_output(path, output_dir) or str(path))
    return f'<a href="{html.escape(_href(path, output_dir, from_page=from_page))}">{label}</a>'


def _md_link(path: Optional[str], output_dir: str, label: Optional[str] = None) -> str:
    if not path:
        return "missing"
    rel = _strip_output(path, output_dir)
    if rel.startswith("report/"):
        href = rel[len("report/") :]
    else:
        href = "../" + rel
    return f"[{label or rel}]({href})"


def _record_error(errors: Optional[List[Dict[str, Any]]], path: str, err_type: str, exc: Exception | str) -> None:
    if errors is None:
        return
    errors.append({
        "path": str(path),
        "type": err_type,
        "error": str(exc),
    })


def safe_load_json(path: str, default: Any = None, errors: Optional[List[Dict[str, Any]]] = None) -> Any:
    """Load JSON without letting corrupt optional artifacts fail the report."""
    if not path or not exists(path):
        return deepcopy(default)
    try:
        with open_path(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        _record_error(errors, path, "invalid_json", exc)
        return deepcopy(default)


def safe_load_jsonl(path: str, max_bad_lines: int = 100,
                    errors: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Load JSONL while skipping bad lines up to a bounded warning count."""
    if not path or not exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    bad = 0
    try:
        with open_path(path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                    else:
                        bad += 1
                        if bad <= max_bad_lines:
                            _record_error(errors, path, "invalid_jsonl_object", f"line {line_no}: non-object row")
                except Exception as exc:
                    bad += 1
                    if bad <= max_bad_lines:
                        _record_error(errors, path, "invalid_jsonl_line", f"line {line_no}: {exc}")
    except Exception as exc:
        _record_error(errors, path, "invalid_jsonl", exc)
    if bad > max_bad_lines:
        _record_error(errors, path, "invalid_jsonl_line_limit", f"{bad} bad lines; reported first {max_bad_lines}")
    return rows


def safe_load_csv(path: str, default: Any = None, errors: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if not path or not exists(path):
        return deepcopy(default) if default is not None else []
    try:
        with open_path(path, "r") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        _record_error(errors, path, "invalid_csv", exc)
        return deepcopy(default) if default is not None else []


def safe_load_npz(path: str, default: Any = None, errors: Optional[List[Dict[str, Any]]] = None) -> Any:
    if not path or not exists(path):
        return deepcopy(default)
    try:
        return read_npz(path)
    except Exception as exc:
        _record_error(errors, path, "invalid_npz", exc)
        return deepcopy(default)


def _first_existing(paths: Sequence[Optional[str]]) -> Optional[str]:
    for path in paths:
        if path and exists(path):
            return path
    return None


def _list_safe(path: str, pattern: str) -> List[str]:
    try:
        return list_paths(path, pattern)
    except Exception:
        return []


def discover_artifacts(output_dir) -> dict:
    """Recursively inspect an analysis output directory and return an inventory."""
    output_dir = str(output_dir).rstrip("/")
    missing: List[Dict[str, str]] = []
    invalid: List[Dict[str, str]] = []

    def expected(rel: str, label: str, *, required: bool = True) -> Optional[str]:
        path = join_path(output_dir, rel)
        if exists(path):
            return path
        if required:
            missing.append({"path": path, "label": label})
        return None

    def stage_dir(stage: str) -> str:
        return join_path(output_dir, stage)

    recursive_paths = _list_safe(output_dir, "**/*")
    eps_summary_paths = (
        _list_safe(stage_dir("prune"), "eps_*/summary.json")
        + _list_safe(stage_dir("prune"), "eps_*/prune_summary.json")
    )
    eps_by_batch_paths = _list_safe(stage_dir("prune"), "eps_*/by_batch.jsonl")
    eps_dirs = sorted({dirname(p) for p in eps_summary_paths + eps_by_batch_paths})
    trace_prompt_jsons = [
        p for p in _list_safe(stage_dir("trace"), "prompt-*.json")
        if basename(p) != "trace_summary.json"
    ]
    patching_results = _result_files(stage_dir("patching"))
    steering_results = _result_files(stage_dir("steering"))

    inventory = {
        "output_dir": output_dir,
        "discovered_at": _utc_now(),
        "all_artifact_count": len(recursive_paths),
        "all_artifact_sample": recursive_paths[:200],
        "eval": {
            "final_eval": expected("eval/final_eval.json", "Final evaluation JSON"),
            "by_batch": expected("eval/final_eval_by_batch.jsonl", "Final evaluation by-batch JSONL", required=False),
        },
        "prune": {
            "summary": expected("prune/prune_sweep.json", "Prune sweep JSON"),
            "curve": expected("prune/prune_curve.csv", "Prune curve CSV"),
            "eps_dirs": eps_dirs,
            "eps_summaries": sorted(eps_summary_paths),
            "eps_by_batch": sorted(eps_by_batch_paths),
        },
        "geometry": {
            "neuron_health": expected("geometry/neuron_health.json", "Neuron health JSON", required=False),
            "weight_analysis": expected("geometry/weight_analysis.json", "Weight analysis JSON", required=False),
            "operator_norm_histograms": expected("geometry/operator_norm_histograms.npz", "Operator norm histograms NPZ", required=False),
            "summary": expected("geometry/operator_geometry_summary.json", "Operator geometry summary JSON"),
        },
        "usage": {
            "summary": expected("usage/operator_usage_summary.json", "Operator usage summary JSON"),
            "by_pool": expected("usage/operator_usage_by_pool.npz", "Operator usage by-pool NPZ", required=False),
            "operator_cards": expected("usage/operator_cards.jsonl", "Operator cards JSONL"),
            "operator_cards_top": expected("usage/operator_cards_top.md", "Operator cards Markdown", required=False),
            "usage_parts": _list_safe(join_path(output_dir, "usage", "usage_parts"), "*.json"),
            "top_contexts": _list_safe(join_path(output_dir, "usage", "top_contexts"), "*.jsonl"),
        },
        "trace": {
            "summary": expected("trace/trace_summary.json", "Trace summary JSON", required=False),
            "prompt_jsons": sorted(trace_prompt_jsons),
            "topk_npzs": sorted(_list_safe(stage_dir("trace"), "prompt-*topk.npz")),
            "heatmap_csvs": sorted(_list_safe(stage_dir("trace"), "prompt-*heatmap.csv")),
        },
        "ablation": {
            "summary": expected("ablation/summary.json", "Ablation summary JSON"),
            "curve": expected("ablation/ablation_curve.csv", "Ablation curve CSV", required=False),
            "jobs": sorted(_list_safe(join_path(output_dir, "ablation", "jobs"), "*.json")),
        },
        "patching": {
            "results": patching_results,
        },
        "steering": {
            "results": steering_results,
        },
        "report": {
            "existing": sorted(_list_safe(join_path(output_dir, "report"), "**/*")),
        },
        "missing": missing,
        "invalid": invalid,
    }
    if not trace_prompt_jsons:
        missing.append({"path": join_path(output_dir, "trace", "prompt-*.json"), "label": "Trace prompt JSON files"})
    if not inventory["trace"]["heatmap_csvs"] and not inventory["trace"]["topk_npzs"]:
        missing.append({"path": join_path(output_dir, "trace", "prompt-*_{heatmap.csv,topk.npz}"), "label": "Trace heatmap/top-k artifacts"})
    if not patching_results:
        missing.append({"path": join_path(output_dir, "patching"), "label": "Patching result files"})
    if not steering_results:
        missing.append({"path": join_path(output_dir, "steering"), "label": "Steering result files"})
    return inventory


def _result_files(path: str) -> List[str]:
    out: List[str] = []
    for pattern in ("**/*.json", "**/*.jsonl", "**/*.csv", "**/*.md", "**/*.npz"):
        out.extend(_list_safe(path, pattern))
    return sorted(set(out))


def _read_baseline_eval(args: Any, errors: List[Dict[str, Any]]) -> tuple[Dict[str, Any], Optional[str], Optional[dict]]:
    eval_json = getattr(args, "baseline_eval_json", None)
    baseline_output = getattr(args, "baseline_analysis_output", None)
    if eval_json:
        path = str(eval_json)
        return safe_load_json(path, {}, errors), path if exists(path) else path, None
    if baseline_output:
        inventory = discover_artifacts(str(baseline_output))
        path = inventory.get("eval", {}).get("final_eval")
        return safe_load_json(path, {}, errors) if path else {}, path, inventory
    return {}, None, None


def _read_prune_rows(prune: Dict[str, Any], errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sweep = safe_load_json(prune.get("summary"), {}, errors) if prune.get("summary") else {}
    rows = sweep.get("eps", []) if isinstance(sweep, dict) else []
    if not rows and prune.get("curve"):
        rows = safe_load_csv(prune.get("curve"), [], errors)
    norm: List[Dict[str, Any]] = []
    for row in rows or []:
        eps = _as_float(row.get("eps"))
        val_loss = _as_float(row.get("val_loss"))
        delta = row.get("delta_loss", row.get("delta_loss_vs_eps0"))
        estimated = row.get("estimated_compute_frac", row.get("compute_frac"))
        norm.append({
            "eps": eps,
            "val_loss": val_loss,
            "delta_loss": _as_float(delta, 0.0 if eps == 0.0 else float("nan")),
            "ppl": _as_float(row.get("ppl")),
            "accuracy": _as_float(row.get("accuracy")),
            "valid_tokens": _as_int(row.get("valid_tokens")),
            "compute_frac": _as_float(row.get("compute_frac", estimated)),
            "gate_mass_retained": _as_float(row.get("gate_mass_retained", row.get("execution_gate_mass_retained"))),
            "estimated_compute_frac": _as_float(estimated),
        })
    if norm and all(not math.isfinite(_as_float(r.get("delta_loss"))) for r in norm):
        base = next((r["val_loss"] for r in norm if _as_float(r.get("eps")) == 0.0 and math.isfinite(_as_float(r.get("val_loss")))), None)
        if base is None:
            base = norm[0].get("val_loss")
        for row in norm:
            row["delta_loss"] = _as_float(row.get("val_loss")) - _as_float(base)
    return norm


def _geometry_rows(geometry: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for pool, rec in sorted((geometry.get("pools") or {}).items()):
        dead_count = (
            _as_int(rec.get("op_key_dead"))
            + _as_int(rec.get("read_dead"))
            + _as_int(rec.get("write_dead"))
        )
        rows.append({
            "pool": pool,
            "N": _as_int(rec.get("N")),
            "effective_rank": _as_float(rec.get("effective_rank")),
            "mean_cosine": _as_float(rec.get("mean_cosine", rec.get("mean_cosine_similarity"))),
            "max_cosine": _as_float(rec.get("max_cosine", rec.get("max_cosine_similarity"))),
            "op_key_norm_mean": _as_float(rec.get("op_key_norm_mean")),
            "op_key_norm_std": _as_float(rec.get("op_key_norm_std")),
            "read_norm_mean": _as_float(rec.get("read_norm_mean")),
            "write_norm_mean": _as_float(rec.get("write_norm_mean")),
            "dead_count": dead_count,
        })
    return rows


def _usage_rows(usage: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    tokens = _as_int(usage.get("tokens_observed"))
    for pool, rec in sorted((usage.get("pools") or {}).items()):
        active = _as_int(rec.get("active_ops", rec.get("active_ops_observed")))
        total = _as_int(rec.get("total_ops", rec.get("N")))
        mass = _as_float(rec.get("total_topk_mass"), 0.0)
        rows.append({
            "pool": pool,
            "active_ops": active,
            "total_ops": total,
            "active_frac": _as_float(rec.get("active_frac", rec.get("active_op_frac_observed")), active / max(1, total)),
            "total_topk_mass": mass,
            "tokens_observed": tokens,
            "mean_mass_per_active_op": mass / max(1, active),
        })
    return rows


def _ablation_rows(ablation: Dict[str, Any], ablation_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_rows = ablation.get("jobs") if isinstance(ablation, dict) else None
    if not raw_rows:
        raw_rows = ablation_curve
    rows = []
    for rec in raw_rows or []:
        operator_ids = rec.get("operator_ids", "")
        if isinstance(operator_ids, (list, tuple)):
            operator_ids_out = json.dumps([int(x) for x in operator_ids])
        else:
            operator_ids_out = str(operator_ids)
        rows.append({
            "pool": rec.get("pool"),
            "strategy": rec.get("strategy"),
            "k": _as_int(rec.get("k")),
            "seed": rec.get("seed", rec.get("random_seed", rec.get("analysis_params", {}).get("seed") if isinstance(rec.get("analysis_params"), dict) else "")),
            "base_loss": _as_float(rec.get("base_loss")),
            "ablated_loss": _as_float(rec.get("ablated_loss")),
            "delta_loss": _as_float(rec.get("delta_loss")),
            "base_acc": _as_float(rec.get("base_acc")),
            "ablated_acc": _as_float(rec.get("ablated_acc")),
            "delta_acc": _as_float(rec.get("delta_acc")),
            "num_examples": _as_int(rec.get("num_examples", rec.get("valid_tokens", rec.get("valid_count")))),
            "operator_ids": operator_ids_out,
        })
    return rows


def _ablation_effect_by_operator(rows: List[Dict[str, Any]]) -> Dict[tuple[str, int], float]:
    out: Dict[tuple[str, int], float] = {}
    for row in rows:
        pool = str(row.get("pool") or "")
        if not pool:
            continue
        try:
            ids = json.loads(row.get("operator_ids") or "[]")
        except Exception:
            ids = []
        delta = _as_float(row.get("delta_loss"), 0.0)
        for op in ids:
            key = (pool, int(op))
            out[key] = max(out.get(key, float("-inf")), delta)
    return {k: v for k, v in out.items() if math.isfinite(v)}


def _operator_card_rows(cards: List[Dict[str, Any]], ablation_rows: List[Dict[str, Any]], max_cards: int) -> List[Dict[str, Any]]:
    effects = _ablation_effect_by_operator(ablation_rows)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for card in cards:
        grouped[str(card.get("pool") or "")].append(card)
    rows: List[Dict[str, Any]] = []
    for pool, pool_cards in sorted(grouped.items()):
        pool_cards = sorted(
            pool_cards,
            key=lambda r: (-_as_float(r.get("total_mass"), 0.0), _as_int(r.get("operator_id"))),
        )
        for rank, card in enumerate(pool_cards, start=1):
            contexts = card.get("top_contexts") or []
            context_texts = []
            for ctx in contexts[:3]:
                if not isinstance(ctx, dict):
                    continue
                text = str(ctx.get("text_window") or "")
                next_token = ctx.get("next_token_id")
                if next_token is not None:
                    text = f"{text} -> next_id:{next_token}"
                context_texts.append(text)
            op_id = _as_int(card.get("operator_id"))
            rows.append({
                "pool": pool,
                "operator_id": op_id,
                "usage_rank": rank,
                "total_mass": _as_float(card.get("total_mass"), 0.0),
                "activation_frequency": _as_float(card.get("activation_frequency"), 0.0),
                "top_context_1": context_texts[0] if len(context_texts) > 0 else "",
                "top_context_2": context_texts[1] if len(context_texts) > 1 else "",
                "top_context_3": context_texts[2] if len(context_texts) > 2 else "",
                "ablation_delta_loss_if_available": effects.get((pool, op_id), ""),
            })
    return rows[:max(0, int(max_cards))]


def _prompt_key(path: str) -> str:
    name = basename(path)
    match = re.search(r"(prompt-\d+)", name)
    return match.group(1) if match else name.split("_")[0].split(".")[0]


def _trace_rows(trace_inventory: Dict[str, Any], trace_summary: Dict[str, Any],
                errors: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    topk_by_key = {_prompt_key(p): p for p in trace_inventory.get("topk_npzs", [])}
    heat_by_key = {_prompt_key(p): p for p in trace_inventory.get("heatmap_csvs", [])}
    rows = []
    prompts = []
    prompt_paths = trace_inventory.get("prompt_jsons", [])
    if prompt_paths:
        for path in prompt_paths:
            meta = safe_load_json(path, {}, errors)
            key = _prompt_key(path)
            rec = {
                "prompt_idx": _as_int(meta.get("prompt_idx", len(rows))),
                "prompt_id": meta.get("prompt_id", key),
                "length": _as_int(meta.get("length")),
                "text": meta.get("text") or "",
                "token_ids": json.dumps(meta.get("token_ids", [])[:256]),
                "heatmap_figure": "",
                "json_artifact": path,
                "topk_npz": topk_by_key.get(key, ""),
                "heatmap_csv": heat_by_key.get(key, ""),
                "summary": meta.get("summary") or {},
            }
            rows.append({k: rec[k] for k in TRACE_FIELDS})
            prompts.append(rec)
    else:
        for idx, meta in enumerate(trace_summary.get("prompts", []) if isinstance(trace_summary, dict) else []):
            key = f"prompt-{idx:06d}"
            rec = {
                "prompt_idx": _as_int(meta.get("prompt_idx", idx)),
                "prompt_id": meta.get("prompt_id", key),
                "length": _as_int(meta.get("length")),
                "text": meta.get("text") or "",
                "token_ids": json.dumps(meta.get("token_ids", [])[:256]),
                "heatmap_figure": "",
                "json_artifact": "",
                "topk_npz": topk_by_key.get(key, ""),
                "heatmap_csv": heat_by_key.get(key, ""),
                "summary": meta.get("summary") or {},
            }
            rows.append({k: rec[k] for k in TRACE_FIELDS})
            prompts.append(rec)
    rows.sort(key=lambda r: _as_int(r.get("prompt_idx")))
    prompts.sort(key=lambda r: _as_int(r.get("prompt_idx")))
    return rows, prompts


def _model_d_route(config: Dict[str, Any]) -> int:
    model = config.get("model", {}) if isinstance(config, dict) else {}
    return _as_int(model.get("d_route", model.get("d_bottleneck")), 128)


def _build_report_data(store, args: Any, inventory: Dict[str, Any], errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    manifest = safe_load_json(store.manifest_path, {}, errors)
    model_info = safe_load_json(store.path("model_info.json"), {}, errors)
    config = safe_load_json(store.path("config_snapshot.json"), {}, errors)
    eval_data = safe_load_json(inventory.get("eval", {}).get("final_eval"), {}, errors)
    baseline_data, baseline_path, baseline_inventory = _read_baseline_eval(args, errors)
    prune_rows = _read_prune_rows(inventory.get("prune", {}), errors)
    geometry = safe_load_json(inventory.get("geometry", {}).get("summary"), {}, errors)
    geometry_rows = _geometry_rows(geometry if isinstance(geometry, dict) else {})
    usage = safe_load_json(inventory.get("usage", {}).get("summary"), {}, errors)
    usage_rows = _usage_rows(usage if isinstance(usage, dict) else {})
    cards = safe_load_jsonl(inventory.get("usage", {}).get("operator_cards"), errors=errors)
    trace_summary = safe_load_json(inventory.get("trace", {}).get("summary"), {}, errors)
    trace_rows, trace_prompts = _trace_rows(inventory.get("trace", {}), trace_summary if isinstance(trace_summary, dict) else {}, errors)
    ablation = safe_load_json(inventory.get("ablation", {}).get("summary"), {}, errors)
    ablation_curve = safe_load_csv(inventory.get("ablation", {}).get("curve"), [], errors)
    ablation_rows = _ablation_rows(ablation if isinstance(ablation, dict) else {}, ablation_curve)
    max_cards = _as_int(getattr(args, "report_max_operator_cards", 100), 100)
    card_rows = _operator_card_rows(cards, ablation_rows, max_cards)
    model_version = (
        manifest.get("model_version")
        or (config.get("model", {}) if isinstance(config, dict) else {}).get(
            "model_version")
        or "unknown"
    )
    model_label = f"DAWN-SRW {model_version}"

    perf_rows = []
    if eval_data:
        baseline_loss = _as_float(baseline_data.get("val_loss")) if baseline_data else float("nan")
        dawn_loss = _as_float(eval_data.get("val_loss"))
        perf_rows.append({
            "model": model_label,
            "checkpoint_step": eval_data.get("checkpoint_step", manifest.get("checkpoint_step")),
            "val_loss": dawn_loss,
            "delta_vs_baseline": dawn_loss - baseline_loss if math.isfinite(baseline_loss) and math.isfinite(dawn_loss) else "",
            "ppl": _as_float(eval_data.get("ppl")),
            "accuracy": _as_float(eval_data.get("accuracy")),
            "valid_tokens": _as_int(eval_data.get("valid_tokens")),
            "tok_per_sec": _as_float(eval_data.get("tok_per_sec")),
        })
    if baseline_data:
        perf_rows.append({
            "model": "Baseline",
            "checkpoint_step": baseline_data.get("checkpoint_step"),
            "val_loss": _as_float(baseline_data.get("val_loss")),
            "delta_vs_baseline": 0.0,
            "ppl": _as_float(baseline_data.get("ppl")),
            "accuracy": _as_float(baseline_data.get("accuracy")),
            "valid_tokens": _as_int(baseline_data.get("valid_tokens")),
            "tok_per_sec": _as_float(baseline_data.get("tok_per_sec")),
        })

    return {
        "generated_at": _utc_now(),
        "model_version": model_version,
        "model_label": model_label,
        "output_dir": store.output_dir,
        "manifest": manifest if isinstance(manifest, dict) else {},
        "model_info": model_info if isinstance(model_info, dict) else {},
        "config": config if isinstance(config, dict) else {},
        "eval": eval_data if isinstance(eval_data, dict) else {},
        "baseline_eval": baseline_data if isinstance(baseline_data, dict) else {},
        "baseline_path": baseline_path,
        "baseline_inventory": baseline_inventory,
        "prune_rows": prune_rows,
        "geometry": geometry if isinstance(geometry, dict) else {},
        "geometry_rows": geometry_rows,
        "usage": usage if isinstance(usage, dict) else {},
        "usage_rows": usage_rows,
        "operator_cards": cards,
        "operator_card_rows": card_rows,
        "trace_summary": trace_summary if isinstance(trace_summary, dict) else {},
        "trace_rows": trace_rows,
        "trace_prompts": trace_prompts,
        "ablation": ablation if isinstance(ablation, dict) else {},
        "ablation_rows": ablation_rows,
        "inventory": inventory,
        "errors": errors,
        "d_route": _model_d_route(config if isinstance(config, dict) else {}),
        "performance_rows": perf_rows,
    }


def _save_figure(fig, path: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    write_bytes_atomic(path, buf.getvalue())
    return path


def _plt():
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = os.path.join(tempfile.gettempdir(), "dawn_analysis_matplotlib")
        os.makedirs(mpl_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_dir
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _valid_rows(rows: Sequence[Dict[str, Any]], *keys: str) -> List[Dict[str, Any]]:
    return [r for r in rows if all(math.isfinite(_as_float(r.get(k))) for k in keys)]


def plot_performance_summary(eval_data, baseline_data, fig_dir):
    if not eval_data:
        return None
    plt = _plt()
    labels = ["DAWN"]
    losses = [_as_float(eval_data.get("val_loss"))]
    accs = [_as_float(eval_data.get("accuracy"))]
    if baseline_data:
        labels.append("Baseline")
        losses.append(_as_float(baseline_data.get("val_loss")))
        accs.append(_as_float(baseline_data.get("accuracy")))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].bar(labels, losses, color=["#3b82f6", "#64748b"][: len(labels)])
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Loss")
    axes[1].bar(labels, accs, color=["#16a34a", "#64748b"][: len(labels)])
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, join_path(fig_dir, "performance_summary.png"))


def plot_prune_loss_vs_compute(prune_df, fig_dir):
    rows = _valid_rows(prune_df, "delta_loss")
    if not rows:
        return None
    use_compute = any(math.isfinite(_as_float(r.get("estimated_compute_frac"))) for r in rows)
    x_key = "estimated_compute_frac" if use_compute else "eps"
    rows = _valid_rows(rows, x_key, "delta_loss")
    if not rows:
        return None
    plt = _plt()
    x = [_as_float(r.get(x_key)) for r in rows]
    y = [_as_float(r.get("delta_loss")) for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(x, y, marker="o", color="#2563eb")
    ax.axhline(0.0, color="#475569", linewidth=1, alpha=0.7)
    for xi, yi, row in zip(x, y, rows):
        ax.annotate(f"eps={_fmt(row.get('eps'), 1)}", (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Estimated compute fraction" if use_compute else "Prune epsilon")
    ax.set_ylabel("Delta validation loss vs eps=0")
    ax.set_title("Prune curve: loss vs compute")
    ax.grid(alpha=0.25)
    return _save_figure(fig, join_path(fig_dir, "prune_curve_loss_vs_compute.png"))


def plot_prune_eps_vs_loss(prune_df, fig_dir):
    rows = _valid_rows(prune_df, "eps", "val_loss")
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: _as_float(r.get("eps")))
    plt = _plt()
    labels = [_fmt(r.get("eps"), 1) for r in rows]
    y = [_as_float(r.get("val_loss")) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(range(len(rows)), y, marker="o", color="#0f766e")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Prune epsilon")
    ax.set_ylabel("Validation loss")
    ax.set_title("Prune curve: loss by epsilon")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, join_path(fig_dir, "prune_curve_eps_vs_loss.png"))


def plot_geometry_rank_cosine(geometry_data, fig_dir):
    rows = geometry_data if isinstance(geometry_data, list) else _geometry_rows(geometry_data or {})
    rows = [r for r in rows if r.get("pool")]
    if not rows:
        return None
    plt = _plt()
    pools = [str(r["pool"]).upper() for r in rows]
    ranks = [_as_float(r.get("effective_rank"), 0.0) for r in rows]
    mean_cos = [_as_float(r.get("mean_cosine"), 0.0) for r in rows]
    max_cos = [_as_float(r.get("max_cosine"), 0.0) for r in rows]
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    ax2 = ax1.twinx()
    ax1.bar(x, ranks, width=0.45, color="#3b82f6", label="effective rank")
    ax2.plot(x, mean_cos, marker="o", color="#f59e0b", label="mean cosine")
    ax2.plot(x, max_cos, marker="s", color="#dc2626", label="max cosine")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pools)
    ax1.set_ylabel("Effective rank")
    ax2.set_ylabel("Cosine similarity")
    ax1.set_title("Operator geometry by pool")
    ax1.grid(axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    return _save_figure(fig, join_path(fig_dir, "geometry_rank_cosine.png"))


def plot_usage_active_ops_by_pool(usage_data, fig_dir):
    rows = usage_data if isinstance(usage_data, list) else _usage_rows(usage_data or {})
    if not rows:
        return None
    plt = _plt()
    pools = [str(r["pool"]).upper() for r in rows]
    vals = [_as_float(r.get("active_frac"), 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.bar(pools, vals, color=["#3b82f6", "#16a34a", "#f59e0b"][: len(rows)])
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{_as_int(row.get('active_ops')):,}/{_as_int(row.get('total_ops')):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, max(vals + [1.0]) * 1.15 if max(vals + [0.0]) > 1.0 else 1.0)
    ax.set_ylabel("Active fraction")
    ax.set_title("Active operators by pool")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, join_path(fig_dir, "usage_active_ops_by_pool.png"))


def plot_usage_mass_by_pool(usage_data, fig_dir):
    rows = usage_data if isinstance(usage_data, list) else _usage_rows(usage_data or {})
    if not rows:
        return None
    plt = _plt()
    pools = [str(r["pool"]).upper() for r in rows]
    vals = [_as_float(r.get("total_topk_mass"), 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(pools, vals, color=["#2563eb", "#059669", "#d97706"][: len(rows)])
    ax.set_ylabel("Total top-k mass")
    ax.set_title("Top-k mass by pool")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, join_path(fig_dir, "usage_mass_by_pool.png"))


def plot_ablation_delta_loss(ablation_df, fig_dir):
    rows = _valid_rows(ablation_df, "delta_loss")
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: (str(r.get("pool")), str(r.get("strategy")), _as_int(r.get("k"))))
    labels = [f"{r.get('pool')}/{r.get('strategy')}/k={r.get('k')}" for r in rows]
    vals = [_as_float(r.get("delta_loss")) for r in rows]
    plt = _plt()
    width = max(7.0, min(14.0, 0.45 * len(rows)))
    fig, ax = plt.subplots(figsize=(width, 4.2))
    ax.bar(range(len(rows)), vals, color="#7c3aed")
    ax.axhline(0.0, color="#475569", linewidth=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Delta validation loss")
    ax.set_title("Causal ablation effect by pool and strategy")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, join_path(fig_dir, "ablation_delta_loss_by_pool_strategy.png"))


def plot_ablation_top_vs_random(ablation_df, fig_dir):
    rows = _valid_rows(ablation_df, "k", "delta_loss")
    strategies = sorted({str(r.get("strategy")) for r in rows if r.get("strategy")})
    if not rows or "top" not in strategies or not ({"random", "low", "low-use"} & set(strategies)):
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {"top": "#dc2626", "random": "#2563eb", "low": "#16a34a", "low-use": "#16a34a"}
    for strategy in strategies:
        sr = [r for r in rows if str(r.get("strategy")) == strategy]
        by_k: Dict[int, List[float]] = defaultdict(list)
        for r in sr:
            by_k[_as_int(r.get("k"))].append(_as_float(r.get("delta_loss")))
        ks = sorted(by_k)
        vals = [float(np.mean(by_k[k])) for k in ks]
        ax.plot(ks, vals, marker="o", label=strategy, color=colors.get(strategy))
    ax.set_xlabel("Suppressed operators k")
    ax.set_ylabel("Mean delta validation loss")
    ax.set_title("Top-use ablation vs comparator")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save_figure(fig, join_path(fig_dir, "ablation_curve_top_vs_random.png"))


def plot_trace_heatmap(trace_data, fig_dir, prompt_id):
    csv_path = trace_data.get("heatmap_csv")
    rows = safe_load_csv(csv_path, [], trace_data.setdefault("errors", [])) if csv_path else []
    if not rows:
        return None
    max_layer = max((_as_int(r.get("layer"), -1) for r in rows), default=-1)
    max_pos = max((_as_int(r.get("position"), -1) for r in rows), default=-1)
    if max_layer < 0 or max_pos < 0:
        return None
    mat = np.zeros((max_layer + 1, max_pos + 1), dtype=np.float32)
    for row in rows:
        layer = _as_int(row.get("layer"), -1)
        pos = _as_int(row.get("position"), -1)
        if layer < 0 or pos < 0:
            continue
        mat[layer, pos] += _as_float(row.get("active_count"), 0.0)
    if not np.any(mat):
        for row in rows:
            layer = _as_int(row.get("layer"), -1)
            pos = _as_int(row.get("position"), -1)
            if layer >= 0 and pos >= 0:
                mat[layer, pos] += _as_float(row.get("top1_frac"), 0.0)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(6.0, min(12.0, mat.shape[1] * 0.18)), 4.2))
    im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_title(f"Trace heatmap: {prompt_id}")
    fig.colorbar(im, ax=ax, label="Active count sum")
    safe_id = (re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prompt_id))[:80] or "prompt").replace("-", "_")
    return _save_figure(fig, join_path(fig_dir, f"trace_gallery_{safe_id}.png"))


def _generate_figures(data: Dict[str, Any], store, args: Any) -> Dict[str, Any]:
    fig_dir = store.path("report", "figures")
    makedirs(fig_dir)
    figures: Dict[str, Any] = {}
    for name, path in (
        ("performance_summary", plot_performance_summary(data["eval"], data["baseline_eval"], fig_dir)),
        ("prune_curve_loss_vs_compute", plot_prune_loss_vs_compute(data["prune_rows"], fig_dir)),
        ("prune_curve_eps_vs_loss", plot_prune_eps_vs_loss(data["prune_rows"], fig_dir)),
        ("geometry_rank_cosine", plot_geometry_rank_cosine(data["geometry_rows"], fig_dir)),
        ("usage_active_ops_by_pool", plot_usage_active_ops_by_pool(data["usage_rows"], fig_dir)),
        ("usage_mass_by_pool", plot_usage_mass_by_pool(data["usage_rows"], fig_dir)),
        ("ablation_delta_loss_by_pool_strategy", plot_ablation_delta_loss(data["ablation_rows"], fig_dir)),
        ("ablation_curve_top_vs_random", plot_ablation_top_vs_random(data["ablation_rows"], fig_dir)),
    ):
        if path:
            figures[name] = path

    trace_figs = []
    max_trace = _as_int(getattr(args, "report_max_trace_figures", 12), 12)
    trace_errors: List[Dict[str, Any]] = []
    for rec in data["trace_prompts"][:max(0, max_trace)]:
        prompt_id = rec.get("prompt_id") or f"prompt-{rec.get('prompt_idx', len(trace_figs)):06d}"
        payload = dict(rec)
        payload["errors"] = trace_errors
        path = plot_trace_heatmap(payload, fig_dir, prompt_id)
        if path:
            rec["heatmap_figure"] = path
            for row in data["trace_rows"]:
                if row.get("prompt_id") == rec.get("prompt_id"):
                    row["heatmap_figure"] = path
                    break
            trace_figs.append(path)
    data["errors"].extend(trace_errors)
    if trace_figs:
        figures["trace_gallery"] = trace_figs
    return figures


def interpret_performance(eval_data, baseline_data=None) -> list[str]:
    if not eval_data:
        return ["Final evaluation is missing, so performance cannot be judged from this report bundle."]
    loss = _as_float(eval_data.get("val_loss"))
    tokens = _as_int(eval_data.get("valid_tokens"))
    acc = _as_float(eval_data.get("accuracy"))
    lines = [
        f"The final checkpoint reaches val_loss={_fmt(loss)} over {_fmt(tokens)} valid tokens with accuracy={_fmt(acc)}."
    ]
    if baseline_data:
        base_loss = _as_float(baseline_data.get("val_loss"))
        gap = loss - base_loss
        if math.isfinite(gap):
            lines.append(
                f"The matched baseline gap is delta_loss={_fmt(gap)}; gaps within {_fmt(PERFORMANCE_LOSS_GAP_THRESHOLD, 3)} support retained performance."
            )
        else:
            lines.append("Baseline evaluation exists but loss fields are incomplete, so parity is not scored.")
    else:
        lines.append("Baseline comparison is missing, so this supports standalone LM competence, not matched-performance parity.")
    if tokens < 1_000_000:
        lines.append("The validation token count is below 1M, so the estimate should be treated as a smoke-scale result.")
    return lines


def interpret_prune(prune_df) -> list[str]:
    if not prune_df:
        return ["The prune curve is missing, so sparse execution cannot be evaluated here."]
    rows = sorted(prune_df, key=lambda r: _as_float(r.get("eps")))
    base = next((r for r in rows if _as_float(r.get("eps")) == 0.0), rows[0])
    compute = _as_float(base.get("estimated_compute_frac", base.get("compute_frac")))
    nonzero = [r for r in rows if _as_float(r.get("eps")) > 0 and math.isfinite(_as_float(r.get("delta_loss")))]
    robust = [r for r in nonzero if _as_float(r.get("delta_loss")) <= SMALL_PRUNE_DELTA_THRESHOLD]
    lines = []
    if math.isfinite(compute):
        lines.append(
            f"The eps=0 estimated compute fraction is {_fmt(compute)}, indicating sparse execution before additional hard pruning."
        )
    else:
        lines.append("Compute fraction is missing from the prune curve, so the report falls back to epsilon-vs-loss inspection.")
    if robust:
        best = min(robust, key=lambda r: _as_float(r.get("estimated_compute_frac"), 1.0))
        lines.append(
            f"At least one nonzero prune point keeps delta_loss <= {_fmt(SMALL_PRUNE_DELTA_THRESHOLD, 3)}; the lowest-compute such point is eps={_fmt(best.get('eps'), 1)}."
        )
    else:
        lines.append("No nonzero prune point clearly preserves loss under the small-delta threshold; robustness to extra pruning is not established.")
    return lines


def interpret_geometry(geometry_data) -> list[str]:
    rows = geometry_data if isinstance(geometry_data, list) else _geometry_rows(geometry_data or {})
    if not rows:
        return ["Geometry artifacts are missing, so operator collapse cannot be assessed."]
    ranks = [_as_float(r.get("effective_rank")) for r in rows if math.isfinite(_as_float(r.get("effective_rank")))]
    means = [_as_float(r.get("mean_cosine")) for r in rows if math.isfinite(_as_float(r.get("mean_cosine")))]
    maxes = [_as_float(r.get("max_cosine")) for r in rows if math.isfinite(_as_float(r.get("max_cosine")))]
    lines = []
    if ranks:
        lines.append(
            f"Pool effective ranks span {_fmt(min(ranks))}-{_fmt(max(ranks))}, which is evidence against global operator-key collapse when high relative to d_route."
        )
    if means:
        lines.append(f"Mean cosine similarity spans {_fmt(min(means))}-{_fmt(max(means))}; lower values indicate broader geometric separation.")
    if maxes and max(maxes) > 0.9:
        lines.append("High max cosine values indicate local near-duplicates or tight clusters even if the global rank remains high.")
    return lines or ["Geometry data exists, but the key rank/cosine fields are incomplete."]


def interpret_usage(usage_data) -> list[str]:
    rows = usage_data if isinstance(usage_data, list) else _usage_rows(usage_data or {})
    if not rows:
        return ["Usage atlas artifacts are missing, so operator usage structure cannot be assessed."]
    parts = [
        f"{str(r.get('pool')).upper()} uses {_fmt(r.get('active_ops'))}/{_fmt(r.get('total_ops'))} operators"
        for r in rows
    ]
    lines = ["; ".join(parts) + "."]
    fracs = [_as_float(r.get("active_frac")) for r in rows if math.isfinite(_as_float(r.get("active_frac")))]
    if len(fracs) >= 2 and max(fracs) - min(fracs) > 0.05:
        lines.append("Active fractions differ meaningfully by pool, suggesting differentiated routing roles rather than one uniform usage pattern.")
    else:
        lines.append("Active fractions are similar or incomplete; inspect operator cards before making a strong structured-usage claim.")
    return lines


def interpret_ablation(ablation_df) -> list[str]:
    if not ablation_df:
        return ["Ablation artifacts are missing, so causal operator claims are not supported by this run."]
    top = [r for r in ablation_df if str(r.get("strategy")) == "top"]
    comps = [r for r in ablation_df if str(r.get("strategy")) in ("random", "low", "low-use")]
    lines = []
    if top:
        best = max(top, key=lambda r: _as_float(r.get("delta_loss"), float("-inf")))
        lines.append(
            f"The strongest top-use ablation is {best.get('pool')}/k={best.get('k')} with delta_loss={_fmt(best.get('delta_loss'))}."
        )
    if comps:
        top_mean = float(np.mean([_as_float(r.get("delta_loss"), 0.0) for r in top])) if top else 0.0
        comp_mean = float(np.mean([_as_float(r.get("delta_loss"), 0.0) for r in comps]))
        lines.append(
            f"Comparator ablations are present; mean top delta_loss={_fmt(top_mean)} versus comparator mean={_fmt(comp_mean)}."
        )
    else:
        lines.append("Random or low-use comparators are missing, so top-use causal effects remain partial evidence.")
    return lines


def _claim(status: str, score: float, headline: str,
           evidence: Sequence[str] = (), missing: Sequence[str] = ()) -> Dict[str, Any]:
    return {
        "status": status,
        "score": float(max(0.0, min(1.0, score))),
        "headline": headline,
        "evidence": list(evidence),
        "missing": list(missing),
    }


def _build_claim_checklist(data: Dict[str, Any], figures: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = data["output_dir"]
    eval_data = data["eval"]
    baseline = data["baseline_eval"]
    inv = data["inventory"]
    claims: Dict[str, Any] = {}

    if eval_data:
        tokens = _as_int(eval_data.get("valid_tokens"))
        if baseline:
            gap = _as_float(eval_data.get("val_loss")) - _as_float(baseline.get("val_loss"))
            supported = tokens > 1_000_000 and math.isfinite(gap) and gap <= PERFORMANCE_LOSS_GAP_THRESHOLD
            claims["performance_retained"] = _claim(
                "supported" if supported else "partial",
                1.0 if supported else 0.55,
                f"Eval exists; baseline gap is {_fmt(gap)} over {_fmt(tokens)} tokens.",
                [inv.get("eval", {}).get("final_eval"), data.get("baseline_path"), figures.get("performance_summary")],
                [] if supported else ["Need >1M tokens and a loss gap within threshold."],
            )
        else:
            claims["performance_retained"] = _claim(
                "partial",
                0.45 if tokens > 0 else 0.2,
                "DAWN eval exists, but no baseline comparison was supplied.",
                [inv.get("eval", {}).get("final_eval"), figures.get("performance_summary")],
                ["Baseline final_eval.json or --baseline-analysis-output is required for parity."],
            )
    else:
        claims["performance_retained"] = _claim("missing", 0.0, "Final eval is missing.", [], ["eval/final_eval.json"])

    prune_rows = data["prune_rows"]
    compute_exists = any(math.isfinite(_as_float(r.get("estimated_compute_frac"))) for r in prune_rows)
    robust = [
        r for r in prune_rows
        if _as_float(r.get("eps")) > 0 and math.isfinite(_as_float(r.get("delta_loss"))) and _as_float(r.get("delta_loss")) <= SMALL_PRUNE_DELTA_THRESHOLD
    ]
    if prune_rows and compute_exists and robust:
        best = min(robust, key=lambda r: _as_float(r.get("estimated_compute_frac"), 1.0))
        claims["sparse_execution_works"] = _claim(
            "supported",
            1.0,
            f"Prune curve has compute fractions and a low-loss nonzero eps point at eps={_fmt(best.get('eps'), 1)}.",
            [inv.get("prune", {}).get("summary"), inv.get("prune", {}).get("curve"), figures.get("prune_curve_loss_vs_compute")],
            [],
        )
    elif prune_rows:
        claims["sparse_execution_works"] = _claim(
            "partial",
            0.55 if compute_exists else 0.35,
            "Prune curve exists, but compute fractions or robust nonzero prune points are incomplete.",
            [inv.get("prune", {}).get("summary"), inv.get("prune", {}).get("curve")],
            ["Need compute_frac and at least one nonzero eps with small delta_loss."],
        )
    else:
        claims["sparse_execution_works"] = _claim("missing", 0.0, "Prune curve is missing.", [], ["prune/prune_sweep.json", "prune/prune_curve.csv"])

    geom_rows = data["geometry_rows"]
    d_route = max(1, _as_int(data.get("d_route"), 128))
    rank_threshold = min(float(d_route) * 0.5, float(d_route))
    geom_supported = bool(geom_rows) and all(_as_float(r.get("effective_rank"), 0.0) >= rank_threshold for r in geom_rows) and all(_as_float(r.get("mean_cosine"), 1.0) < 0.35 for r in geom_rows)
    if geom_supported:
        claims["operators_non_collapsed"] = _claim(
            "supported",
            1.0,
            f"All pools have effective rank >= {_fmt(rank_threshold)} and moderate mean cosine.",
            [inv.get("geometry", {}).get("summary"), figures.get("geometry_rank_cosine")],
            [],
        )
    elif geom_rows:
        claims["operators_non_collapsed"] = _claim(
            "partial",
            0.55,
            "Geometry exists, but rank/cosine thresholds are not fully met or config d_route is uncertain.",
            [inv.get("geometry", {}).get("summary"), figures.get("geometry_rank_cosine")],
            ["Inspect per-pool rank, mean cosine, max cosine, and dead counts."],
        )
    else:
        claims["operators_non_collapsed"] = _claim("missing", 0.0, "Geometry summary is missing.", [], ["geometry/operator_geometry_summary.json"])

    usage_rows = data["usage_rows"]
    fracs = [_as_float(r.get("active_frac")) for r in usage_rows if math.isfinite(_as_float(r.get("active_frac")))]
    top_contexts = inv.get("usage", {}).get("top_contexts") or []
    cards = data.get("operator_cards") or []
    usage_structured = bool(usage_rows) and len(fracs) >= 2 and (max(fracs) - min(fracs) > 0.05) and bool(cards or top_contexts)
    if usage_structured:
        claims["usage_is_structured"] = _claim(
            "supported",
            1.0,
            "Usage atlas shows pool-level active-fraction differences and context evidence.",
            [inv.get("usage", {}).get("summary"), inv.get("usage", {}).get("operator_cards")],
            [],
        )
    elif usage_rows:
        claims["usage_is_structured"] = _claim(
            "partial",
            0.55,
            "Usage summary exists, but pool differences or top contexts are incomplete.",
            [inv.get("usage", {}).get("summary")],
            ["Need operator cards/top contexts and meaningful pool-level usage differences."],
        )
    else:
        claims["usage_is_structured"] = _claim("missing", 0.0, "Usage atlas is missing.", [], ["usage/operator_usage_summary.json", "usage/operator_cards.jsonl"])

    abl_rows = data["ablation_rows"]
    top_rows = [r for r in abl_rows if str(r.get("strategy")) == "top" and _as_float(r.get("delta_loss"), 0.0) > ABLATION_DELTA_THRESHOLD]
    comp_rows = [r for r in abl_rows if str(r.get("strategy")) in ("random", "low", "low-use")]
    top_mean = float(np.mean([_as_float(r.get("delta_loss"), 0.0) for r in top_rows])) if top_rows else 0.0
    comp_mean = float(np.mean([_as_float(r.get("delta_loss"), 0.0) for r in comp_rows])) if comp_rows else 0.0
    if top_rows and comp_rows and top_mean > comp_mean:
        claims["operators_are_causal"] = _claim(
            "supported",
            1.0,
            f"Top-use ablations increase loss more than comparators ({_fmt(top_mean)} vs {_fmt(comp_mean)} mean delta_loss).",
            [inv.get("ablation", {}).get("summary"), figures.get("ablation_delta_loss_by_pool_strategy")],
            [],
        )
    elif top_rows:
        claims["operators_are_causal"] = _claim(
            "partial",
            0.6,
            "Top-use ablations increase loss, but random/low-use comparators are missing or weaker evidence is incomplete.",
            [inv.get("ablation", {}).get("summary"), figures.get("ablation_delta_loss_by_pool_strategy")],
            ["Run --ablation-strategies top,random,low for comparator evidence."],
        )
    elif abl_rows:
        claims["operators_are_causal"] = _claim(
            "partial",
            0.35,
            "Ablation rows exist, but top-use loss increases are not clear.",
            [inv.get("ablation", {}).get("summary")],
            ["Inspect ablation jobs and rerun with more examples if deltas are noisy."],
        )
    else:
        claims["operators_are_causal"] = _claim("missing", 0.0, "Ablation results are missing.", [], ["ablation/summary.json", "ablation/ablation_curve.csv"])

    trace_count = len(data.get("trace_prompts") or [])
    trace_fig_count = len(figures.get("trace_gallery", [])) if isinstance(figures.get("trace_gallery"), list) else 0
    if trace_count > 0 and trace_fig_count > 0:
        claims["token_traces_available"] = _claim(
            "supported",
            1.0,
            f"{trace_count} trace prompt(s) discovered and {trace_fig_count} heatmap figure(s) generated.",
            [inv.get("trace", {}).get("summary"), *figures.get("trace_gallery", [])],
            [],
        )
    elif trace_count > 0:
        claims["token_traces_available"] = _claim(
            "partial",
            0.55,
            "Trace metadata exists, but heatmap figures could not be generated.",
            [inv.get("trace", {}).get("summary")],
            ["Need prompt-*heatmap.csv or valid compact top-k trace artifacts."],
        )
    else:
        claims["token_traces_available"] = _claim("missing", 0.0, "No trace prompts were discovered.", [], ["trace/prompt-*.json", "trace/prompt-*heatmap.csv"])

    patch_results = inv.get("patching", {}).get("results") or []
    steering_results = inv.get("steering", {}).get("results") or []
    if patch_results and steering_results:
        status, score = "supported", 1.0
        headline = "Patching and steering result files are present."
        missing = []
    elif patch_results or steering_results:
        status, score = "partial", 0.5
        headline = "Only one of patching or steering has result files."
        missing = ["Need both patching and steering result artifacts for the full claim."]
    else:
        status, score = "missing", 0.0
        headline = "Patching/steering directories contain no result files; scaffolding alone is not evidence."
        missing = ["Run actual patching/steering analyses and write result artifacts."]
    claims["patching_steering_available"] = _claim(
        status,
        score,
        headline,
        patch_results[:5] + steering_results[:5],
        missing,
    )
    return claims


def _table_preview(rows: Sequence[Dict[str, Any]], fields: Sequence[str], limit: int = 12,
                   output_dir: Optional[str] = None, from_page: bool = False) -> str:
    if not rows:
        return '<p class="muted">No rows available.</p>'
    cols = list(fields)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body_rows = []
    for row in rows[:limit]:
        cells = []
        for col in cols:
            val = row.get(col, "")
            if col.endswith("artifact") or col.endswith("csv") or col.endswith("npz") or col.endswith("figure"):
                cell = _artifact_link(val, output_dir or "", from_page=from_page) if val else ""
            else:
                cell = html.escape(_fmt(val) if isinstance(val, (float, int, np.floating, np.integer)) else str(val))
            cells.append(f"<td>{cell}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    more = f'<p class="muted">Showing {min(limit, len(rows)):,} of {len(rows):,} rows.</p>' if len(rows) > limit else ""
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>{more}'


def _figure_html(path: Optional[str], output_dir: str, *, from_page: bool = False, alt: str = "") -> str:
    if not path:
        return '<p class="muted">Figure unavailable for this section.</p>'
    return f'<figure><img src="{html.escape(_href(path, output_dir, from_page=from_page))}" alt="{html.escape(alt)}"></figure>'


def _findings_html(lines: Sequence[str]) -> str:
    if not lines:
        return '<p class="muted">No findings available.</p>'
    return "<ul>" + "".join(f"<li>{html.escape(str(line))}</li>" for line in lines) + "</ul>"


def _section_html(section: Dict[str, Any], data: Dict[str, Any], *, from_page: bool = False) -> str:
    output_dir = data["output_dir"]
    figs = "".join(_figure_html(p, output_dir, from_page=from_page, alt=section["title"]) for p in section.get("figures", []) if p)
    table = _table_preview(
        section.get("table_rows", []),
        section.get("table_fields", []),
        limit=section.get("table_limit", 12),
        output_dir=output_dir,
        from_page=from_page,
    ) if section.get("table_fields") else ""
    artifacts = section.get("artifacts") or []
    artifact_html = ""
    if artifacts:
        artifact_html = (
            '<div class="artifact-list"><b>Source artifacts:</b><ul>'
            + "".join(f"<li>{_artifact_link(p, output_dir, from_page=from_page)}</li>" for p in artifacts if p)
            + "</ul></div>"
        )
    caveats = _findings_html(section.get("caveats", []))
    return f"""
<section id="{html.escape(section['id'])}">
  <div class="section-head">
    <h2>{html.escape(section['title'])}</h2>
    <p class="headline">{html.escape(section.get('headline') or '')}</p>
  </div>
  <div class="grid two">
    <div>
      <h3>Key Findings</h3>
      {_findings_html(section.get('findings', []))}
      <h3>Interpretation</h3>
      {_findings_html(section.get('interpretation', []))}
    </div>
    <div>
      {figs or '<p class="muted">No figure generated for this section.</p>'}
    </div>
  </div>
  {table}
  {artifact_html}
  <div class="grid two">
    <div><h3>Caveats</h3>{caveats}</div>
    <div><h3>Next Recommended Check</h3><p>{html.escape(section.get('next_check') or 'Inspect the linked source artifacts.')}</p></div>
  </div>
</section>
"""


def _claim_dashboard_html(checklist: Dict[str, Any], data: Dict[str, Any], *, from_page: bool = False) -> str:
    rows = []
    for key, rec in checklist.items():
        evidence = rec.get("evidence") or []
        evidence_links = ", ".join(_artifact_link(p, data["output_dir"], from_page=from_page) for p in evidence[:3] if p) or "none"
        inspect = rec.get("missing") or [rec.get("headline", "")]
        rows.append(
            "<tr>"
            f"<td>{html.escape(CLAIM_NAMES.get(key, key))}</td>"
            f"<td><span class=\"status {html.escape(rec.get('status', 'missing'))}\">{html.escape(rec.get('status', 'missing'))}</span></td>"
            f"<td>{html.escape(rec.get('headline', ''))}<br><span class=\"muted\">score={_fmt(rec.get('score'), 2)}</span></td>"
            f"<td>{evidence_links}</td>"
            f"<td>{html.escape('; '.join(str(x) for x in inspect[:2]))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap claim-table"><table><thead><tr>'
        "<th>Claim</th><th>Status</th><th>Evidence Summary</th><th>Evidence</th><th>What To Inspect</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _run_metadata_html(data: Dict[str, Any]) -> str:
    manifest = data.get("manifest") or {}
    model_info = data.get("model_info") or {}
    eval_data = data.get("eval") or {}
    rows = [
        ("model version", manifest.get("model_version") or data.get("config", {}).get("model", {}).get("model_version") or "unknown"),
        ("checkpoint", manifest.get("checkpoint_path") or model_info.get("checkpoint_path_resolved") or "unknown"),
        ("checkpoint step", manifest.get("checkpoint_step") or eval_data.get("checkpoint_step") or model_info.get("checkpoint_step") or "unknown"),
        ("params", model_info.get("param_count") or manifest.get("param_count") or "unknown"),
        ("val tokens", eval_data.get("valid_tokens") or "missing"),
        ("analysis output path", data["output_dir"]),
        ("report generated time", data["generated_at"]),
    ]
    return '<div class="meta-card">' + "".join(
        f'<div><span>{html.escape(k)}</span><b>{html.escape(_fmt(v) if isinstance(v, (int, float)) else str(v))}</b></div>'
        for k, v in rows
    ) + "</div>"


def _build_sections(data: Dict[str, Any], figures: Dict[str, Any], checklist: Dict[str, Any]) -> List[Dict[str, Any]]:
    inv = data["inventory"]
    eval_data = data["eval"]
    prune_rows = data["prune_rows"]
    geom_rows = data["geometry_rows"]
    usage_rows = data["usage_rows"]
    ablation_rows = data["ablation_rows"]
    trace_prompts = data["trace_prompts"]
    baseline = data["baseline_eval"]

    performance_headline = (
        f"Final val_loss={_fmt(eval_data.get('val_loss'))}, ppl={_fmt(eval_data.get('ppl'))}, accuracy={_fmt(eval_data.get('accuracy'))}."
        if eval_data else
        "Final evaluation is missing."
    )
    prune_headline = "Prune curve missing."
    if prune_rows:
        base = next((r for r in prune_rows if _as_float(r.get("eps")) == 0.0), prune_rows[0])
        prune_headline = f"eps=0 compute fraction={_fmt(base.get('estimated_compute_frac'))}; {len(prune_rows)} prune point(s) discovered."
    geom_headline = "Geometry summary missing."
    if geom_rows:
        geom_headline = "; ".join(f"{r['pool'].upper()} rank={_fmt(r.get('effective_rank'))}" for r in geom_rows)
    usage_headline = "Usage atlas missing."
    if usage_rows:
        usage_headline = "; ".join(f"{r['pool'].upper()} active={_fmt(r.get('active_ops'))}/{_fmt(r.get('total_ops'))}" for r in usage_rows)
    ablation_headline = "Ablation results missing."
    if ablation_rows:
        best = max(ablation_rows, key=lambda r: _as_float(r.get("delta_loss"), float("-inf")))
        ablation_headline = f"Strongest ablation: {best.get('pool')}/{best.get('strategy')}/k={best.get('k')} delta_loss={_fmt(best.get('delta_loss'))}."
    trace_headline = f"{len(trace_prompts)} prompt trace(s) discovered." if trace_prompts else "Token-level traces missing."

    comp_strategies = {str(r.get("strategy")) for r in ablation_rows}
    return [
        {
            "id": "performance",
            "title": "Section 1. Performance",
            "headline": performance_headline,
            "findings": [
                checklist["performance_retained"]["headline"],
                "This section answers whether the checkpoint has enough standalone and matched-baseline evidence for performance claims.",
            ],
            "interpretation": interpret_performance(eval_data, baseline),
            "caveats": [
                "Performance parity is not supported without a baseline eval artifact.",
                "Small-token smoke runs can validate plumbing but not paper-scale performance.",
            ],
            "next_check": "Provide --baseline-analysis-output or --baseline-eval-json for matched parity, then inspect loss gap and token count.",
            "figures": [figures.get("performance_summary")],
            "table_rows": data["performance_rows"],
            "table_fields": PERFORMANCE_FIELDS,
            "artifacts": [inv.get("eval", {}).get("final_eval"), data.get("baseline_path")],
        },
        {
            "id": "prune",
            "title": "Section 2. Sparse Execution / Prune Curve",
            "headline": prune_headline,
            "findings": [
                checklist["sparse_execution_works"]["headline"],
                "The loss-vs-compute curve is the primary evidence for meaningful sparse execution.",
            ],
            "interpretation": interpret_prune(prune_rows),
            "caveats": [
                "If compute fraction is missing, epsilon alone cannot quantify savings.",
                "A nearest-loss eps=0 point means extra hard pruning has not yet been shown robust.",
            ],
            "next_check": "Inspect prune_curve_loss_vs_compute.png and rerun prune with more eps values around the best low-loss point.",
            "figures": [figures.get("prune_curve_loss_vs_compute"), figures.get("prune_curve_eps_vs_loss")],
            "table_rows": prune_rows,
            "table_fields": PRUNE_FIELDS,
            "artifacts": [inv.get("prune", {}).get("summary"), inv.get("prune", {}).get("curve")],
        },
        {
            "id": "geometry",
            "title": "Section 3. Operator Geometry",
            "headline": geom_headline,
            "findings": [
                checklist["operators_non_collapsed"]["headline"],
                "Effective rank and cosine statistics test whether operator-key geometry remains broad or collapses.",
            ],
            "interpretation": interpret_geometry(geom_rows),
            "caveats": [
                "High max cosine can coexist with high rank; it points to local duplicates, not necessarily full collapse.",
                "Rank thresholds depend on d_route and sample size.",
            ],
            "next_check": "Inspect high max-cosine pools and compare with norm histograms if duplicates look severe.",
            "figures": [figures.get("geometry_rank_cosine")],
            "table_rows": geom_rows,
            "table_fields": GEOMETRY_FIELDS,
            "artifacts": [inv.get("geometry", {}).get("summary"), inv.get("geometry", {}).get("operator_norm_histograms")],
        },
        {
            "id": "usage",
            "title": "Section 4. Operator Usage Atlas",
            "headline": usage_headline,
            "findings": [
                checklist["usage_is_structured"]["headline"],
                "Active fraction, total top-k mass, and card contexts show whether usage is broad, concentrated, or pool-specific.",
            ],
            "interpretation": interpret_usage(usage_rows),
            "caveats": [
                "The atlas is based on compact top-k summaries, not full gate tensors.",
                "Usage evidence should not be auto-labeled semantically; inspect contexts directly.",
            ],
            "next_check": "Open operator_cards.html and compare top-mass contexts across QK, V, and RST.",
            "figures": [figures.get("usage_active_ops_by_pool"), figures.get("usage_mass_by_pool")],
            "table_rows": usage_rows,
            "table_fields": USAGE_FIELDS,
            "artifacts": [inv.get("usage", {}).get("summary"), inv.get("usage", {}).get("operator_cards")],
        },
        {
            "id": "operator_cards",
            "title": "Section 5. Operator Cards",
            "headline": f"{len(data['operator_card_rows'])} top operator card row(s) rendered.",
            "findings": [
                "Operator cards present evidence: mass, activation frequency, contexts, and any matched ablation effect.",
                "No semantic labels are assigned automatically.",
            ],
            "interpretation": [
                "Cards are the right place to inspect whether high-usage operators fire on coherent local contexts.",
                "A card with high mass and positive ablation delta is stronger evidence than usage alone.",
            ],
            "caveats": [
                "Cards are selected examples and should not be treated as a complete ontology.",
                "Context windows can be tokenizer-id based when text decoding is unavailable.",
            ],
            "next_check": "Inspect several top cards per pool and compare any ablation_delta_loss_if_available values.",
            "figures": [],
            "table_rows": data["operator_card_rows"],
            "table_fields": OPERATOR_CARD_FIELDS,
            "table_limit": 20,
            "artifacts": [inv.get("usage", {}).get("operator_cards"), inv.get("usage", {}).get("operator_cards_top")],
        },
        {
            "id": "traces",
            "title": "Section 6. Token-Level Traces",
            "headline": trace_headline,
            "findings": [
                checklist["token_traces_available"]["headline"],
                "Trace heatmaps summarize compact top-k route activity by layer and token position.",
            ],
            "interpretation": [
                "Token traces are most useful for debugging where sparse activity concentrates across depth.",
                "Prompt-level summaries show active counts, gate mass, and top-1 concentration by pool when available.",
            ],
            "caveats": [
                "Trace figures intentionally avoid loading huge full gate dumps.",
                "Only the configured report-max trace prompts are plotted.",
            ],
            "next_check": "Open traces.html and inspect prompts whose heatmaps show unusually concentrated or empty activity.",
            "figures": figures.get("trace_gallery", [])[:3] if isinstance(figures.get("trace_gallery"), list) else [],
            "table_rows": data["trace_rows"],
            "table_fields": TRACE_FIELDS,
            "artifacts": [inv.get("trace", {}).get("summary"), *(inv.get("trace", {}).get("prompt_jsons") or [])[:3]],
        },
        {
            "id": "ablation",
            "title": "Section 7. Causal Ablation",
            "headline": ablation_headline,
            "findings": [
                checklist["operators_are_causal"]["headline"],
                "Top-use ablations support a causal claim only when they increase loss and comparators are smaller.",
            ],
            "interpretation": interpret_ablation(ablation_rows),
            "caveats": [
                "Missing random/low-use comparators weaken causal interpretation." if not ({"random", "low", "low-use"} & comp_strategies) else "Comparator strategies are present; still inspect k and pool coverage.",
                "Small ablation budgets can be noisy.",
            ],
            "next_check": "Run ablation with strategies top,random,low if comparator rows are absent, then compare delta_loss curves.",
            "figures": [figures.get("ablation_delta_loss_by_pool_strategy"), figures.get("ablation_curve_top_vs_random")],
            "table_rows": ablation_rows,
            "table_fields": ABLATION_FIELDS,
            "artifacts": [inv.get("ablation", {}).get("summary"), inv.get("ablation", {}).get("curve")],
        },
        {
            "id": "patching",
            "title": "Section 8. Patching / Steering",
            "headline": checklist["patching_steering_available"]["headline"],
            "findings": [
                "Patching/steering is marked supported only when actual result files are present.",
                "Empty scaffold directories are reported as missing evidence.",
            ],
            "interpretation": [
                checklist["patching_steering_available"]["headline"],
            ],
            "caveats": [
                "Scaffolding alone is not evidence for paper claims.",
                "This report does not fabricate patching or steering conclusions from other stages.",
            ],
            "next_check": "Run patching/steering analyses and regenerate the report to populate this section.",
            "figures": [],
            "table_rows": [{"artifact": p} for p in (inv.get("patching", {}).get("results") or []) + (inv.get("steering", {}).get("results") or [])],
            "table_fields": ["artifact"],
            "artifacts": (inv.get("patching", {}).get("results") or [])[:5] + (inv.get("steering", {}).get("results") or [])[:5],
        },
    ]


def _missing_notes(data: Dict[str, Any], checklist: Dict[str, Any]) -> List[str]:
    notes = []
    if not data.get("baseline_eval"):
        notes.append("Baseline comparison missing.")
    strategies = {str(r.get("strategy")) for r in data.get("ablation_rows", [])}
    if "random" not in strategies:
        notes.append("Random ablation comparator missing.")
    if not ({"low", "low-use"} & strategies):
        notes.append("Low-use ablation comparator missing.")
    if not data["inventory"].get("patching", {}).get("results"):
        notes.append("Patching directory has no result files.")
    if not data["inventory"].get("steering", {}).get("results"):
        notes.append("Steering directory has no result files.")
    for key, rec in checklist.items():
        if rec.get("status") != "supported":
            notes.append(f"{CLAIM_NAMES.get(key, key)} is {rec.get('status')}: {rec.get('headline')}")
    return list(dict.fromkeys(notes))


def _css() -> str:
    return """
:root { color-scheme: light; --ink:#111827; --muted:#64748b; --line:#d8dee9; --soft:#f8fafc; --blue:#2563eb; --green:#16a34a; --amber:#d97706; --red:#dc2626; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:#ffffff; line-height:1.5; }
header { padding:32px 40px 24px; background:#f1f5f9; border-bottom:1px solid var(--line); }
main { max-width:1240px; margin:0 auto; padding:28px 24px 56px; }
h1 { margin:0 0 8px; font-size:34px; line-height:1.12; letter-spacing:0; }
h2 { margin:0; font-size:24px; letter-spacing:0; }
h3 { margin:18px 0 8px; font-size:16px; letter-spacing:0; }
p { margin:8px 0; }
a { color:var(--blue); text-decoration:none; }
a:hover { text-decoration:underline; }
nav { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
nav a { padding:6px 9px; border:1px solid var(--line); border-radius:8px; background:#fff; font-size:13px; }
section { padding:26px 0; border-bottom:1px solid var(--line); }
.section-head { margin-bottom:14px; }
.headline { color:#334155; font-size:17px; margin-top:6px; }
.muted { color:var(--muted); font-size:13px; }
.meta-card { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin-top:18px; }
.meta-card div, .warning-card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; }
.meta-card span { display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.meta-card b { display:block; overflow-wrap:anywhere; font-size:14px; }
.grid.two { display:grid; grid-template-columns:minmax(0,1fr) minmax(280px,0.95fr); gap:24px; align-items:start; }
figure { margin:0 0 12px; border:1px solid var(--line); border-radius:8px; padding:8px; background:#fff; }
img { width:100%; height:auto; display:block; }
.table-wrap { overflow:auto; border:1px solid var(--line); border-radius:8px; margin:14px 0; }
table { width:100%; border-collapse:collapse; font-size:13px; background:#fff; }
th, td { padding:8px 10px; border-bottom:1px solid #e5e7eb; vertical-align:top; text-align:left; overflow-wrap:anywhere; }
th { position:sticky; top:0; background:#f8fafc; z-index:1; }
.status { display:inline-block; padding:3px 8px; border-radius:999px; font-weight:700; font-size:12px; }
.status.supported { color:#14532d; background:#dcfce7; }
.status.partial { color:#713f12; background:#fef3c7; }
.status.missing { color:#7f1d1d; background:#fee2e2; }
.artifact-list ul { margin-top:6px; }
.operator-card { border:1px solid var(--line); border-radius:8px; padding:12px; margin:10px 0; background:#fff; }
.operator-card h3 { margin-top:0; }
code { background:#f1f5f9; padding:1px 4px; border-radius:4px; }
@media (max-width: 820px) { header { padding:24px 18px; } main { padding:20px 14px 40px; } .grid.two { grid-template-columns:1fr; } h1 { font-size:28px; } }
"""


def _html_doc(title: str, body: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_css()}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )


def render_html_report(report_data, output_dir):
    """Render the primary static HTML report and companion pages."""
    data = report_data["data"]
    checklist = report_data["claim_checklist"]
    sections = report_data["sections"]
    figures = report_data["figures"]
    missing_notes = report_data["missing_notes"]
    report_dir = join_path(output_dir, "report")
    pages_dir = join_path(report_dir, "pages")
    makedirs(pages_dir)

    nav_links = [
        ("Performance", "#performance"),
        ("Prune", "#prune"),
        ("Geometry", "#geometry"),
        ("Usage", "#usage"),
        ("Operator Cards", "#operator_cards"),
        ("Traces", "#traces"),
        ("Ablation", "#ablation"),
        ("Patching", "#patching"),
        ("Missing", "#missing"),
        ("Appendix", "#appendix"),
    ]
    nav = "<nav>" + "".join(f'<a href="{href}">{label}</a>' for label, href in nav_links) + "</nav>"
    warning_cards = ""
    if missing_notes:
        warning_cards = (
            '<section id="missing"><h2>Section 9. Limitations And Missing Artifacts</h2>'
            + "".join(f'<div class="warning-card">{html.escape(note)}</div>' for note in missing_notes[:40])
            + '<p><a href="pages/missing_artifacts.html">Open full missing artifact page</a></p></section>'
        )
    appendix = _appendix_html(data, output_dir, from_page=False)
    body = f"""
<header>
  <h1>{html.escape(data.get('model_label', 'DAWN-SRW'))} Analysis Report</h1>
  <p class="headline">Complete static report bundle generated from discovered analysis artifacts.</p>
  {_run_metadata_html(data)}
  {nav}
</header>
<main>
  <section id="claims">
    <h2>Claim Dashboard</h2>
    {_claim_dashboard_html(checklist, data)}
  </section>
  {''.join(_section_html(section, data) for section in sections)}
  {warning_cards}
  {appendix}
</main>
"""
    write_text_atomic(
        join_path(report_dir, "index.html"),
        _html_doc(f"{data.get('model_label', 'DAWN-SRW')} Analysis Report", body),
    )

    page_map = {
        "performance.html": "performance",
        "prune.html": "prune",
        "geometry.html": "geometry",
        "usage.html": "usage",
        "operator_cards.html": "operator_cards",
        "traces.html": "traces",
        "ablation.html": "ablation",
    }
    for filename, section_id in page_map.items():
        section = next((s for s in sections if s["id"] == section_id), None)
        if not section:
            continue
        extra = ""
        if section_id == "operator_cards":
            extra = _operator_cards_detail_html(data, from_page=True)
        elif section_id == "traces":
            extra = _trace_detail_html(data, figures, from_page=True)
        page_body = f"""
<header><h1>{html.escape(section['title'])}</h1><p><a href="../index.html">Back to index</a></p></header>
<main>{_section_html(section, data, from_page=True)}{extra}</main>
"""
        write_text_atomic(join_path(pages_dir, filename), _html_doc(section["title"], page_body))

    missing_body = f"""
<header><h1>Missing / Invalid Artifacts</h1><p><a href="../index.html">Back to index</a></p></header>
<main>
  <section><h2>Missing Or Partial Evidence</h2>
  {''.join(f'<div class="warning-card">{html.escape(note)}</div>' for note in missing_notes)}
  </section>
  <section><h2>Exact Missing Paths</h2>
  {_table_preview(data['inventory'].get('missing', []), ['label', 'path'], limit=200, output_dir=output_dir, from_page=True)}
  </section>
  <section><h2>Invalid / Corrupt Files</h2>
  {_table_preview(data.get('errors', []), ['type', 'path', 'error'], limit=200, output_dir=output_dir, from_page=True)}
  </section>
</main>
"""
    write_text_atomic(join_path(pages_dir, "missing_artifacts.html"), _html_doc("Missing Artifacts", missing_body))


def _appendix_html(data: Dict[str, Any], output_dir: str, *, from_page: bool = False) -> str:
    table_specs = [
        ("Performance Table", data["performance_rows"], PERFORMANCE_FIELDS),
        ("Prune Curve", data["prune_rows"], PRUNE_FIELDS),
        ("Geometry Table", data["geometry_rows"], GEOMETRY_FIELDS),
        ("Usage Summary", data["usage_rows"], USAGE_FIELDS),
        ("Ablation Table", data["ablation_rows"], ABLATION_FIELDS),
        ("Operator Cards Top", data["operator_card_rows"], OPERATOR_CARD_FIELDS),
        ("Trace Prompt Index", data["trace_rows"], TRACE_FIELDS),
    ]
    return (
        '<section id="appendix"><h2>Appendix: All Tables</h2>'
        + "".join(
            f"<h3>{html.escape(title)}</h3>{_table_preview(rows, fields, limit=50, output_dir=output_dir, from_page=from_page)}"
            for title, rows, fields in table_specs
        )
        + "</section>"
    )


def _operator_cards_detail_html(data: Dict[str, Any], *, from_page: bool = False) -> str:
    rows = data["operator_card_rows"]
    if not rows:
        return '<section><h2>Card Details</h2><p class="muted">No operator cards available.</p></section>'
    cards = []
    for row in rows:
        contexts = [row.get("top_context_1"), row.get("top_context_2"), row.get("top_context_3")]
        ctx_items = "".join(f"<li>{html.escape(str(c))}</li>" for c in contexts if c)
        cards.append(f"""
<div class="operator-card">
  <h3>{html.escape(str(row.get('pool')).upper())} operator #{html.escape(str(row.get('operator_id')))}</h3>
  <p><b>Total mass:</b> {_fmt(row.get('total_mass'))}</p>
  <p><b>Activation frequency:</b> {_fmt(row.get('activation_frequency'))}</p>
  <h4>Top contexts</h4>
  <ol>{ctx_items or '<li class="muted">No context rows available.</li>'}</ol>
  <p><b>Causal effect:</b> delta_loss={html.escape(str(row.get('ablation_delta_loss_if_available') or 'missing'))}</p>
</div>
""")
    return '<section><h2>Card Details</h2>' + "".join(cards) + "</section>"


def _trace_detail_html(data: Dict[str, Any], figures: Dict[str, Any], *, from_page: bool = False) -> str:
    output_dir = data["output_dir"]
    if not data["trace_prompts"]:
        return '<section><h2>Trace Gallery</h2><p class="muted">No traces available.</p></section>'
    blocks = []
    for rec in data["trace_prompts"]:
        summary = rec.get("summary") or {}
        summary_rows = [
            {"pool": pool, **vals}
            for pool, vals in summary.items()
            if isinstance(vals, dict)
        ]
        token_preview = rec.get("token_ids") or "[]"
        blocks.append(f"""
<section>
  <h2>{html.escape(str(rec.get('prompt_id')))}</h2>
  <p>{html.escape(str(rec.get('text') or ''))}</p>
  <p class="muted">length={_fmt(rec.get('length'))} tokens={html.escape(str(token_preview)[:800])}</p>
  {_figure_html(rec.get('heatmap_figure'), output_dir, from_page=from_page, alt=str(rec.get('prompt_id')))}
  <h3>Pool-Level Summary</h3>
  {_table_preview(summary_rows, ['pool', 'active_mean', 'active_max', 'mass_mean', 'top1_frac_mean'], output_dir=output_dir, from_page=from_page)}
  <h3>Artifacts</h3>
  <ul>
    <li>{_artifact_link(rec.get('json_artifact'), output_dir, from_page=from_page)}</li>
    <li>{_artifact_link(rec.get('heatmap_csv'), output_dir, from_page=from_page)}</li>
    <li>{_artifact_link(rec.get('topk_npz'), output_dir, from_page=from_page)}</li>
  </ul>
</section>
""")
    return '<section><h2>Trace Gallery</h2></section>' + "".join(blocks)


def _markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str], limit: Optional[int] = None) -> str:
    rows = list(rows if limit is None else rows[:limit])
    fields = list(fields)
    if not fields:
        return ""
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        vals = []
        for field in fields:
            val = row.get(field, "")
            if isinstance(val, (float, np.floating)):
                val = _fmt(float(val))
            text = str(val).replace("\n", " ").replace("|", "\\|")
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    if not rows:
        lines.append("| " + " | ".join("" for _ in fields) + " |")
    return "\n".join(lines)


def _render_summary_md(data: Dict[str, Any], checklist: Dict[str, Any], sections: Sequence[Dict[str, Any]],
                       missing_notes: Sequence[str], output_dir: str) -> str:
    lines = [
        f"# {data.get('model_label', 'DAWN-SRW')} Analysis Summary",
        "",
        f"Generated: {data['generated_at']}",
        f"Output: `{data['output_dir']}`",
        "",
        "## Headline Findings",
        "",
    ]
    for section in sections[:7]:
        lines.append(f"- **{section['title']}**: {section['headline']}")
    lines.extend(["", "## Paper Claim Status", ""])
    lines.append("| Claim | Status | Score | Headline |")
    lines.append("| --- | --- | --- | --- |")
    for key, rec in checklist.items():
        lines.append(
            f"| {CLAIM_NAMES.get(key, key)} | {rec.get('status')} | {_fmt(rec.get('score'), 2)} | {str(rec.get('headline', '')).replace('|', '/')} |"
        )
    lines.extend(["", "## Most Important Missing Items", ""])
    if missing_notes:
        for note in missing_notes[:12]:
            lines.append(f"- {note}")
    else:
        lines.append("- No missing claim-critical items were detected.")
    lines.extend(["", "## Primary Files", ""])
    for rel in ("index.html", "full_report.md", "claim_checklist.json", "artifact_inventory.json", "missing_or_invalid_artifacts.json"):
        path = join_path(output_dir, "report", rel)
        lines.append(f"- {_md_link(path, output_dir, rel)}")
    lines.append("")
    return "\n".join(lines)


def _render_full_md(data: Dict[str, Any], checklist: Dict[str, Any], sections: Sequence[Dict[str, Any]],
                    figures: Dict[str, Any], missing_notes: Sequence[str], output_dir: str) -> str:
    lines = [
        f"# {data.get('model_label', 'DAWN-SRW')} Full Analysis Report",
        "",
        f"Generated: {data['generated_at']}",
        f"Output: `{data['output_dir']}`",
        "",
        "## Claim Dashboard",
        "",
        "| Claim | Status | Score | Evidence | Missing / Next Inspect |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, rec in checklist.items():
        evidence = ", ".join(_md_link(p, output_dir) for p in (rec.get("evidence") or [])[:3] if p) or "none"
        missing = "; ".join(str(x) for x in (rec.get("missing") or []))
        lines.append(
            f"| {CLAIM_NAMES.get(key, key)} | {rec.get('status')} | {_fmt(rec.get('score'), 2)} | {evidence} | {missing.replace('|', '/')} |"
        )
    for section in sections:
        lines.extend(["", f"## {section['title']}", "", f"**Headline:** {section['headline']}", ""])
        lines.append("**Key findings**")
        for item in section.get("findings", []):
            lines.append(f"- {item}")
        lines.extend(["", "**Interpretation**"])
        for item in section.get("interpretation", []):
            lines.append(f"- {item}")
        if section.get("figures"):
            lines.extend(["", "**Figures**"])
            for fig in section["figures"]:
                if fig:
                    lines.append(f"- {_md_link(fig, output_dir)}")
        if section.get("artifacts"):
            lines.extend(["", "**Source artifacts**"])
            for artifact in section["artifacts"]:
                if artifact:
                    lines.append(f"- {_md_link(artifact, output_dir)}")
        lines.extend(["", "**Caveats**"])
        for item in section.get("caveats", []):
            lines.append(f"- {item}")
        lines.extend(["", f"**Next recommended check:** {section.get('next_check')}", ""])
        if section.get("table_fields"):
            lines.append(_markdown_table(section.get("table_rows", []), section.get("table_fields", [])))
    lines.extend(["", "## Operator Card Excerpts", ""])
    for row in data["operator_card_rows"][:30]:
        lines.append(f"### {str(row.get('pool')).upper()} operator #{row.get('operator_id')}")
        lines.append(f"- total_mass={_fmt(row.get('total_mass'))}, activation_frequency={_fmt(row.get('activation_frequency'))}")
        for idx in range(1, 4):
            ctx = row.get(f"top_context_{idx}")
            if ctx:
                lines.append(f"- context {idx}: {ctx}")
        if row.get("ablation_delta_loss_if_available") != "":
            lines.append(f"- ablation_delta_loss_if_available={row.get('ablation_delta_loss_if_available')}")
        lines.append("")
    lines.extend(["", "## Trace Prompt Summaries", ""])
    for rec in data["trace_prompts"]:
        lines.append(f"### {rec.get('prompt_id')}")
        lines.append(f"- length={rec.get('length')}")
        if rec.get("heatmap_figure"):
            lines.append(f"- heatmap={_md_link(rec.get('heatmap_figure'), output_dir)}")
        for pool, vals in (rec.get("summary") or {}).items():
            if isinstance(vals, dict):
                lines.append(f"- {pool}: active_mean={_fmt(vals.get('active_mean'))}, mass_mean={_fmt(vals.get('mass_mean'))}, top1_frac_mean={_fmt(vals.get('top1_frac_mean'))}")
        lines.append("")
    lines.extend(["", "## Ablation Job Summaries", ""])
    lines.append(_markdown_table(data["ablation_rows"], ABLATION_FIELDS))
    lines.extend(["", "## Missing / Incomplete", ""])
    for note in missing_notes:
        lines.append(f"- {note}")
    lines.extend(["", "## Appendix: All Tables", ""])
    for title, rows, fields in (
        ("Performance Table", data["performance_rows"], PERFORMANCE_FIELDS),
        ("Prune Curve", data["prune_rows"], PRUNE_FIELDS),
        ("Geometry Table", data["geometry_rows"], GEOMETRY_FIELDS),
        ("Usage Summary", data["usage_rows"], USAGE_FIELDS),
        ("Ablation Table", data["ablation_rows"], ABLATION_FIELDS),
        ("Operator Cards Top", data["operator_card_rows"], OPERATOR_CARD_FIELDS),
        ("Trace Prompt Index", data["trace_rows"], TRACE_FIELDS),
    ):
        lines.extend(["", f"### {title}", "", _markdown_table(rows, fields)])
    lines.append("")
    return "\n".join(lines)


def _write_tables(data: Dict[str, Any], store) -> Dict[str, str]:
    table_dir = store.path("report", "tables")
    makedirs(table_dir)
    specs = {
        "performance_table": ("performance_table.csv", data["performance_rows"], PERFORMANCE_FIELDS),
        "prune_curve": ("prune_curve.csv", data["prune_rows"], PRUNE_FIELDS),
        "geometry_table": ("geometry_table.csv", data["geometry_rows"], GEOMETRY_FIELDS),
        "usage_summary": ("usage_summary.csv", data["usage_rows"], USAGE_FIELDS),
        "ablation_table": ("ablation_table.csv", data["ablation_rows"], ABLATION_FIELDS),
        "operator_cards_top": ("operator_cards_top.csv", data["operator_card_rows"], OPERATOR_CARD_FIELDS),
        "trace_prompt_index": ("trace_prompt_index.csv", data["trace_rows"], TRACE_FIELDS),
    }
    paths = {}
    for key, (filename, rows, fields) in specs.items():
        path = join_path(table_dir, filename)
        write_csv_atomic(path, rows, fieldnames=fields)
        paths[key] = path
    return paths


def _write_operator_cards_markdown(data: Dict[str, Any], store) -> str:
    path = store.path("report", "operator_cards_top.md")
    lines = ["# Operator Cards Top", ""]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in data["operator_card_rows"]:
        grouped[str(row.get("pool"))].append(row)
    for pool in sorted(grouped):
        lines.extend(["", f"## {pool.upper()}", ""])
        for row in grouped[pool]:
            lines.append(f"### Operator #{row.get('operator_id')} (rank {row.get('usage_rank')})")
            lines.append(f"- total_mass={_fmt(row.get('total_mass'))}")
            lines.append(f"- activation_frequency={_fmt(row.get('activation_frequency'))}")
            for idx in range(1, 4):
                ctx = row.get(f"top_context_{idx}")
                if ctx:
                    lines.append(f"- context {idx}: {ctx}")
            effect = row.get("ablation_delta_loss_if_available")
            lines.append(f"- causal effect: delta_loss={effect if effect != '' else 'missing'}")
            lines.append("")
    write_text_atomic(path, "\n".join(lines))
    return path


def _data_index(data: Dict[str, Any], figures: Dict[str, Any], tables: Dict[str, str], pages: Dict[str, str],
                checklist: Dict[str, Any], missing_notes: Sequence[str]) -> Dict[str, Any]:
    return {
        "generated_at": data["generated_at"],
        "output_dir": data["output_dir"],
        "primary_output": join_path(data["output_dir"], "report", "index.html"),
        "figures": figures,
        "tables": tables,
        "pages": pages,
        "claim_checklist": checklist,
        "source_artifacts": data["inventory"],
        "missing_or_partial_notes": list(missing_notes),
    }


def run_report_stage(ctx: AnalysisContext | None, store=None, args: Any = None) -> Dict[str, Any]:
    if ctx is not None:
        store = ctx.store
        args = ctx.args
    if store is None:
        raise ValueError("run_report_stage requires a context or store.")
    if args is None:
        class _Args:
            report_format = "html,md"
            report_max_operator_cards = 100
            report_max_trace_figures = 12
            report_include_appendix = True
            baseline_analysis_output = None
            baseline_eval_json = None
        args = _Args()

    stage = "report"
    store.set_stage_status(stage, "running")
    report_dir = store.path("report")
    for rel in ("", "figures", "tables", "pages"):
        makedirs(join_path(report_dir, rel))

    errors: List[Dict[str, Any]] = []
    inventory = discover_artifacts(store.output_dir)
    data = _build_report_data(store, args, inventory, errors)
    tables = _write_tables(data, store)
    figures = _generate_figures(data, store, args)
    checklist = _build_claim_checklist(data, figures)
    sections = _build_sections(data, figures, checklist)
    missing_notes = _missing_notes(data, checklist)

    report_bundle = {
        "data": data,
        "figures": figures,
        "tables": tables,
        "claim_checklist": checklist,
        "sections": sections,
        "missing_notes": missing_notes,
    }
    formats = {x.strip().lower() for x in str(getattr(args, "report_format", "html,md")).split(",") if x.strip()}
    if not formats:
        formats = {"html", "md"}
    if "html" in formats:
        render_html_report(report_bundle, store.output_dir)
    if "md" in formats:
        write_text_atomic(
            store.path("report", "summary.md"),
            _render_summary_md(data, checklist, sections, missing_notes, store.output_dir),
        )
        write_text_atomic(
            store.path("report", "full_report.md"),
            _render_full_md(data, checklist, sections, figures, missing_notes, store.output_dir),
        )
    else:
        write_text_atomic(
            store.path("report", "summary.md"),
            _render_summary_md(data, checklist, sections, missing_notes, store.output_dir),
        )
    cards_md = _write_operator_cards_markdown(data, store)

    missing_payload = {
        "missing": inventory.get("missing", []),
        "invalid": errors,
        "notes": missing_notes,
    }
    pages = {
        "index": store.path("report", "index.html"),
        "performance": store.path("report", "pages", "performance.html"),
        "prune": store.path("report", "pages", "prune.html"),
        "geometry": store.path("report", "pages", "geometry.html"),
        "usage": store.path("report", "pages", "usage.html"),
        "operator_cards": store.path("report", "pages", "operator_cards.html"),
        "traces": store.path("report", "pages", "traces.html"),
        "ablation": store.path("report", "pages", "ablation.html"),
        "missing_artifacts": store.path("report", "pages", "missing_artifacts.html"),
    }
    write_json_atomic(store.path("report", "artifact_inventory.json"), inventory)
    write_json_atomic(store.path("report", "missing_or_invalid_artifacts.json"), missing_payload)
    write_json_atomic(store.path("report", "claim_checklist.json"), checklist)
    write_json_atomic(store.path("report", "data_index.json"), _data_index(data, figures, tables, pages, checklist, missing_notes))

    result = {
        "index_html": store.path("report", "index.html"),
        "summary_md": store.path("report", "summary.md"),
        "full_report_md": store.path("report", "full_report.md"),
        "claim_checklist": store.path("report", "claim_checklist.json"),
        "artifact_inventory": store.path("report", "artifact_inventory.json"),
        "missing_or_invalid": store.path("report", "missing_or_invalid_artifacts.json"),
        "operator_cards_md": cards_md,
        "figures": figures,
        "tables": tables,
        "supported_claims": sum(1 for rec in checklist.values() if rec.get("status") == "supported"),
        "total_claims": len(checklist),
    }
    store.mark_job_complete(stage, "report", result["index_html"], result)
    store.set_stage_status(stage, "complete")
    store.log_event(
        stage,
        "summary",
        message=(
            f"REPORT SUMMARY supported_claims={result['supported_claims']}/{result['total_claims']} "
            f"path={result['index_html']}"
        ),
        **result,
    )
    return result
