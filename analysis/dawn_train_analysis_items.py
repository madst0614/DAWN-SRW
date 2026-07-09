"""Train-analysis item registry and summary formatting.

This module owns the reusable item pool for the lightweight
``--train-analysis`` checkpoint-state analyzer. The main analyzer computes the
summary payload; this module decides which item ids are valid, how presets map
to items, and how each item renders in the text summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


DEFAULT_TRAIN_ANALYSIS_PRESET = "qk_closed"

TRAIN_ANALYSIS_ITEM_DEFS = {
    "target_ratio": {
        "title": "Target ratio",
        "measures": "target, active_tau, admission, effective, and effective/target by pool.",
        "summary": "TARGET_RATIO rows identify which pool is under/over target.",
        "requires": ("active",),
    },
    "layer_selectivity": {
        "title": "Layer selectivity",
        "measures": "per-layer active/admission/effective/top1 plus score/tau/margin layer distributions.",
        "summary": "LAYER_SELECTIVITY shows whether a pool is globally closed or layer-local, and why.",
        "requires": ("active",),
    },
    "prune_breakdown": {
        "title": "Pruned eval breakdown",
        "measures": "eps-wise loss delta, total compute, and pool-level compute/effective estimates.",
        "summary": "PRUNE_BREAKDOWN shows which pool loses compute as eps increases.",
        "requires": ("prune",),
    },
    "execution_profile": {
        "title": "Execution profile",
        "measures": "pool size, active/effective operator counts, top1 concentration, and effective ratios.",
        "summary": "EXECUTION_PROFILE translates sparse percentages into counts and collapse indicators.",
        "requires": ("active",),
    },
    "num_health": {
        "title": "Numerical health",
        "measures": "residual norms, q/k/v norms, attention logits, softmax entropy, and output norms.",
        "summary": "NUM_HEALTH catches scale drift before selector collapse shows up.",
        "requires": ("active",),
    },
    "prompt_trace": {
        "title": "Prompt trace",
        "measures": "prompt token/layer q/k/v/rst active counts, gate mass, top1, top operator ids, and output norms.",
        "summary": "PROMPT_TRACE shows v4166 route behavior on diagnostic prompts without generating new text.",
        "requires": ("prompt_trace",),
    },
    "prompt_decision": {
        "title": "Prompt decision",
        "measures": "prompt-level qk/v/rst activity balance, selector concentration, and RST-vs-attention norm ratio.",
        "summary": "PROMPT_DECISION flags prompt-specific route imbalance or collapse signatures.",
        "requires": ("prompt_trace",),
    },
    "generation_samples": {
        "title": "Generation samples",
        "measures": "v4166 KV-cache prompt continuations, generated token count, and decode throughput.",
        "summary": "GENERATION_SAMPLES gives a quick qualitative read on the current checkpoint.",
        "requires": ("generation",),
    },
    "decision_reason": {
        "title": "Decision reason",
        "measures": "explicit numeric guardrails behind the final keep/watch/change decision.",
        "summary": "Decision lines include thresholds such as eff/target and collapse checks.",
        "requires": ("active",),
    },
}

TRAIN_ANALYSIS_ITEM_ALIASES = {
    "per_layer_active": "layer_selectivity",
    "select_dist": "layer_selectivity",
    "concentration": "execution_profile",
    "exec_counts": "execution_profile",
    "trace": "prompt_trace",
    "prompt": "prompt_trace",
    "decision_probe": "prompt_decision",
    "generation": "generation_samples",
    "samples": "generation_samples",
}

V4166_1B_ITEMS = (
    "target_ratio",
    "layer_selectivity",
    "prune_breakdown",
    "execution_profile",
    "prompt_trace",
    "prompt_decision",
    "generation_samples",
    "decision_reason",
)

TRAIN_ANALYSIS_PRESETS = {
    "minimal": ("target_ratio", "prune_breakdown", "decision_reason"),
    "qk_closed": (
        "target_ratio",
        "layer_selectivity",
        "prune_breakdown",
        "execution_profile",
        "decision_reason",
    ),
    "compute": ("target_ratio", "prune_breakdown", "execution_profile", "decision_reason"),
    "health": ("target_ratio", "layer_selectivity", "num_health", "decision_reason"),
    "prompt_debug": ("target_ratio", "layer_selectivity", "prompt_trace", "prompt_decision", "decision_reason"),
    "sample": ("target_ratio", "generation_samples", "decision_reason"),
    "v4166_1b": V4166_1B_ITEMS,
    "deep": V4166_1B_ITEMS,
    "full": tuple(TRAIN_ANALYSIS_ITEM_DEFS.keys()),
}

TRAIN_ANALYSIS_PRESET_ALIASES = {
    "v4166-1b": "v4166_1b",
    "v4166-1p3b": "v4166_1b",
    "v4166-1p3b-c4-20b": "v4166_1b",
    "v4166-1p3b-c4-20b-v4-64": "v4166_1b",
}


@dataclass(frozen=True)
class TrainAnalysisFormatters:
    num: Callable[[Any, int], str]
    delta: Callable[[Any, int], str]
    pct: Callable[[Any, int], str]
    eps: Callable[[Any], str]
    safe_float: Callable[[Any], Optional[float]]


def canonical_train_analysis_item(item: str) -> str:
    key = str(item).strip().lower()
    return TRAIN_ANALYSIS_ITEM_ALIASES.get(key, key)


def parse_train_analysis_items(preset: Optional[str], items: Optional[str]) -> List[str]:
    if items:
        raw = [part.strip() for part in str(items).split(",") if part.strip()]
        selected = list(TRAIN_ANALYSIS_ITEM_DEFS.keys()) if raw == ["all"] else raw
    else:
        preset_key = str(preset or DEFAULT_TRAIN_ANALYSIS_PRESET).strip().lower()
        if preset_key == "all":
            preset_key = "full"
        preset_key = TRAIN_ANALYSIS_PRESET_ALIASES.get(preset_key, preset_key)
        if preset_key not in TRAIN_ANALYSIS_PRESETS:
            raise ValueError(
                f"Unknown train-analysis preset {preset!r}; "
                f"known={','.join(TRAIN_ANALYSIS_PRESETS)}"
            )
        selected = list(TRAIN_ANALYSIS_PRESETS[preset_key])

    canonical = [canonical_train_analysis_item(item) for item in selected]
    bad = [item for item in canonical if item not in TRAIN_ANALYSIS_ITEM_DEFS]
    if bad:
        raise ValueError(
            f"Unknown train-analysis item(s): {','.join(bad)}; "
            f"known={','.join(TRAIN_ANALYSIS_ITEM_DEFS)}"
        )
    dedup: List[str] = []
    for item in canonical:
        if item not in dedup:
            dedup.append(item)
    return dedup


def selected_item_catalog(items: List[str]) -> List[Dict[str, Any]]:
    return [
        {"id": item, **TRAIN_ANALYSIS_ITEM_DEFS[item]}
        for item in items
    ]


def train_analysis_required_sections(items: List[str]) -> List[str]:
    sections: List[str] = []
    for item in items:
        for section in TRAIN_ANALYSIS_ITEM_DEFS[item].get("requires", ()):
            if section not in sections:
                sections.append(section)
    return sections


def train_analysis_catalog_text() -> str:
    lines = [
        "DAWN-SRW train-analysis item pool",
        "Scope: train_analysis checkpoint-state items only. Full pipeline stages remain eval,prune,geometry,usage,trace,ablation,report.",
        "",
        "Presets:",
    ]
    for name, items in TRAIN_ANALYSIS_PRESETS.items():
        lines.append(f"  {name}: {','.join(items)}")
    lines.extend(["", "Aliases:"])
    for old, new in sorted(TRAIN_ANALYSIS_ITEM_ALIASES.items()):
        lines.append(f"  {old} -> {new}")
    lines.extend(["", "Preset aliases:"])
    for old, new in sorted(TRAIN_ANALYSIS_PRESET_ALIASES.items()):
        lines.append(f"  {old} -> {new}")
    lines.extend(["", "Items:"])
    for item_id, meta in TRAIN_ANALYSIS_ITEM_DEFS.items():
        lines.extend([
            f"  {item_id}: {meta['title']}",
            f"    requires: {','.join(meta.get('requires', ())) or 'none'}",
            f"    measures: {meta['measures']}",
            f"    summary : {meta['summary']}",
        ])
    return "\n".join(lines)


def _item_lines_target_ratio(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    active = summary["active_dynamics"]
    out = [
        "  TARGET_RATIO:",
        "  pool  target   active_tau admission effective eff/target status",
    ]
    for row in active.get("target_ratio", []):
        out.append(
            "  "
            f"{row.get('pool'):<5} "
            f"{fmt.num(row.get('target'), 3):<8} "
            f"{fmt.num(row.get('active_tau'), 3):<10} "
            f"{fmt.num(row.get('admission'), 3):<9} "
            f"{fmt.num(row.get('effective'), 3):<9} "
            f"{fmt.num(row.get('eff_target_ratio'), 2):<10} "
            f"{row.get('status', 'n/a')}"
        )
    return out


def _item_lines_layer_selectivity(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    active = summary["active_dynamics"]
    per_layer = active.get("per_layer_active", {})
    out = [
        "  PER_LAYER_SUMMARY:",
        "  pool  eff_min eff_p10 eff_mean eff_p90 eff_max dead closed",
    ]
    for pool in ("qk", "v", "rst"):
        row = per_layer.get("summary", {}).get(pool, {})
        out.append(
            "  "
            f"{pool:<5} "
            f"{fmt.num(row.get('min'), 3):<7} "
            f"{fmt.num(row.get('p10'), 3):<7} "
            f"{fmt.num(row.get('mean'), 3):<8} "
            f"{fmt.num(row.get('p90'), 3):<7} "
            f"{fmt.num(row.get('max'), 3):<7} "
            f"{str(row.get('dead_layers', 'n/a')):<4} "
            f"{row.get('closed_layers', 'n/a')}"
        )

    out.extend([
        "  PER_LAYER_ACTIVE:",
        "  lyr qk_act qk_adm qk_eff qk_top1 | v_act v_adm v_eff v_top1 | rst_act rst_adm rst_eff rst_top1",
    ])
    for row in per_layer.get("layers", []):
        out.append(
            "  "
            f"{int(row.get('layer', 0)):02d}  "
            f"{fmt.num(row.get('qk_active_tau'), 3):<6} "
            f"{fmt.num(row.get('qk_admission'), 3):<6} "
            f"{fmt.num(row.get('qk_effective'), 3):<6} "
            f"{fmt.num(row.get('qk_top1'), 3):<7} | "
            f"{fmt.num(row.get('v_active_tau'), 3):<5} "
            f"{fmt.num(row.get('v_admission'), 3):<5} "
            f"{fmt.num(row.get('v_effective'), 3):<5} "
            f"{fmt.num(row.get('v_top1'), 3):<6} | "
            f"{fmt.num(row.get('rst_active_tau'), 3):<7} "
            f"{fmt.num(row.get('rst_admission'), 3):<7} "
            f"{fmt.num(row.get('rst_effective'), 3):<7} "
            f"{fmt.num(row.get('rst_top1'), 3)}"
        )

    out.extend([
        "  SELECT_DIST:",
        "  pool score_p50 score_p90 score_max tau_p50 tau_p90 margin_p50 margin_p90 pos_margin_frac",
    ])
    for pool in ("qk", "v", "rst"):
        row = active.get("select_distribution", {}).get(pool, {})
        score = row.get("score_layer", {})
        tau = row.get("tau_layer", {})
        margin = row.get("margin_layer", {})
        out.append(
            "  "
            f"{pool:<5} "
            f"{fmt.num(score.get('p50'), 4):<9} "
            f"{fmt.num(score.get('p90'), 4):<9} "
            f"{fmt.num(row.get('score_max'), 4):<9} "
            f"{fmt.num(tau.get('p50'), 4):<7} "
            f"{fmt.num(tau.get('p90'), 4):<7} "
            f"{fmt.delta(margin.get('p50'), 4):<10} "
            f"{fmt.delta(margin.get('p90'), 4):<10} "
            f"{fmt.num(row.get('pos_margin_frac'), 3)}"
        )
    return out


def _item_lines_prune_breakdown(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    prune = summary["effective_prune"]
    baseline = prune.get("baseline") or {}
    prune_rows = ([baseline] if baseline else []) + list(prune.get("eps", []))
    base_compute = fmt.safe_float(baseline.get("compute_frac"))
    out = [
        "  PRUNE_BREAKDOWN:",
        "  eps   val_loss delta_ce total qk_cmp v_cmp rst_cmp | qk_eff v_eff rst_eff saved gate_mass no_active",
    ]
    if not prune_rows:
        out.append("  n/a")
    for row in prune_rows:
        compute = fmt.safe_float(row.get("compute_frac"))
        saved = None
        if base_compute is not None and base_compute > 0.0 and compute is not None:
            saved = (base_compute - compute) / base_compute
        out.append(
            "  "
            f"{fmt.eps(row.get('eps')):<5} "
            f"{fmt.num(row.get('val_loss'), 6):<8} "
            f"{fmt.delta(row.get('loss_delta'), 4):<8} "
            f"{fmt.num(row.get('compute_frac'), 4):<5} "
            f"{fmt.num(row.get('qk_compute'), 4):<6} "
            f"{fmt.num(row.get('v_compute'), 4):<5} "
            f"{fmt.num(row.get('rst_compute'), 4):<7} | "
            f"{fmt.num(row.get('qk_eff'), 3):<6} "
            f"{fmt.num(row.get('v_eff'), 3):<5} "
            f"{fmt.num(row.get('rst_eff'), 3):<7} "
            f"{fmt.pct(saved, 1):<5} "
            f"{fmt.num(row.get('gate_mass_retained'), 3):<8} "
            f"{fmt.num(row.get('no_active_frac'), 3)}"
        )
    return out


def _item_lines_execution_profile(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    pools = summary["active_dynamics"].get("pools", {})
    out = [
        "  EXECUTION_PROFILE:",
        "  pool pool_size active_frac active_ops effective_frac effective_ops top1 top1_max eff_ratio",
    ]
    for pool in ("qk", "v", "rst"):
        pdata = pools.get(pool, {})
        out.append(
            "  "
            f"{pool:<5} "
            f"{str(pdata.get('pool_size') or 'n/a'):<9} "
            f"{fmt.num(pdata.get('active_tau'), 3):<11} "
            f"{fmt.num(pdata.get('active_ops_mean'), 1):<10} "
            f"{fmt.num(pdata.get('effective'), 3):<14} "
            f"{fmt.num(pdata.get('effective_ops_mean'), 1):<13} "
            f"{fmt.num(pdata.get('top1'), 3):<5} "
            f"{fmt.num(pdata.get('top1_max'), 3):<8} "
            f"{fmt.num(pdata.get('effective_ops_ratio'), 3)}"
        )
    return out


def _item_lines_num_health(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    health = summary["active_dynamics"].get("num_health", {})
    out = [
        "  NUM_HEALTH:",
        "  metric                         mean     p90      max",
    ]
    for key in (
        "residual_norm",
        "residual_norm_max",
        "q_norm",
        "k_norm",
        "v_norm",
        "attn_logit_max",
        "attn_softmax_entropy_mean",
        "attn_o_output_norm_mean",
    ):
        row = health.get(key, {})
        out.append(
            "  "
            f"{key:<30} "
            f"{fmt.num(row.get('mean'), 3):<8} "
            f"{fmt.num(row.get('p90'), 3):<8} "
            f"{fmt.num(row.get('max'), 3)}"
        )
    out.append(f"  no_nan                         {health.get('no_nan')}")
    return out


def _item_lines_prompt_trace(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    trace = summary.get("prompt_trace", {})
    if trace.get("status") not in ("ready", "empty"):
        return ["  PROMPT_TRACE:", f"  status={trace.get('status', 'missing')} reason={trace.get('reason', 'n/a')}"]
    out = [
        "  PROMPT_TRACE:",
        "  prompt_id       len q_act k_act v_act rst_act q_top1 k_top1 v_top1 rst_top1 rst/attn",
    ]
    rows = trace.get("prompts", [])
    if not rows:
        out.append("  n/a")
    for row in rows:
        prompt_summary = row.get("summary", {})
        pools = prompt_summary.get("pools", {})
        out.append(
            "  "
            f"{str(row.get('prompt_id', 'n/a')):<15} "
            f"{str(row.get('length', 'n/a')):<3} "
            f"{fmt.num(pools.get('q', {}).get('active_mean'), 1):<5} "
            f"{fmt.num(pools.get('k', {}).get('active_mean'), 1):<5} "
            f"{fmt.num(pools.get('v', {}).get('active_mean'), 1):<5} "
            f"{fmt.num(pools.get('rst', {}).get('active_mean'), 1):<7} "
            f"{fmt.num(pools.get('q', {}).get('top1_max'), 3):<6} "
            f"{fmt.num(pools.get('k', {}).get('top1_max'), 3):<6} "
            f"{fmt.num(pools.get('v', {}).get('top1_max'), 3):<6} "
            f"{fmt.num(pools.get('rst', {}).get('top1_max'), 3):<8} "
            f"{fmt.num(prompt_summary.get('rst_attn_norm_ratio'), 3)}"
        )
    return out


def _item_lines_prompt_decision(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    decision = summary.get("prompt_decision", {})
    if decision.get("status") != "ready":
        return ["  PROMPT_DECISION:", f"  status={decision.get('status', 'missing')} reason={decision.get('reason', 'n/a')}"]
    out = [
        "  PROMPT_DECISION:",
        "  prompt_id       status qk_act v_act rst_act rst_top1 rst/attn reason",
    ]
    rows = decision.get("rows", [])
    if not rows:
        out.append("  n/a")
    for row in rows:
        out.append(
            "  "
            f"{str(row.get('prompt_id', 'n/a')):<15} "
            f"{str(row.get('status', 'n/a')):<6} "
            f"{fmt.num(row.get('qk_active_mean'), 1):<6} "
            f"{fmt.num(row.get('v_active_mean'), 1):<5} "
            f"{fmt.num(row.get('rst_active_mean'), 1):<7} "
            f"{fmt.num(row.get('rst_top1_max'), 3):<8} "
            f"{fmt.num(row.get('rst_attn_norm_ratio'), 3):<7} "
            f"{row.get('reason', 'n/a')}"
        )
    return out


def _item_lines_generation_samples(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    generation = summary.get("generation_samples", {})
    if generation.get("status") not in ("ready", "empty"):
        return ["  GENERATION_SAMPLES:", f"  status={generation.get('status', 'missing')} reason={generation.get('reason', 'n/a')}"]
    out = [
        "  GENERATION_SAMPLES:",
        "  prompt_id    new_tok tok/s continuation",
    ]
    rows = generation.get("samples", [])
    if not rows:
        out.append("  n/a")
    for row in rows:
        continuation = str(row.get("continuation") or "").replace("\n", " ").strip()
        if len(continuation) > 96:
            continuation = continuation[:93] + "..."
        out.append(
            "  "
            f"{str(row.get('prompt_id', 'n/a')):<12} "
            f"{str(row.get('new_tokens', 'n/a')):<7} "
            f"{fmt.num(row.get('tokens_per_sec'), 1):<5} "
            f"{continuation}"
        )
    return out


def _item_lines_decision_reason(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    return [f"  - {line_i}" for line_i in (summary.get("decision") or ["n/a"])]


TRAIN_ANALYSIS_ITEM_FORMATTERS = {
    "target_ratio": _item_lines_target_ratio,
    "layer_selectivity": _item_lines_layer_selectivity,
    "prune_breakdown": _item_lines_prune_breakdown,
    "execution_profile": _item_lines_execution_profile,
    "num_health": _item_lines_num_health,
    "prompt_trace": _item_lines_prompt_trace,
    "prompt_decision": _item_lines_prompt_decision,
    "generation_samples": _item_lines_generation_samples,
    "decision_reason": _item_lines_decision_reason,
}


def format_train_analysis_item(
    summary: Dict[str, Any],
    item: str,
    fmt: TrainAnalysisFormatters,
) -> List[str]:
    meta = TRAIN_ANALYSIS_ITEM_DEFS.get(item, {"title": item, "summary": "n/a"})
    formatter = TRAIN_ANALYSIS_ITEM_FORMATTERS.get(item)
    out = [
        "",
        f"ITEM {item}: {meta.get('title', item)}",
        f"  summary: {meta.get('summary', 'n/a')}",
    ]
    if formatter is None:
        out.append("  n/a")
    else:
        out.extend(formatter(summary, fmt))
    return out


def format_train_analysis_items(
    summary: Dict[str, Any],
    items: List[str],
    fmt: TrainAnalysisFormatters,
) -> List[str]:
    out = ["", "Analysis item summaries:"]
    for item in items:
        out.extend(format_train_analysis_item(summary, item, fmt))
    return out


def item_status(summary: Dict[str, Any], item: str) -> str:
    active = summary.get("active_dynamics", {})
    prune = summary.get("effective_prune", {})
    if item == "target_ratio":
        return "ready" if active.get("target_ratio") else "missing"
    if item == "layer_selectivity":
        if active.get("per_layer_active", {}).get("layers") and active.get("select_distribution"):
            return "ready"
        return "partial" if active.get("per_layer_active") or active.get("select_distribution") else "missing"
    if item == "prune_breakdown":
        return "ready" if (prune.get("baseline") or prune.get("eps")) else "missing"
    if item == "execution_profile":
        return "ready" if active.get("pools") else "missing"
    if item == "num_health":
        return "ready" if active.get("num_health") else "missing"
    if item == "prompt_trace":
        trace = summary.get("prompt_trace", {})
        return str(trace.get("status") or "missing")
    if item == "prompt_decision":
        decision = summary.get("prompt_decision", {})
        return str(decision.get("status") or "missing")
    if item == "generation_samples":
        generation = summary.get("generation_samples", {})
        return str(generation.get("status") or "missing")
    if item == "decision_reason":
        return "ready" if summary.get("decision") else "missing"
    return "unknown"


def emit_train_analysis_item_progress(summary: Dict[str, Any], items: List[str]) -> None:
    print(f"TRAIN_ANALYSIS ITEMS START count={len(items)}", flush=True)
    for idx, item in enumerate(items, start=1):
        meta = TRAIN_ANALYSIS_ITEM_DEFS.get(item, {})
        print(
            "TRAIN_ANALYSIS ITEM "
            f"{idx:02d}/{len(items):02d} id={item} "
            f"title={meta.get('title', item)!r} "
            f"status={item_status(summary, item)}",
            flush=True,
        )
    print("TRAIN_ANALYSIS ITEMS DONE", flush=True)
