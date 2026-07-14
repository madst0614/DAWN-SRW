"""Train-analysis item registry and summary formatting.

This module owns the reusable item pool for the lightweight
``--train-analysis`` checkpoint-state analyzer. The main analyzer computes the
summary payload; this module decides which item ids are valid, how presets map
to items, and how each item renders in the text summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from analysis.dawn_operator_datasets import (
    OPERATOR_ANALYSIS_ITEM_IDS,
    OPERATOR_DATASET_SPECS,
)


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
    "target_quantile_gap": {
        "title": "Target quantile gap",
        "measures": "score quantile needed for target/candidate admission versus current tau.",
        "summary": "TARGET_QUANTILE_GAP shows how far tau sits above or below the target score boundary.",
        "requires": ("active",),
    },
    "calibration_state": {
        "title": "Calibration state",
        "measures": "target, candidate, observed admission, admission error, tau, and calibration update availability.",
        "summary": "CALIBRATION_STATE shows whether selection calibration is moving toward the requested candidate.",
        "requires": ("active",),
    },
    "qk_split": {
        "title": "Q/K split",
        "measures": "separate Q and K active/effective fractions and per-layer split diagnostics.",
        "summary": "QK_SPLIT separates Q from K so a thin QK path can be attributed to one side.",
        "requires": ("active",),
    },
    "concentration_max": {
        "title": "Concentration max",
        "measures": "the layer with max top1 concentration for each pool, plus local active/effective values.",
        "summary": "CONCENTRATION_MAX points to the layer responsible for max route concentration.",
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
    "composition_health": {
        "title": "Composition denominator health",
        "measures": "v4171 admission mass, composition denominator range, floor fraction, and configured denominator power by pool.",
        "summary": "COMPOSITION_HEALTH verifies that v4171 operator composition normalization is finite and not pinned to its floor.",
        "requires": ("active",),
    },
    "prompt_trace": {
        "title": "Prompt trace",
        "measures": "prompt token/layer q/k/v/rst active counts, gate mass, top1, top operator ids, and output norms.",
        "summary": "PROMPT_TRACE shows version-matched route behavior on diagnostic prompts without generating new text.",
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
        "measures": "version-matched KV-cache prompt continuations, generated token count, and decode throughput.",
        "summary": "GENERATION_SAMPLES gives a quick qualitative read on the current checkpoint.",
        "requires": ("generation",),
    },
    "operator_dataset_manifest": {
        "title": "Operator dataset manifest",
        "measures": "configured GCS dataset root, manifest path, per-dataset artifact paths, and observed public dataset sizes.",
        "summary": "OPERATOR_DATASET_MANIFEST is the path contract for TPU-side operator-family experiments.",
        "requires": ("operator_datasets",),
    },
    "operator_behavior_eval": {
        "title": "Operator behavior evaluation",
        "measures": "Teacher-forced full-sequence and continuation margins, accuracy, known-correct subsets, and logical-example bootstrap intervals.",
        "summary": "OPERATOR_BEHAVIOR_EVAL runs the restored production checkpoint on every selected prepared behavior row.",
        "requires": ("operator_datasets",),
    },
    "operator_function_reuse": {
        "title": "Operator function reuse",
        "measures": "Same-function path similarity, length-matched random null, effect, and logical-pair bootstrap interval.",
        "summary": "OPERATOR_FUNCTION_REUSE compares reusable transition paths with cross-function controls.",
        "requires": ("operator_datasets",),
    },
    "operator_route_specificity": {
        "title": "Operator route specificity",
        "measures": "Within-group and between-group routing/transition overlap, captured mass, specificity gap, and enriched operators.",
        "summary": "OPERATOR_ROUTE_SPECIFICITY reports only captured-mass-qualified route evidence.",
        "requires": ("operator_datasets",),
    },
    "operator_causal_specificity": {
        "title": "Operator causal specificity",
        "measures": "Task-margin drop for selected, matched-active, active-random, inactive-random, and cross-function strategies.",
        "summary": "OPERATOR_CAUSAL_SPECIFICITY uses production contribution subtraction and blocks on zero-vector parity.",
        "requires": ("operator_datasets",),
    },
    "operator_analysis_summary": {
        "title": "Operator analysis summary",
        "measures": "Behavior competence, function reuse, route specificity, causal specificity, validity, and limitations.",
        "summary": "OPERATOR_ANALYSIS_SUMMARY is the artifact-backed cross-dataset result.",
        "requires": ("operator_datasets",),
    },
    "ravel_operator_disentanglement": {
        "title": "RAVEL operator disentanglement",
        "measures": "attribute-specific gate signatures, same-attribute overlap, and cross-attribute causal drops.",
        "summary": "RAVEL checks whether operation space is addressable by semantic attribute rather than entity memorization.",
        "requires": ("operator_datasets",),
    },
    "ioi_operator_circuit": {
        "title": "IOI operator circuit",
        "measures": "clean/corrupt gate deltas and QK/V/RST causal drops for indirect-object identification.",
        "summary": "IOI checks whether the known name circuit appears as selected RW operator families.",
        "requires": ("operator_datasets",),
    },
    "blimp_operator_grammar": {
        "title": "BLiMP operator grammar",
        "measures": "minimal-pair likelihood margins and phenomenon-specific operator ablations.",
        "summary": "BLiMP checks whether reusable grammar phenomena map to reusable operator families.",
        "requires": ("operator_datasets",),
    },
    "lama_counterfact_factual_recall": {
        "title": "LAMA/CounterFact factual recall",
        "measures": "known-fact relation margins, true/new object margins, and relation-specific RST/QK/V causal drops.",
        "summary": "LAMA/CounterFact checks whether factual recall is mediated by relation-specific residual-write operators.",
        "requires": ("operator_datasets",),
    },
    "synthetic_binding_sanity": {
        "title": "Synthetic binding sanity",
        "measures": "controlled entity-attribute binding margins and operator family specificity.",
        "summary": "Synthetic binding provides a small controlled sanity check for entity binding and retrieval writes.",
        "requires": ("operator_datasets",),
    },
    "global_router_audit": {
        "title": "v4171 global router audit",
        "measures": "restored router/pool parameter paths, hidden block-local routing parameters, sharing, geometry, and composition settings.",
        "summary": "GLOBAL_ROUTER_AUDIT fails loudly if v4171 is not using one global router and one global operator pool tree.",
        "requires": ("v4171_transition",),
    },
    "trajectory_trace": {
        "title": "v4171 target-token trajectory trace",
        "measures": "target-span residual/query trajectories, sparse operator ids, captured mass, read responses, coefficients, Q/K/V SRW features, and actual attention/RST residual updates.",
        "summary": "TRAJECTORY_TRACE reuses one target-only transition cache and never returns the full gate tensor to the host.",
        "requires": ("v4171_transition",),
    },
    "context_divergence": {
        "title": "v4171 context divergence",
        "measures": "same-surface pair state/query/sparse-gate/update similarity, first divergence, maximum divergence, and late reconvergence.",
        "summary": "CONTEXT_DIVERGENCE identifies where controlled lexical-ambiguity pairs separate, with captured-mass-qualified gate metrics.",
        "requires": ("v4171_transition",),
    },
    "state_transition_decoupling": {
        "title": "v4171 state-transition decoupling",
        "measures": "state/query/gate/delta/path similarity, data-quantile quadrants, random null percentiles, correlations, effect, and bootstrap interval.",
        "summary": "STATE_TRANSITION_DECOUPLING tests whether distant representation states can share transition paths without using a fixed arbitrary threshold.",
        "requires": ("v4171_transition",),
    },
    "causal_intervention": {
        "title": "v4171 canonical causal intervention",
        "measures": "target-token/layer/pool top-contribution, top-gate, active/inactive random, and matched-active contribution subtraction with CE, log-prob, KL, prediction, and residual effects.",
        "summary": "CAUSAL_INTERVENTION subtracts a selected post-denominator production contribution and blocks on exact zero-vector parity.",
        "requires": ("v4171_transition",),
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
    "dataset_manifest": "operator_dataset_manifest",
    "operator_manifest": "operator_dataset_manifest",
    "ravel": "ravel_operator_disentanglement",
    "ioi": "ioi_operator_circuit",
    "blimp": "blimp_operator_grammar",
    "lama": "lama_counterfact_factual_recall",
    "counterfact": "lama_counterfact_factual_recall",
    "synthetic": "synthetic_binding_sanity",
}

V4166_1B_ITEMS = (
    "target_ratio",
    "layer_selectivity",
    "target_quantile_gap",
    "calibration_state",
    "qk_split",
    "concentration_max",
    "prune_breakdown",
    "execution_profile",
    "prompt_trace",
    "prompt_decision",
    "generation_samples",
    "decision_reason",
)

V4171_SELF_ORGANIZATION_ITEMS = (
    "global_router_audit",
    "trajectory_trace",
    "context_divergence",
    "state_transition_decoupling",
    "causal_intervention",
)

V4171_ITEMS = tuple(
    item for item in TRAIN_ANALYSIS_ITEM_DEFS
    if item not in V4171_SELF_ORGANIZATION_ITEMS
    and item not in OPERATOR_ANALYSIS_ITEM_IDS
)

TRAIN_ANALYSIS_PRESETS = {
    "minimal": ("target_ratio", "prune_breakdown", "decision_reason"),
    "qk_closed": (
        "target_ratio",
        "layer_selectivity",
        "target_quantile_gap",
        "calibration_state",
        "qk_split",
        "concentration_max",
        "prune_breakdown",
        "execution_profile",
        "decision_reason",
    ),
    "compute": ("target_ratio", "prune_breakdown", "execution_profile", "decision_reason"),
    "health": (
        "target_ratio", "layer_selectivity", "num_health",
        "composition_health", "decision_reason"),
    "prompt_debug": ("target_ratio", "layer_selectivity", "prompt_trace", "prompt_decision", "decision_reason"),
    "sample": ("target_ratio", "generation_samples", "decision_reason"),
    "operator_datasets": ("operator_dataset_manifest",),
    "operator_analysis": OPERATOR_ANALYSIS_ITEM_IDS,
    "v4171_operator_monitor": tuple(dict.fromkeys(
        V4171_SELF_ORGANIZATION_ITEMS + OPERATOR_ANALYSIS_ITEM_IDS)),
    "v4171_complete": tuple(dict.fromkeys(
        V4171_ITEMS + V4171_SELF_ORGANIZATION_ITEMS + OPERATOR_ANALYSIS_ITEM_IDS)),
    "v4166_1b": V4166_1B_ITEMS,
    "v4171": V4171_ITEMS,
    "v4171_self_organization": V4171_SELF_ORGANIZATION_ITEMS,
    "deep": V4166_1B_ITEMS,
    "full": V4171_ITEMS,
}

TRAIN_ANALYSIS_PRESET_ALIASES = {
    "v4166-1b": "v4166_1b",
    "v4166-1p3b": "v4166_1b",
    "v4166-1p3b-c4-20b": "v4166_1b",
    "v4166-1p3b-c4-20b-v4-64": "v4166_1b",
    "v4171-400m": "v4171",
    "v4171-1p3b": "v4171",
    "v4171-400m-c4-40b-v4-64": "v4171",
    "v4171-1p3b-c4-20b-v4-64": "v4171",
    "operator": "operator_analysis",
    "operator-analysis": "operator_analysis",
    "operator-datasets": "operator_datasets",
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


def _item_lines_target_quantile_gap(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    active = summary["active_dynamics"]
    rows = active.get("target_quantile_gap", [])
    out = [
        "  TARGET_QUANTILE_GAP:",
        "  source=layer_score_quantile_approx",
        "  pool target candidate q@target q@candidate tau gap_target gap_candidate",
    ]
    if not rows:
        out.append("  n/a")
    for row in rows:
        out.append(
            "  "
            f"{str(row.get('pool', 'n/a')):<5} "
            f"{fmt.num(row.get('target'), 3):<6} "
            f"{fmt.num(row.get('candidate'), 3):<9} "
            f"{fmt.num(row.get('score_q_target'), 4):<8} "
            f"{fmt.num(row.get('score_q_candidate'), 4):<11} "
            f"{fmt.num(row.get('tau'), 4):<7} "
            f"{fmt.delta(row.get('gap_target'), 4):<10} "
            f"{fmt.delta(row.get('gap_candidate'), 4)}"
        )
    return out


def _item_lines_calibration_state(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    active = summary["active_dynamics"]
    rows = active.get("calibration_state", [])
    out = [
        "  CALIBRATION_STATE:",
        "  pool target candidate observed_adm error tau_before tau_after tau_delta clamp stopgrad mode",
    ]
    if not rows:
        out.append("  n/a")
    for row in rows:
        out.append(
            "  "
            f"{str(row.get('pool', 'n/a')):<5} "
            f"{fmt.num(row.get('target'), 3):<6} "
            f"{fmt.num(row.get('candidate'), 3):<9} "
            f"{fmt.num(row.get('observed_admission'), 3):<12} "
            f"{fmt.delta(row.get('error'), 3):<7} "
            f"{fmt.num(row.get('tau_before'), 4):<10} "
            f"{fmt.num(row.get('tau_after'), 4):<9} "
            f"{fmt.delta(row.get('tau_delta'), 4):<9} "
            f"{str(row.get('clamp_hit', 'n/a')):<5} "
            f"{str(row.get('stopgrad', 'n/a')):<8} "
            f"{row.get('mode', 'n/a')}"
        )
    return out


def _item_lines_qk_split(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    active = summary["active_dynamics"]
    qk_split = active.get("qk_split", {})
    split_summary = qk_split.get("summary", {})
    out = [
        "  QK_SPLIT_SUMMARY:",
        "  side active_tau admission effective active_ops effective_ops eff_min eff_mean eff_max",
    ]
    for side in ("q", "k"):
        row = split_summary.get(side, {})
        out.append(
            "  "
            f"{side:<4} "
            f"{fmt.num(row.get('active_tau'), 3):<10} "
            f"{fmt.num(row.get('admission'), 3):<9} "
            f"{fmt.num(row.get('effective'), 3):<9} "
            f"{fmt.num(row.get('active_ops_mean'), 1):<10} "
            f"{fmt.num(row.get('effective_ops_mean'), 1):<13} "
            f"{fmt.num(row.get('eff_min'), 3):<7} "
            f"{fmt.num(row.get('eff_mean'), 3):<8} "
            f"{fmt.num(row.get('eff_max'), 3)}"
        )
    out.append(f"  q/k_effective_balance={fmt.num(split_summary.get('qk_effective_balance'), 3)}")
    out.extend([
        "  QK_SPLIT_LAYERS:",
        "  lyr q_act q_adm q_eff q_ops | k_act k_adm k_eff k_ops | qk_eff",
    ])
    rows = qk_split.get("layers", [])
    if not rows:
        out.append("  n/a")
    for row in rows:
        out.append(
            "  "
            f"{int(row.get('layer', 0)):02d}  "
            f"{fmt.num(row.get('q_active_tau'), 3):<5} "
            f"{fmt.num(row.get('q_admission'), 3):<5} "
            f"{fmt.num(row.get('q_effective'), 3):<5} "
            f"{fmt.num(row.get('q_effective_ops'), 1):<5} | "
            f"{fmt.num(row.get('k_active_tau'), 3):<5} "
            f"{fmt.num(row.get('k_admission'), 3):<5} "
            f"{fmt.num(row.get('k_effective'), 3):<5} "
            f"{fmt.num(row.get('k_effective_ops'), 1):<5} | "
            f"{fmt.num(row.get('qk_effective'), 3)}"
        )
    return out


def _item_lines_concentration_max(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    active = summary["active_dynamics"]
    rows = active.get("concentration_max", [])
    out = [
        "  CONCENTRATION_MAX:",
        "  pool layer layer_top1_max global_top1_max layer_top1_mean active effective active_ops effective_ops operator_id",
    ]
    if not rows:
        out.append("  n/a")
    for row in rows:
        out.append(
            "  "
            f"{str(row.get('pool', 'n/a')):<5} "
            f"{str(row.get('layer', 'n/a')):<5} "
            f"{fmt.num(row.get('layer_top1_max'), 3):<14} "
            f"{fmt.num(row.get('global_top1_max'), 3):<15} "
            f"{fmt.num(row.get('top1_mean'), 3):<15} "
            f"{fmt.num(row.get('active'), 3):<6} "
            f"{fmt.num(row.get('effective'), 3):<9} "
            f"{fmt.num(row.get('active_ops'), 1):<10} "
            f"{fmt.num(row.get('effective_ops'), 1):<13} "
            f"{row.get('operator_id') or 'n/a'}"
        )
    return out


def _item_lines_prune_breakdown(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    prune = summary["effective_prune"]
    baseline = prune.get("baseline") or {}
    prune_rows = ([baseline] if baseline else []) + list(prune.get("eps", []))
    base_compute = fmt.safe_float(baseline.get("compute_frac"))
    out = [
        "  PRUNE_BREAKDOWN:",
        "  eps   val_loss delta_ce total qk_cmp v_cmp rst_cmp | qk_eff v_eff rst_eff saved gate_mass_raw no_active",
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


def _item_lines_composition_health(
    summary: Dict[str, Any], fmt: TrainAnalysisFormatters
) -> List[str]:
    composition = summary.get("active_dynamics", {}).get(
        "composition_health", {})
    model_version = summary.get("run", {}).get("model_version")
    out = [
        "  COMPOSITION_HEALTH:",
        f"  model_version={model_version or 'unknown'} "
        f"admission_den_power={fmt.num(composition.get('admission_den_power'), 3)}",
    ]
    if not composition.get("available"):
        out.append("  not available for this model/checkpoint metric schema")
        return out
    out.extend([
        "  pool  admission_mean admission_max den_mean den_min den_max floor_frac",
    ])
    for pool in ("qk", "v", "rst"):
        row = composition.get("pools", {}).get(pool, {})
        out.append(
            "  "
            f"{pool:<5} "
            f"{fmt.num(row.get('admission_mass_mean'), 4):<14} "
            f"{fmt.num(row.get('admission_mass_max'), 4):<13} "
            f"{fmt.num(row.get('composition_den_mean'), 4):<8} "
            f"{fmt.num(row.get('composition_den_min'), 4):<7} "
            f"{fmt.num(row.get('composition_den_max'), 4):<7} "
            f"{fmt.num(row.get('composition_den_floor_frac'), 4)}"
        )
    return out


def _item_lines_prompt_trace(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    trace = summary.get("prompt_trace", {})
    if trace.get("status") not in ("ready", "empty"):
        return ["  PROMPT_TRACE:", f"  status={trace.get('status', 'missing')} reason={trace.get('reason', 'n/a')}"]
    out = [
        "  PROMPT_TRACE:",
        f"  boundary_power={fmt.num((trace.get('inference_model_cfg') or {}).get('soft_gate_boundary_power'), 3)}",
        "  prompt_id       len q_frac k_frac v_frac rst_frac q_top1 k_top1 v_top1 rst_top1 rst/attn",
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
            f"{fmt.num(pools.get('q', {}).get('active_frac_mean'), 3):<6} "
            f"{fmt.num(pools.get('k', {}).get('active_frac_mean'), 3):<6} "
            f"{fmt.num(pools.get('v', {}).get('active_frac_mean'), 3):<6} "
            f"{fmt.num(pools.get('rst', {}).get('active_frac_mean'), 3):<8} "
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
        "  prompt_id       status qk_frac v_frac rst_frac rst_top1 rst/attn reason",
    ]
    rows = decision.get("rows", [])
    if not rows:
        out.append("  n/a")
    for row in rows:
        out.append(
            "  "
            f"{str(row.get('prompt_id', 'n/a')):<15} "
            f"{str(row.get('status', 'n/a')):<6} "
            f"{fmt.num(row.get('qk_active_frac'), 3):<7} "
            f"{fmt.num(row.get('v_active_frac'), 3):<6} "
            f"{fmt.num(row.get('rst_active_frac'), 3):<8} "
            f"{fmt.num(row.get('rst_top1_max'), 3):<8} "
            f"{fmt.num(row.get('rst_attn_norm_ratio'), 3):<7} "
            f"{row.get('reason', 'n/a')}"
        )
    return out


def _compact_text(value: Any, max_chars: int = 320) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    if not text:
        return "n/a"
    if len(text) > max_chars:
        return text[: max(0, max_chars - 3)] + "..."
    return text


def _top_token_line(tokens: List[Dict[str, Any]], fmt: TrainAnalysisFormatters) -> str:
    parts = []
    for row in tokens[:8]:
        token = str(row.get("token", "n/a")).replace(" ", "_")
        parts.append(
            f"{token}#{row.get('id', 'n/a')}:{fmt.pct(row.get('prob'), 1)}"
        )
    return ", ".join(parts) if parts else "n/a"


def _item_lines_generation_samples(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    generation = summary.get("generation_samples", {})
    if generation.get("status") not in ("ready", "empty"):
        return ["  GENERATION_SAMPLES:", f"  status={generation.get('status', 'missing')} reason={generation.get('reason', 'n/a')}"]
    out = [
        "  GENERATION_SAMPLES:",
        (
            "  "
            f"decode_mode={generation.get('decode_mode', 'n/a')} "
            f"max_new={generation.get('max_new_tokens', 'n/a')} "
            f"temp={fmt.num(generation.get('temperature'), 2)} "
            f"top_k={generation.get('top_k', 'n/a')} "
            f"seed={generation.get('sampling_seed', 'n/a')} "
            f"boundary_power={fmt.num((generation.get('inference_model_cfg') or {}).get('soft_gate_boundary_power'), 3)}"
        ),
    ]
    rows = generation.get("samples", [])
    if not rows:
        out.append("  n/a")
    for row in rows:
        dominant = row.get("dominant_generated_token", {}) or {}
        out.append(
            "  "
            f"{str(row.get('prompt_id', 'n/a')):<12} "
            f"new_tok={str(row.get('new_tokens', 'n/a')):<3} "
            f"tok/s={fmt.num(row.get('tokens_per_sec'), 1):<5} "
            f"unique={dominant.get('unique', 'n/a')} "
            f"dom={dominant.get('token', 'n/a')}:{fmt.pct(dominant.get('frac'), 1)}"
        )
        out.append(f"    prompt   : {_compact_text(row.get('prompt'), 220)}")
        out.append(f"    generated: {_compact_text(row.get('continuation'), 360)}")
        out.append(f"    full     : {_compact_text(row.get('full_text'), 420)}")
        out.append(f"    first_top: {_top_token_line(row.get('first_step_top_tokens', []), fmt)}")
    return out


def _operator_dataset_block(summary: Dict[str, Any], dataset_id: str) -> List[str]:
    analysis = summary.get("operator_analysis", {})
    spec = (analysis.get("datasets") or {}).get(dataset_id, {})
    if not spec:
        return [
            f"  dataset      : {dataset_id}",
            "  status       : missing_dataset",
        ]
    behavior = spec.get("behavior", {})
    route = spec.get("route", {})
    causal = spec.get("causal", {})
    out = [
        f"  dataset      : {dataset_id}",
        f"  status       : {spec.get('status', 'missing')}",
        ("  behavior     : "
         f"n={behavior.get('n', 0)} accuracy={behavior.get('accuracy')} "
         f"margin={behavior.get('mean_margin')} known={behavior.get('known_correct', 0)}"),
        ("  route        : "
         f"actual={route.get('actual_path_similarity')} "
         f"null={route.get('random_null_path_similarity')} "
         f"effect={route.get('effect_vs_null')} captured={route.get('captured_mass_mean')}"),
        ("  causal       : "
         f"jobs={causal.get('jobs', 0)} skipped={causal.get('skipped_jobs', 0)} "
         f"effect={causal.get('selected_minus_control_effect')} "
         f"parity={(causal.get('causal_parity') or {}).get('machine_exact')}"),
    ]
    artifacts = spec.get("artifacts", {})
    if artifacts:
        out.append("  artifacts:")
        for name, path in sorted(artifacts.items()):
            out.append(f"    {name:<20} {path}")
    return out


def _item_lines_operator_dataset_manifest(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    datasets = summary.get("operator_analysis", {}).get("dataset_manifest", {})
    rows = datasets.get("datasets", {})
    out = [
        "  OPERATOR_DATASET_MANIFEST:",
        f"  root     : {datasets.get('root', 'n/a')}",
        f"  manifest : {datasets.get('manifest', 'n/a')}",
        f"  status   : {datasets.get('status', 'missing')}",
        f"  build_id : {datasets.get('build_id', 'n/a')}",
        f"  schema   : {datasets.get('schema', 'n/a')} v{datasets.get('schema_version', 'n/a')}",
        "  id          rows shards checksummed",
    ]
    for dataset_id in OPERATOR_DATASET_SPECS:
        row = rows.get(dataset_id, {})
        out.append(
            f"  {dataset_id:<11} {str(row.get('rows', 'n/a')):<6} "
            f"{str(row.get('shards', 'n/a')):<6} {row.get('checksummed_shards', 'n/a')}")
    return out


def _item_lines_operator_cross(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    analysis = summary.get("operator_analysis", {})
    out = [
        "  OPERATOR_ANALYSIS:",
        f"  status={analysis.get('status', 'missing')} profile={analysis.get('profile', 'n/a')}",
        f"  build={analysis.get('build_id', 'n/a')} config={analysis.get('config_hash', 'n/a')}",
    ]
    for dataset in sorted((analysis.get("datasets") or {})):
        out.extend(_operator_dataset_block(summary, dataset)[1:4])
    return out


def _item_lines_ravel_operator_disentanglement(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    return ["  RAVEL_OPERATOR_DISENTANGLEMENT:"] + _operator_dataset_block(summary, "ravel")


def _item_lines_ioi_operator_circuit(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    return ["  IOI_OPERATOR_CIRCUIT:"] + _operator_dataset_block(summary, "ioi")


def _item_lines_blimp_operator_grammar(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    return ["  BLIMP_OPERATOR_GRAMMAR:"] + _operator_dataset_block(summary, "blimp")


def _item_lines_lama_counterfact_factual_recall(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    return (
        ["  LAMA_COUNTERFACT_FACTUAL_RECALL:"]
        + _operator_dataset_block(summary, "lama")
        + ["", "  COUNTERFACT:"]
        + _operator_dataset_block(summary, "counterfact")
    )


def _item_lines_synthetic_binding_sanity(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    return ["  SYNTHETIC_BINDING_SANITY:"] + _operator_dataset_block(summary, "synthetic")


def _v4171_item(summary: Dict[str, Any], item: str) -> Dict[str, Any]:
    return dict((summary.get("v4171_transition_analysis") or {}).get(item) or {})


def _item_lines_global_router_audit(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    data = _v4171_item(summary, "global_router_audit")
    pools = data.get("pool_sizes", {})
    return [
        "  GLOBAL_ROUTER_AUDIT:",
        f"  status                    : {data.get('status', 'missing')}",
        f"  router_param_count        : {data.get('router_param_count', 'n/a')}",
        f"  shared_across_layers      : {data.get('shared_across_layers', 'n/a')}",
        f"  hidden_layer_router_params: {data.get('hidden_layer_router_params', [])}",
        f"  operator_keys_shared      : {data.get('operator_keys_shared', 'n/a')}",
        f"  operator_rw_shared        : {data.get('operator_rw_shared', 'n/a')}",
        f"  pool_sizes                : qk={pools.get('qk')} v={pools.get('v')} rst={pools.get('rst')}",
        f"  geometry                  : d_route={data.get('d_route')} d_model={data.get('d_model')} layers={data.get('n_layers')}",
        f"  composition               : mode={data.get('composition_mode')} den_power={data.get('admission_den_power')}",
    ]


def _item_lines_trajectory_trace(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    data = _v4171_item(summary, "trajectory_trace")
    decoupling = _v4171_item(summary, "state_transition_decoupling")
    pair_counts = decoupling.get("counts", {})
    captured = data.get("captured_mass", {})
    captured_by_pool = data.get("captured_mass_by_pool", {})
    semantics = data.get("tensor_semantics", {})
    out = [
        "  TRAJECTORY_TRACE:",
        f"  status          : {data.get('status', 'missing')}",
        f"  prompt_set      : {data.get('prompt_set', 'n/a')}",
        f"  prompt_hash     : {data.get('prompt_set_hash', 'n/a')}",
        f"  prompts/subtoken: {data.get('num_prompts', 0)}/{data.get('num_target_subtokens', 0)}",
        f"  span_aggregate  : {data.get('span_aggregation', 'n/a')}",
        f"  sparse_topk     : {data.get('trace_topk', 'n/a')}",
        f"  captured_mass   : mean={fmt.pct(captured.get('mean'), 2)} min={fmt.pct(captured.get('min'), 2)} p10={fmt.pct(captured.get('p10'), 2)}",
        f"  valid_pair_rows : {fmt.pct(pair_counts.get('valid_pair_metric_row_fraction'), 2)}",
        f"  excluded_lowcap : {pair_counts.get('excluded_low_capture_rows', 'n/a')}",
        f"  qkv_semantics   : {semantics.get('qkv', 'n/a')}",
        f"  attention_delta : {semantics.get('attention', 'n/a')}",
        f"  rst_delta       : {semantics.get('rst', 'n/a')}",
        f"  artifact        : {(data.get('artifacts') or {}).get('trajectory_traces', 'n/a')}",
    ]
    if data.get("captured_mass_warning"):
        out.append(f"  warning         : {data['captured_mass_warning']}")
    for pool in ("q", "k", "v", "rst"):
        row = captured_by_pool.get(pool, {})
        out.append(
            f"  captured {pool:<3}    : mean={fmt.pct(row.get('mean'), 2)} "
            f"min={fmt.pct(row.get('min'), 2)} p10={fmt.pct(row.get('p10'), 2)}")
    return out


def _item_lines_context_divergence(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    data = _v4171_item(summary, "context_divergence")
    out = [
        "  CONTEXT_DIVERGENCE:",
        f"  status     : {data.get('status', 'missing')}",
        f"  pair_count : {data.get('num_pairs', 0)}",
        f"  significant: {data.get('num_significant_pool_pairs', 0)}",
        f"  no_signal  : {data.get('num_no_significant_pool_pairs', 0)}",
        f"  gate_metric: {data.get('gate_metric', 'n/a')}",
        "  significant pair/pool first max reconv reconv_layer evidence captured_min",
    ]
    for row in [
        value for value in (data.get("pairs") or [])
        if value.get("status") == "significant_divergence"
    ][:16]:
        out.append(
            "  "
            f"{row.get('pair_id')}/{row.get('pool')} "
            f"{row.get('first_divergence_layer')} "
            f"{row.get('maximum_divergence_layer')} "
            f"{row.get('late_reconvergence')} "
            f"{row.get('late_reconvergence_layer')} "
            f"{fmt.num(row.get('maximum_divergence_evidence'), 4)} "
            f"{fmt.pct(row.get('min_captured_mass'), 1)}"
        )
    return out


def _item_lines_state_transition_decoupling(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    data = _v4171_item(summary, "state_transition_decoupling")
    actual = data.get("path_similarity", {})
    null = data.get("random_null_path_similarity", {})
    counts = data.get("counts", {})
    out = [
        "  STATE_TRANSITION_DECOUPLING:",
        f"  status               : {data.get('status', 'missing')}",
        f"  state_low_q25        : {fmt.num(data.get('state_low_threshold_data_q25'), 4)}",
        f"  transition_high_null: {fmt.num(data.get('transition_high_threshold_random_null_q75'), 4)}",
        f"  quadrants            : {data.get('quadrants', {})}",
        f"  quadrant unique pairs: {data.get('quadrant_unique_pairs', {})}",
        f"  path_similarity      : mean={fmt.num(actual.get('mean'), 4)} ci95={actual.get('ci95')} n={actual.get('n', 0)}",
        f"  random_null          : mean={fmt.num(null.get('mean'), 4)} ci95={null.get('ci95')} n={null.get('n', 0)}",
        f"  effect_vs_random     : {fmt.delta(data.get('path_similarity_effect_vs_random'), 4)}",
        f"  pair counts          : actual={counts.get('actual_unique_pairs', 0)} null={counts.get('null_unique_pairs', 0)}",
        f"  valid pool-pairs     : actual={counts.get('actual_valid_pool_pairs', 0)} null={counts.get('null_valid_pool_pairs', 0)}",
        f"  excluded low-capture: {counts.get('excluded_low_capture_rows', 0)}",
        f"  path definition      : {data.get('path_definition', 'n/a')}",
    ]
    for pool, row in (data.get("correlations") or {}).items():
        out.append(
            f"  corr {pool}: state-gate={fmt.num(row.get('corr_state_gate'), 3)} "
            f"state-delta={fmt.num(row.get('corr_state_delta'), 3)} "
            f"query-gate={fmt.num(row.get('corr_query_gate'), 3)} "
            f"gate-delta={fmt.num(row.get('corr_gate_delta'), 3)}"
        )
    return out


def _item_lines_causal_intervention(summary: Dict[str, Any], fmt: TrainAnalysisFormatters) -> List[str]:
    data = _v4171_item(summary, "causal_intervention")
    selected = data.get(
        "selected_behavior_score_drop",
        data.get("selected_abs_target_logprob_delta", {}))
    control = data.get(
        "control_behavior_score_drop",
        data.get("control_abs_target_logprob_delta", {}))
    diagnostic = data.get("zero_subtraction_parity") or data.get(
        "intervention_forward_cross_graph_diagnostic",
        data.get("intervention_forward_parity", {}),
    )
    out = [
        "  CAUSAL_INTERVENTION:",
        f"  status          : {data.get('status', 'missing')}",
        f"  type            : {data.get('intervention_type', 'n/a')}",
        f"  canonical_den   : {data.get('canonical_unpruned_admission_denominator', False)}",
        f"  baseline        : {data.get('canonical_baseline_source', 'legacy')}",
        f"  effect reference: {data.get('effect_reference', 'legacy')}",
        f"  prompts/jobs    : {data.get('num_prompts', 0)}/{data.get('num_interventions', 0)} skipped={data.get('num_skipped', 0)}",
        "  INTERVENTION_FORWARD_DIAGNOSTIC:",
        f"  diagnostic      : status={diagnostic.get('status', 'missing')} blocking={diagnostic.get('blocking', 'legacy')} threshold_passed={diagnostic.get('threshold_passed', 'n/a')}",
        f"  diagnostic CE/logit: ce={fmt.num(diagnostic.get('ce_abs_diff'), 6)} mean={fmt.num(diagnostic.get('mean_logit_abs_diff'), 6)} max={fmt.num(diagnostic.get('max_logit_abs_diff'), 6)}",
        f"  diagnostic top1/res: top1={fmt.num(diagnostic.get('top1_agreement'), 6)} residual_cos={fmt.num(diagnostic.get('final_residual_cosine'), 8)}",
        f"  selected effect : mean={fmt.num(selected.get('mean'), 5)} ci95={selected.get('ci95')} n={selected.get('n', 0)}",
        f"  control effect  : mean={fmt.num(control.get('mean'), 5)} ci95={control.get('ci95')} n={control.get('n', 0)}",
        f"  selected-control: {fmt.delta(data.get('selected_minus_control_effect'), 5)}",
        f"  artifact        : {data.get('artifact', 'n/a')}",
    ]
    effects = data.get("effects", {})
    for group_name in ("by_strategy", "by_pool", "by_phenomenon"):
        for key, row in (effects.get(group_name) or {}).items():
            out.append(
                f"  effect {group_name}/{key}: n={row.get('n', 0)} "
                f"abs_dlogp={fmt.num(row.get('mean_abs_target_logprob_delta'), 5)} "
                f"ci95={row.get('bootstrap_ci95')} kl={fmt.num(row.get('mean_kl'), 5)} "
                f"top1_change={fmt.pct(row.get('top_prediction_changed_fraction'), 2)}")
    return out


def _item_lines_decision_reason(summary: Dict[str, Any], _fmt: TrainAnalysisFormatters) -> List[str]:
    return [f"  - {line_i}" for line_i in (summary.get("decision") or ["n/a"])]


TRAIN_ANALYSIS_ITEM_FORMATTERS = {
    "target_ratio": _item_lines_target_ratio,
    "layer_selectivity": _item_lines_layer_selectivity,
    "target_quantile_gap": _item_lines_target_quantile_gap,
    "calibration_state": _item_lines_calibration_state,
    "qk_split": _item_lines_qk_split,
    "concentration_max": _item_lines_concentration_max,
    "prune_breakdown": _item_lines_prune_breakdown,
    "execution_profile": _item_lines_execution_profile,
    "num_health": _item_lines_num_health,
    "composition_health": _item_lines_composition_health,
    "prompt_trace": _item_lines_prompt_trace,
    "prompt_decision": _item_lines_prompt_decision,
    "generation_samples": _item_lines_generation_samples,
    "operator_dataset_manifest": _item_lines_operator_dataset_manifest,
    "operator_behavior_eval": _item_lines_operator_cross,
    "operator_function_reuse": _item_lines_operator_cross,
    "operator_route_specificity": _item_lines_operator_cross,
    "operator_causal_specificity": _item_lines_operator_cross,
    "operator_analysis_summary": _item_lines_operator_cross,
    "ravel_operator_disentanglement": _item_lines_ravel_operator_disentanglement,
    "ioi_operator_circuit": _item_lines_ioi_operator_circuit,
    "blimp_operator_grammar": _item_lines_blimp_operator_grammar,
    "lama_counterfact_factual_recall": _item_lines_lama_counterfact_factual_recall,
    "synthetic_binding_sanity": _item_lines_synthetic_binding_sanity,
    "global_router_audit": _item_lines_global_router_audit,
    "trajectory_trace": _item_lines_trajectory_trace,
    "context_divergence": _item_lines_context_divergence,
    "state_transition_decoupling": _item_lines_state_transition_decoupling,
    "causal_intervention": _item_lines_causal_intervention,
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
    if item == "target_quantile_gap":
        return "ready" if active.get("target_quantile_gap") else "missing"
    if item == "calibration_state":
        return "ready" if active.get("calibration_state") else "missing"
    if item == "qk_split":
        qk_split = active.get("qk_split", {})
        if qk_split.get("layers"):
            return "ready"
        return "partial" if qk_split.get("summary") else "missing"
    if item == "concentration_max":
        return "ready" if active.get("concentration_max") else "missing"
    if item == "prune_breakdown":
        return "ready" if (prune.get("baseline") or prune.get("eps")) else "missing"
    if item == "execution_profile":
        return "ready" if active.get("pools") else "missing"
    if item == "num_health":
        return "ready" if active.get("num_health") else "missing"
    if item == "composition_health":
        composition = active.get("composition_health", {})
        if composition.get("available"):
            return "ready"
        model_version = str(summary.get("run", {}).get("model_version", ""))
        return "not_applicable" if model_version.endswith("4.1.6.6") else "missing"
    if item == "prompt_trace":
        trace = summary.get("prompt_trace", {})
        return str(trace.get("status") or "missing")
    if item == "prompt_decision":
        decision = summary.get("prompt_decision", {})
        return str(decision.get("status") or "missing")
    if item == "generation_samples":
        generation = summary.get("generation_samples", {})
        return str(generation.get("status") or "missing")
    if item in V4171_SELF_ORGANIZATION_ITEMS:
        data = (summary.get("v4171_transition_analysis") or {}).get(item, {})
        return str(data.get("status") or "missing")
    if item in OPERATOR_ANALYSIS_ITEM_IDS:
        result = (summary.get("operator_analysis", {}).get("items", {}) or {}).get(item, {})
        status = str(result.get("status") or "missing_dataset")
        allowed = {
            "ready", "partial", "insufficient_evidence", "missing_dataset",
            "failed", "unsupported",
        }
        return status if status in allowed else "failed"
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
