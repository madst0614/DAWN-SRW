"""Paper-oriented report aggregation."""

from __future__ import annotations

from typing import Any, Dict, List

from analysis.dawn_analysis_common import AnalysisContext
from analysis.dawn_analysis_storage import (
    exists,
    list_paths,
    open_path,
    read_json,
    read_jsonl,
    should_skip_job,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


def _read_if(path: str, default):
    return read_json(path, default=default) if exists(path) else default


def _status_for(paths: List[str]) -> str:
    if all(exists(p) for p in paths):
        return "supported"
    if any(exists(p) for p in paths):
        return "partial"
    return "missing"


def _copy_csv_rows(path: str) -> List[Dict[str, Any]]:
    if not exists(path):
        return []
    # Keep CSV regeneration simple: stage CSVs are already written.  Report
    # tables below are reconstructed from JSON artifacts when possible.
    return []


def _performance_table(eval_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not eval_summary:
        return []
    return [{
        "checkpoint_step": eval_summary.get("checkpoint_step"),
        "val_loss": eval_summary.get("val_loss"),
        "ppl": eval_summary.get("ppl"),
        "accuracy": eval_summary.get("accuracy"),
        "valid_tokens": eval_summary.get("valid_tokens"),
        "tok_per_sec": eval_summary.get("tok_per_sec"),
    }]


def _geometry_table(geom: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for pool, rec in geom.get("pools", {}).items():
        rows.append({
            "pool": pool,
            "N": rec.get("N"),
            "effective_rank": rec.get("effective_rank"),
            "mean_cosine_similarity": rec.get("mean_cosine_similarity"),
            "max_cosine_similarity": rec.get("max_cosine_similarity"),
            "op_key_dead": rec.get("op_key_dead"),
            "read_dead": rec.get("read_dead"),
            "write_dead": rec.get("write_dead"),
        })
    return rows


def _ablation_table(ablation: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "pool": r.get("pool"),
            "strategy": r.get("strategy"),
            "k": r.get("k"),
            "base_loss": r.get("base_loss"),
            "ablated_loss": r.get("ablated_loss"),
            "delta_loss": r.get("delta_loss"),
            "base_acc": r.get("base_acc"),
            "ablated_acc": r.get("ablated_acc"),
            "delta_acc": r.get("delta_acc"),
            "valid_tokens": r.get("valid_tokens"),
        }
        for r in ablation.get("jobs", [])
    ]


def _summary_markdown(ctx: AnalysisContext | None,
                      eval_summary: Dict[str, Any],
                      prune: Dict[str, Any],
                      geom: Dict[str, Any],
                      usage: Dict[str, Any],
                      trace: Dict[str, Any],
                      ablation: Dict[str, Any],
                      checklist: Dict[str, Any]) -> str:
    lines = ["# DAWN-SRW v4166 1.3B Analysis Summary", ""]
    lines.append("## 1. Performance")
    if eval_summary:
        lines.append(
            f"- val_loss={float(eval_summary.get('val_loss', 0.0)):.6f}, "
            f"ppl={float(eval_summary.get('ppl', 0.0)):.3f}, "
            f"accuracy={float(eval_summary.get('accuracy', 0.0)):.4f}, "
            f"valid_tokens={int(eval_summary.get('valid_tokens', 0)):,}"
        )
    else:
        lines.append("- Missing eval/final_eval.json.")
    lines.append("")

    lines.append("## 2. Sparse execution / prune curve")
    eps_rows = prune.get("eps", []) if isinstance(prune, dict) else []
    if eps_rows:
        best = min(eps_rows, key=lambda r: abs(float(r.get("delta_loss_vs_eps0") or 0.0)))
        lines.append(
            f"- eps points={len(eps_rows)}, nearest-loss eps={best.get('eps')}, "
            f"delta_loss={best.get('delta_loss_vs_eps0')}, "
            f"compute_frac={best.get('estimated_compute_frac')}"
        )
    else:
        lines.append("- Missing prune sweep.")
    lines.append("")

    lines.append("## 3. Operator geometry")
    if geom.get("pools"):
        for pool, rec in geom["pools"].items():
            lines.append(
                f"- {pool}: N={rec.get('N'):,}, "
                f"rank={float(rec.get('effective_rank', 0.0)):.2f}, "
                f"mean_cos={float(rec.get('mean_cosine_similarity', 0.0)):.5f}, "
                f"max_cos={float(rec.get('max_cosine_similarity', 0.0)):.5f}"
            )
    else:
        lines.append("- Missing geometry summary.")
    lines.append("")

    lines.append("## 4. Operator usage atlas")
    if usage.get("pools"):
        lines.append(f"- tokens_observed={int(usage.get('tokens_observed', 0)):,}")
        for pool, rec in usage["pools"].items():
            lines.append(
                f"- {pool}: active_ops={int(rec.get('active_ops_observed', 0)):,}/"
                f"{int(rec.get('N', 0)):,}, total_topk_mass={float(rec.get('total_topk_mass', 0.0)):.4g}"
            )
    else:
        lines.append("- Missing usage atlas.")
    lines.append("")

    lines.append("## 5. Token-level traces")
    if trace:
        lines.append(f"- prompts={int(trace.get('num_prompts', 0))}")
    else:
        lines.append("- Missing trace artifacts.")
    lines.append("")

    lines.append("## 6. Causal ablation")
    jobs = ablation.get("jobs", [])
    if jobs:
        top = sorted(jobs, key=lambda r: -float(r.get("delta_loss", 0.0)))[:5]
        for r in top:
            lines.append(
                f"- {r.get('pool')}/{r.get('strategy')}/k={r.get('k')}: "
                f"delta_loss={float(r.get('delta_loss', 0.0)):.6f}, "
                f"delta_acc={float(r.get('delta_acc', 0.0)):.4f}"
            )
    else:
        lines.append("- Missing ablation jobs.")
    lines.append("")

    lines.append("## 7. Patching / steering if available")
    lines.append("- Scaffolding directories are present; default run does not execute patching/steering unless enabled.")
    lines.append("")
    lines.append("## 8. Limitations")
    lines.append("- Usage and traces are compact top-k summaries, not full gate tensor dumps.")
    lines.append("- Operator labels are not assigned automatically; cards provide context evidence only.")
    lines.append("")
    lines.append("## 9. Missing analyses / recommended next run")
    missing = [name for name, rec in checklist.items() if rec.get("status") != "supported"]
    if missing:
        lines.append("- Missing or partial: " + ", ".join(missing))
    else:
        lines.append("- All primary claim buckets have at least one evidence artifact.")
    lines.append("")
    return "\n".join(lines)


def run_report_stage(ctx: AnalysisContext | None, store=None) -> Dict[str, Any]:
    if ctx is not None:
        store = ctx.store
    if store is None:
        raise ValueError("run_report_stage requires a context or store.")
    stage = "report"
    store.set_stage_status(stage, "running")
    eval_summary = _read_if(store.path("eval", "final_eval.json"), {})
    prune = _read_if(store.path("prune", "prune_sweep.json"), {})
    geom = _read_if(store.path("geometry", "operator_geometry_summary.json"), {})
    usage = _read_if(store.path("usage", "operator_usage_summary.json"), {})
    trace = _read_if(store.path("trace", "trace_summary.json"), {})
    ablation = _read_if(store.path("ablation", "summary.json"), {})

    checklist = {
        "performance_retained": {
            "status": _status_for([store.path("eval", "final_eval.json")]),
            "evidence_files": [store.path("eval", "final_eval.json")],
        },
        "sparse_execution_works": {
            "status": _status_for([store.path("prune", "prune_sweep.json"), store.path("prune", "prune_curve.csv")]),
            "evidence_files": [store.path("prune", "prune_sweep.json"), store.path("prune", "prune_curve.csv")],
        },
        "operators_are_interpretable": {
            "status": _status_for([store.path("usage", "operator_usage_summary.json"), store.path("usage", "operator_cards.jsonl")]),
            "evidence_files": [store.path("usage", "operator_usage_summary.json"), store.path("usage", "operator_cards.jsonl")],
        },
        "operators_are_causal": {
            "status": _status_for([store.path("ablation", "summary.json"), store.path("ablation", "ablation_curve.csv")]),
            "evidence_files": [store.path("ablation", "summary.json"), store.path("ablation", "ablation_curve.csv")],
        },
        "scales_to_1p3B": {
            "status": _status_for([store.path("manifest.json"), store.path("model_info.json")]),
            "evidence_files": [store.path("manifest.json"), store.path("model_info.json")],
        },
    }
    figures = {
        "performance_table": store.path("report", "performance_table.csv"),
        "prune_curve": store.path("report", "prune_curve.csv"),
        "geometry_table": store.path("report", "geometry_table.csv"),
        "ablation_table": store.path("report", "ablation_table.csv"),
        "operator_card_examples": store.path("report", "operator_card_examples.md"),
    }
    summary_md = _summary_markdown(ctx, eval_summary, prune, geom, usage, trace, ablation, checklist)
    if store.is_primary:
        write_text_atomic(store.path("report", "summary.md"), summary_md)
        write_json_atomic(store.path("report", "claim_checklist.json"), checklist)
        write_csv_atomic(store.path("report", "performance_table.csv"), _performance_table(eval_summary))
        prune_rows = prune.get("eps", []) if isinstance(prune, dict) else []
        write_csv_atomic(store.path("report", "prune_curve.csv"), prune_rows)
        write_csv_atomic(store.path("report", "geometry_table.csv"), _geometry_table(geom))
        write_csv_atomic(store.path("report", "ablation_table.csv"), _ablation_table(ablation))
        cards_md = ""
        if exists(store.path("usage", "operator_cards_top.md")):
            with open_path(store.path("usage", "operator_cards_top.md"), "r") as f:
                cards_md = f.read()
        write_text_atomic(store.path("report", "operator_card_examples.md"), cards_md)
        write_json_atomic(store.path("report", "figures_manifest.json"), figures)
        result = {
            "claim_checklist": checklist,
            "figures_manifest": figures,
            "summary_path": store.path("report", "summary.md"),
        }
        store.mark_job_complete(stage, "report", store.path("report", "summary.md"), result)
        store.set_stage_status(stage, "complete")
        supported = sum(1 for rec in checklist.values() if rec["status"] == "supported")
        store.log_event(
            stage,
            "summary",
            message=f"REPORT SUMMARY supported_claims={supported}/{len(checklist)} path={store.path('report', 'summary.md')}",
            **result,
        )
        return result
    return {}
