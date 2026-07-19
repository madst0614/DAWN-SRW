"""Self-contained text/plain reporting for one ``train_analysis_pool`` run.

Compact JSON files retain machine-readable protocol records.  ``summary.log``
is the complete human interpretation surface: it mirrors every completed item
to stdout and records the run identity, preregistered standards, aggregate
metrics, uncertainty, blockers, and the final claim decision.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from analysis.dawn_analysis_storage import (
    AnalysisStore,
    write_text_atomic,
)
from analysis.train_analysis_pool_items import item_definition


_LINE = "=" * 72
_ITEM_LINE = "-" * 72
_MAX_RESULT_LINES = 120
_DETAIL_ONLY_KEYS = {
    "base_margin",
    "base_negative_logp",
    "base_positive_logp",
    "base_known_correct",
    "capture",
    "causal_variable_profile_summary",
    "checkpoint_identity_record",
    "controls",
    "corrupted_margin",
    "evidence",
    "example_ids",
    "known_correct",
    "parameter_schema",
    "per_variable",
    "pool_provenance",
    "protocol",
    "ranked_site_preview",
    "rows",
    "site_summaries",
    "selected_units_by_causal_variable",
    "source_behavior_scored",
    "source_known_correct",
    "source_own_margin",
    "triplet_group_ids",
    "upstream_item_artifacts",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.8g}"
    if isinstance(value, str):
        compact = " ".join(value.splitlines())
        return compact if len(compact) <= 240 else compact[:237] + "..."
    return str(value)


def _sequence_summary(values: Sequence[Any]) -> str:
    count = len(values)
    if count == 0:
        return "count=0"
    if all(isinstance(value, bool) for value in values):
        true_count = sum(bool(value) for value in values)
        return f"count={count} true={true_count} false={count - true_count}"
    if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values):
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return f"count={count} finite=0"
        return (
            f"count={count} min={_scalar(min(finite))} "
            f"max={_scalar(max(finite))} "
            f"mean={_scalar(sum(finite) / len(finite))}")
    if all(value is None or isinstance(value, (str, int, float, bool))
           for value in values) and count <= 8:
        return json.dumps(list(values), ensure_ascii=False)
    return f"count={count} nonessential_values_omitted=true"


def _ordered_mapping_items(value: Mapping[str, Any]):
    items = list(value.items())
    return sorted(items, key=lambda item: (
        item[0] in _DETAIL_ONLY_KEYS,
        isinstance(item[1], (list, tuple)),
        isinstance(item[1], Mapping),
    ))


def _result_lines(result: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    omitted = 0

    def add(path: str, value: Any, depth: int) -> None:
        nonlocal omitted
        if len(lines) >= _MAX_RESULT_LINES:
            omitted += 1
            return
        if path.rsplit(".", 1)[-1] in _DETAIL_ONLY_KEYS:
            if isinstance(value, Mapping):
                description = (
                    f"keys={len(value)} nonessential_values_omitted=true")
            elif isinstance(value, (list, tuple)):
                description = _sequence_summary(value)
            else:
                description = "nonessential_value_omitted=true"
            lines.append(f"  {path}: {description}")
            return
        if isinstance(value, Mapping):
            if not value:
                lines.append(f"  {path}: {{}}")
                return
            if depth >= 6:
                lines.append(
                    f"  {path}: keys={len(value)} "
                    "nonessential_values_omitted=true")
                return
            for key, child in _ordered_mapping_items(value):
                child_path = f"{path}.{key}" if path else str(key)
                add(child_path, child, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            if value and len(value) <= 10 and all(
                    isinstance(child, Mapping) for child in value):
                for index, child in enumerate(value):
                    add(f"{path}[{index}]", child, depth + 1)
                return
            lines.append(f"  {path}: {_sequence_summary(value)}")
            return
        lines.append(f"  {path}: {_scalar(value)}")

    for key, value in _ordered_mapping_items(result):
        add(str(key), value, 0)
    if omitted:
        lines.append(
            "  ... additional nonessential machine-audit detail omitted")
    return lines


def _paired_trajectory_lines(result: Mapping[str, Any]) -> list[str]:
    lines = ["Paired S2 operator trajectory summary:"]
    baseline = result.get("behavioral_baseline")
    if isinstance(baseline, Mapping):
        lines.extend([
            "  Behavioral baseline:",
            "    phase | paired-correct | base accuracy | source accuracy | base margin | source margin",
        ])
        for phase in ("discovery", "validation"):
            row = baseline.get(phase)
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "    " + " | ".join((
                    phase,
                    _scalar(row.get("paired_correct_used")),
                    _scalar(row.get("base_accuracy")),
                    _scalar(row.get("source_accuracy")),
                    _scalar(row.get("base_margin_mean")),
                    _scalar(row.get("source_own_margin_mean")),
                )))
    trace = result.get("full_active_trace")
    if isinstance(trace, Mapping):
        completeness = trace.get("trace_completeness")
        if isinstance(completeness, Mapping):
            lines.extend([
                "  Trace completeness:",
                "    route | initial K | final K | max active | omitted | retries | replay max error",
            ])
            for route in ("q", "k", "v", "rst"):
                row = completeness.get(route)
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "    " + " | ".join((
                        route,
                        _scalar(row.get("initial_width")),
                        _scalar(row.get("final_width")),
                        _scalar(row.get("max_active_count")),
                        _scalar(row.get("omitted_active_count")),
                        _scalar(row.get("retry_count")),
                        _scalar(row.get("all_active_replay_error")),
                    )))
    execution = result.get("tpu_execution")
    if isinstance(execution, Mapping):
        lines.append(
            "  TPU forward calls (successful): "
            f"{_scalar(execution.get('successful_forward_call_count'))}")
        stage_counts = execution.get("stage_forward_call_count")
        if isinstance(stage_counts, Mapping):
            lines.extend(
                f"    {stage}: {_scalar(count)}"
                for stage, count in stage_counts.items())
    divergence = result.get("divergence_atlas")
    if isinstance(divergence, Mapping) and isinstance(
            divergence.get("extrema"), Mapping):
        lines.extend([
            "  Divergence atlas:",
            "    event | layer | role | route | effect",
        ])
        for name, row in divergence["extrema"].items():
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "    " + " | ".join((
                    str(name), _scalar(row.get("layer")),
                    _scalar(row.get("semantic_role")),
                    _scalar(row.get("route")),
                    _scalar(row.get("effect_size")),
                )))
    causal = result.get("discovery_causal_patch")
    if isinstance(causal, Mapping):
        rows = [
            row for row in causal.get("site_summaries", ())
            if isinstance(row, Mapping) and row.get("patch_kind") == "route"
        ]
        rows.sort(key=lambda row: -float(
            row.get("bidirectional_specific_effect_mean", 0.0)))
        if rows:
            lines.extend([
                "  Causal complete-route patch (top discovery sites):",
                "    site | source-to-base shift | base-to-source shift | paired flip | mismatch shift | self max error",
            ])
            for row in rows[:8]:
                directions = row.get("directions") or {}
                sb = directions.get("source_to_base") or {}
                bs = directions.get("base_to_source") or {}
                lines.append(
                    "    " + " | ".join((
                        f"L{row.get('layer')}:{row.get('semantic_role')}:{row.get('route')}",
                        _scalar(sb.get("paired_margin_shift_mean")),
                        _scalar(bs.get("paired_margin_shift_mean")),
                        _scalar((float(sb.get("paired_answer_flip_fraction", 0.0))
                                 + float(bs.get("paired_answer_flip_fraction", 0.0))) / 2.0),
                        _scalar(row.get("bidirectional_mismatched_effect_mean")),
                        _scalar(max(
                            float(sb.get("self_reconstruction_max_abs", 0.0)),
                            float(bs.get("self_reconstruction_max_abs", 0.0)))),
                    )))
    operator = result.get("operator_decomposition")
    if isinstance(operator, Mapping):
        rows = [
            row for row in operator.get("site_summaries", ())
            if isinstance(row, Mapping)
        ]
        rows.sort(key=lambda row: (
            int(row.get("candidate_index", 0)),
            str(row.get("group_kind", ""))))
        if rows:
            lines.extend([
                "  Operator group decomposition:",
                "    site | group | common fraction | base-only | source-only | paired shift | specific shift",
            ])
            for row in rows[:24]:
                lines.append(
                    "    " + " | ".join((
                        f"L{row.get('layer')}:{row.get('semantic_role')}:{row.get('route')}",
                        _scalar(row.get("group_kind")),
                        _scalar(row.get("common_active_fraction_mean")),
                        _scalar(row.get("base_only_count_mean")),
                        _scalar(row.get("source_only_count_mean")),
                        _scalar(row.get("bidirectional_paired_effect_mean")),
                        _scalar(row.get("bidirectional_specific_effect_mean")),
                    )))
    human = result.get("human_summary")
    if isinstance(human, Mapping):
        lines.extend([
            "  Frozen validation path:",
            f"    selection hash: {_scalar(human.get('frozen_path_hash'))}",
            f"    path length: {_scalar(human.get('frozen_path_length'))}",
            f"    paired effect: {_scalar(human.get('validation_paired_effect'))}",
            f"    mismatched effect: {_scalar(human.get('validation_mismatched_effect'))}",
            f"    paired flip rate: {_scalar(human.get('validation_flip_fraction'))}",
            f"    test consulted: {_scalar(human.get('test_consulted'))}",
        ])
        narrative = human.get("narrative")
        if isinstance(narrative, Sequence) and not isinstance(
                narrative, (str, bytes)):
            lines.append("  Interpretation:")
            lines.extend(f"    {str(value)}" for value in narrative)
    return lines


def format_item_text(payload: Mapping[str, Any], *, artifact_path: str,
                     event: str) -> str:
    item_id = str(payload.get("item_id") or "unknown")
    definition = item_definition(item_id)
    result = payload.get("result")
    if not isinstance(result, Mapping):
        result = {}
    lines = [
        _ITEM_LINE,
        f"TRAIN_ANALYSIS_POOL ITEM {item_id}",
        f"  event: {event}",
        f"  status: {_scalar(payload.get('status'))}",
        f"  title: {definition['title']}",
        f"  scientific_question: {definition['scientific_question']}",
        f"  decision_standard: {definition['standard']}",
        f"  claim_role: {definition['claim_role']}",
        f"  scientific_role: {definition.get('scientific_role') or 'not_set'}",
        f"  backend: {payload.get('backend')}",
        f"  analysis_kind: {payload.get('analysis_kind')}",
    ]
    if payload.get("test_used") is not None:
        lines.append(f"  test_used: {_scalar(payload.get('test_used'))}")
    if payload.get("benchmark_id") is not None:
        lines.append(f"  benchmark: {payload.get('benchmark_id')}")
    if payload.get("task_id") is not None:
        lines.append(f"  task: {payload.get('task_id')}")
    artifact_warnings = payload.get("artifact_warnings")
    if isinstance(artifact_warnings, Sequence) and not isinstance(
            artifact_warnings, (str, bytes)) and artifact_warnings:
        lines.extend([
            "Warnings:",
            *_result_lines({"artifact_warnings": list(artifact_warnings)}),
        ])
    if payload.get("analysis_kind") == "native_operator_program":
        human_summary = result.get("human_summary")
        if isinstance(human_summary, Mapping):
            lines.extend([
                "Native operator program decision:",
                *_result_lines(human_summary),
            ])
    if payload.get("analysis_kind") == "paired_operator_trajectory":
        lines.extend(_paired_trajectory_lines(result))
    lines.extend([
        f"  artifact: {artifact_path}",
        "Result:",
        *_result_lines(result),
        _ITEM_LINE,
    ])
    return "\n".join(lines)


class TrainAnalysisPoolTextReporter:
    """Mirror canonical item payloads to stdout and checkpoint-side text."""

    def __init__(
            self, store: AnalysisStore, *, preset: str,
            requested_items: Sequence[str], executed_items: Sequence[str],
            target_id: str | None, runtime_id: str, checkpoint_path: str,
            checkpoint_step: int, run_id: str, run_label: str,
            benchmark_build_id: str | None = None,
            benchmark_manifest_path: str | None = None,
            checkpoint_config_hash: str | None = None,
            max_item_json_bytes: int | None = None,
            mechanistic_protocol_config: Mapping[str, Any] | None = None
    ) -> None:
        self.store = store
        self.report_path = store.path("summary.log")
        self._blocks: dict[str, str] = {}
        self._order: list[str] = []
        self._statuses: dict[str, Any] = {}
        self._footer: list[str] = []
        self._header = [
            _LINE,
            "DAWN TRAIN ANALYSIS POOL",
            _LINE,
            f"started_at: {_utc_now()}",
            f"run_id: {run_id}",
            f"run_label: {run_label}",
            f"run_root: {store.output_dir}",
            f"preset: {preset}",
            f"target: {target_id or 'ad-hoc'}",
            f"runtime: {runtime_id}",
            f"checkpoint: {checkpoint_path}",
            f"checkpoint_step: {int(checkpoint_step)}",
            f"checkpoint_config_hash: {checkpoint_config_hash or 'not_available'}",
            f"benchmark_build_id: {benchmark_build_id or 'not_applicable'}",
            "benchmark_manifest: "
            f"{benchmark_manifest_path or 'not_applicable'}",
            f"requested_items: {','.join(requested_items)}",
            f"executed_items: {','.join(executed_items)}",
            "mechanistic_protocol_config: " + json.dumps(
                dict(mechanistic_protocol_config or {}),
                sort_keys=True, ensure_ascii=False),
            f"protocol_bound_item_json_warning_bytes: "
            f"{max_item_json_bytes or 'not_applicable'}",
            f"json_summary: {store.path('summary.json')}",
            f"summary_log: {self.report_path}",
            "",
        ]

    def _render(self) -> str:
        lines = list(self._header)
        for item_id in self._order:
            lines.extend((self._blocks[item_id], ""))
        lines.extend(self._footer)
        return "\n".join(lines).rstrip() + "\n"

    def _write(self, text: str) -> None:
        write_text_atomic(
            self.report_path, text,
            content_type="text/plain; charset=utf-8",
            content_disposition="inline",
            require_metadata=True)

    def start(self) -> None:
        if not self.store.is_primary:
            return
        text = "\n".join(self._header).rstrip()
        print(text, flush=True)
        self._write(self._render())

    def emit(self, payload: Mapping[str, Any], *, artifact_path: str,
             event: str) -> None:
        if not self.store.is_primary:
            return
        item_id = str(payload.get("item_id") or "unknown")
        block = format_item_text(
            payload, artifact_path=artifact_path, event=event)
        if item_id not in self._blocks:
            self._order.append(item_id)
        self._blocks[item_id] = block
        self._statuses[item_id] = payload.get("status")
        print(block, flush=True)
        self._write(self._render())

    def finish(self, summary: Mapping[str, Any]) -> None:
        if not self.store.is_primary:
            return
        item_status = dict(self._statuses)
        item_status.update(dict(summary.get("item_status") or {}))
        blockers = [
            (item_id, status) for item_id, status in item_status.items()
            if status not in {"ready", "complete", "passed", "selected"}
        ]
        artifact_warnings = [
            dict(row) for row in summary.get("artifact_warnings", ())
            if isinstance(row, Mapping)
        ]
        artifact_warning_lines = [
            "  "
            f"{row.get('code')}: path={row.get('path')} "
            f"bytes={row.get('encoded_bytes')} "
            f"threshold={row.get('warning_threshold_bytes')} "
            f"write_continued={str(bool(row.get('write_continued'))).lower()}"
            for row in artifact_warnings
        ]
        self._footer = [
            _LINE,
            "TRAIN_ANALYSIS_POOL COMPLETE",
            f"completed_at: {_utc_now()}",
            f"status: {summary.get('status')}",
            f"strongest_supported_claim: "
            f"{summary.get('strongest_supported_claim') or 'not_evaluated'}",
            f"item_count: {len(self._order)}",
            "run_semantics: independent_checkpoint_step_preset_invocation",
            f"model_version: {summary.get('model_version')}",
            "effective_mesh: " + json.dumps(
                summary.get("effective_mesh"), sort_keys=True,
                ensure_ascii=False),
            "checkpoint_mesh: " + json.dumps(
                summary.get("checkpoint_mesh"), sort_keys=True,
                ensure_ascii=False),
            "item_status:",
            *[
                f"  {item_id}: {status}"
                for item_id, status in item_status.items()
            ],
            "interpretation_blockers:",
            *(
                [f"  {item_id}: {status}" for item_id, status in blockers]
                if blockers else ["  none"]
            ),
            f"artifact_warning_count: {len(artifact_warnings)}",
            "artifact_warnings:",
            *(artifact_warning_lines or ["  none"]),
            f"json_summary: {self.store.path('summary.json')}",
            f"summary_log: {self.report_path}",
            "raw_parameters_persisted: false",
            "dense_capture_rows_persisted: false",
            "interpretation_contract: summary.log contains all durable "
            "human-facing metrics; item JSON is compact machine-readable audit data",
            _LINE,
        ]
        text = self._render()
        self._write(text)
        print(
            "TRAIN_ANALYSIS_POOL saved "
            f"summary_log={self.report_path}",
            flush=True,
        )

    def fail(self, error: Exception) -> None:
        if not self.store.is_primary:
            return
        self._footer = [
            _LINE,
            "TRAIN_ANALYSIS_POOL FAILED",
            f"failed_at: {_utc_now()}",
            "status: failed",
            f"error_type: {type(error).__name__}",
            f"error: {_scalar(str(error))}",
            f"completed_item_count: {len(self._order)}",
            "completed_item_status:",
            *(
                [
                    f"  {item_id}: {self._statuses.get(item_id)}"
                    for item_id in self._order
                ] if self._order else ["  none"]
            ),
            "run_restart_contract: launch a new independent run; partial "
            "artifacts are never resumed",
            f"summary_log: {self.report_path}",
            "raw_parameters_persisted: false",
            "dense_capture_rows_persisted: false",
            _LINE,
        ]
        self._write(self._render())
        print(
            "TRAIN_ANALYSIS_POOL failed "
            f"error={type(error).__name__} "
            f"summary_log={self.report_path}",
            flush=True,
        )
