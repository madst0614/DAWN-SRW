#!/usr/bin/env python3
"""Create the paper's reproducible DAWN exact-support FLOP artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from analysis.dawn_analysis_common import git_info
from analysis.dawn_analysis_storage import (
    join_path,
    open_path,
    read_json,
    write_bytes_atomic,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from analysis.operator_interpretability.benchmark_schema import canonical_hash
from analysis.paper_compute import compute_flop_accounting


def _read_yaml(path: str) -> dict[str, Any]:
    with open_path(path, "r") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return value


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open_path(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _component_rows(accounting: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column_id, column in accounting["columns"].items():
        for component, flops in column["components_flops"].items():
            rows.append({
                "column": column_id,
                "component": component,
                "macs": column["components_macs"][component],
                "flops": flops,
                "gflops": float(flops) / 1.0e9,
            })
        rows.append({
            "column": column_id,
            "component": "TOTAL_BACKBONE",
            "macs": column["total_macs"],
            "flops": column["total_flops"],
            "gflops": float(column["total_flops"]) / 1.0e9,
        })
    return rows


def _paper_table_rows(accounting: Mapping[str, Any]) -> list[dict[str, Any]]:
    parameters = accounting["parameter_counts"]
    columns = accounting["columns"]
    current = columns["dawn_current_dense_execution"]
    exact = columns["dawn_exact_support_estimate"]
    dense = columns["dense_transformer"]
    return [
        {
            "model": "Dense Transformer",
            "parameter_count": parameters["dense_baseline"],
            "address_execution": "none",
            "rw_execution": "none",
            "backbone_gflops": dense["total_flops"] / 1.0e9,
            "output_equivalence": "reference",
            "measured_latency": "",
        },
        {
            "model": "DAWN current",
            "parameter_count": parameters["dawn"],
            "address_execution": "full pool",
            "rw_execution": "full pool dense contraction",
            "backbone_gflops": current["total_flops"] / 1.0e9,
            "output_equivalence": "reference",
            "measured_latency": "",
        },
        {
            "model": "DAWN exact-support estimate",
            "parameter_count": parameters["dawn"],
            "address_execution": "full pool",
            "rw_execution": "measured margin>0 support",
            "backbone_gflops": exact["total_flops"] / 1.0e9,
            "output_equivalence": "mathematically exact",
            "measured_latency": "not implemented",
        },
    ]


def _figure_bytes(accounting: Mapping[str, Any], suffix: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = accounting["columns"]
    ordered = (
        ("dense_transformer", "Dense\nTransformer"),
        ("dawn_current_dense_execution", "DAWN\ncurrent"),
        ("dawn_exact_support_estimate", "DAWN\nexact support"),
    )
    component_order = (
        "qkv_and_output_projections",
        "ffn",
        "route_and_tau_projections",
        "full_pool_address_scoring",
        "full_pool_rw_application",
        "exact_support_rw_application",
        "causal_attention",
        "attention_output_projection",
    )
    labels = {
        "qkv_and_output_projections": "QKV/O projections",
        "ffn": "FFN",
        "route_and_tau_projections": "Query/tau projections",
        "full_pool_address_scoring": "Full-pool address scoring",
        "full_pool_rw_application": "Full-pool RW application",
        "exact_support_rw_application": "Exact-support RW application",
        "causal_attention": "Causal attention",
        "attention_output_projection": "DAWN output projection",
    }
    colors = {
        "qkv_and_output_projections": "#4C78A8",
        "ffn": "#72B7B2",
        "route_and_tau_projections": "#F2CF5B",
        "full_pool_address_scoring": "#ECA82C",
        "full_pool_rw_application": "#E45756",
        "exact_support_rw_application": "#B279A2",
        "causal_attention": "#59A14F",
        "attention_output_projection": "#9D755D",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x_positions = list(range(len(ordered)))
    bottoms = [0.0] * len(ordered)
    for component in component_order:
        values = [
            float(columns[column_id]["components_flops"].get(component, 0.0))
            / 1.0e12
            for column_id, _ in ordered
        ]
        if not any(value > 0.0 for value in values):
            continue
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            width=0.66,
            label=labels[component],
            color=colors[component],
        )
        bottoms = [
            bottom + value for bottom, value in zip(bottoms, values)]
    for x, total in zip(x_positions, bottoms):
        ax.text(
            x, total + max(bottoms) * 0.015,
            f"{total:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(
        x_positions, [label for _, label in ordered])
    ax.set_ylabel("Major forward FLOPs / model call (TFLOPs)")
    ax.set_title(
        "Current dense execution vs measured exact-support opportunity")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format=suffix, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def run_accounting(
        *, dawn_config_path: str, baseline_config_path: str,
        support_json_path: str, output_dir: str,
        batch_size: int, sequence_length: int,
        dawn_parameter_count: int | None = None,
        baseline_parameter_count: int | None = None,
        write_figure: bool = True) -> dict[str, Any]:
    dawn_config = _read_yaml(dawn_config_path)
    baseline_config = _read_yaml(baseline_config_path)
    support_summary = read_json(support_json_path)
    if not isinstance(support_summary, dict):
        raise ValueError(
            f"support summary is not a mapping: {support_json_path}")
    accounting = compute_flop_accounting(
        dawn_config,
        baseline_config,
        support_summary,
        batch_size=batch_size,
        sequence_length=sequence_length,
        dawn_parameter_count=dawn_parameter_count,
        baseline_parameter_count=baseline_parameter_count,
    )
    accounting.pop("result_hash", None)
    accounting["identity"] = {
        "dawn_config": dawn_config_path,
        "dawn_config_sha256": _sha256_path(dawn_config_path),
        "baseline_config": baseline_config_path,
        "baseline_config_sha256": _sha256_path(baseline_config_path),
        "support_json": support_json_path,
        "support_json_sha256": _sha256_path(support_json_path),
        "accounting_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "accounting_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "analysis_code": git_info(),
    }
    accounting["result_hash"] = canonical_hash(accounting)

    write_json_atomic(
        join_path(output_dir, "flop_accounting.json"), accounting)
    write_csv_atomic(
        join_path(output_dir, "flop_components.csv"),
        _component_rows(accounting))
    write_csv_atomic(
        join_path(output_dir, "computational_characteristics.csv"),
        _paper_table_rows(accounting))
    if write_figure:
        write_bytes_atomic(
            join_path(output_dir, "flops_stacked.png"),
            _figure_bytes(accounting, "png"),
            content_type="image/png",
        )
        write_bytes_atomic(
            join_path(output_dir, "flops_stacked.pdf"),
            _figure_bytes(accounting, "pdf"),
            content_type="application/pdf",
        )

    comparisons = accounting["comparisons"]
    summary_lines = [
        "PAPER FLOP ACCOUNTING",
        f"result_hash={accounting['result_hash']}",
        f"batch_size={batch_size} sequence_length={sequence_length}",
        "1_MAC=2_FLOPs",
        (
            "current_dawn_backbone_TFLOPs="
            f"{accounting['columns']['dawn_current_dense_execution']['total_flops'] / 1e12:.9f}"
        ),
        (
            "exact_support_backbone_TFLOPs="
            f"{accounting['columns']['dawn_exact_support_estimate']['total_flops'] / 1e12:.9f}"
        ),
        (
            "dense_transformer_backbone_TFLOPs="
            f"{accounting['columns']['dense_transformer']['total_flops'] / 1e12:.9f}"
        ),
        (
            "current_dawn_vs_dense="
            f"{comparisons['current_dawn_vs_dense_transformer']:.9f}"
        ),
        (
            "exact_support_vs_current="
            f"{comparisons['exact_support_vs_current_dawn']:.9f}"
        ),
        (
            "exact_support_vs_dense="
            f"{comparisons['exact_support_vs_dense_transformer']:.9f}"
        ),
        "indexed_addressing_included=false",
        "measured_latency_claimed=false",
    ]
    write_text_atomic(
        join_path(output_dir, "flop_accounting.log"),
        "\n".join(summary_lines) + "\n",
    )
    return accounting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dawn-config", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--dawn-parameter-count", type=int, default=None)
    parser.add_argument("--baseline-parameter-count", type=int, default=None)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_accounting(
        dawn_config_path=args.dawn_config,
        baseline_config_path=args.baseline_config,
        support_json_path=args.support_json,
        output_dir=args.output,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        dawn_parameter_count=args.dawn_parameter_count,
        baseline_parameter_count=args.baseline_parameter_count,
        write_figure=not args.no_figure,
    )
    print(
        "PAPER_FLOP_ACCOUNTING_COMPLETE "
        f"result_hash={result['result_hash']} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
