"""Canonical benchmark registry; no user-authored prompt sets are accepted."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from analysis.operator_interpretability.protocol import SUPPORTED_MODEL_VERSIONS


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    title: str
    source_dataset: str
    source_config: str | None
    split_map: Mapping[str, str]
    adapter_module: str
    adapter_name: str
    track: str
    metric: str
    counterfactual_column: str | None
    causal_variables: tuple[str, ...]
    supported_model_versions: tuple[str, ...] = SUPPORTED_MODEL_VERSIONS
    source_revision_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["split_map"] = dict(self.split_map)
        return value


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "mib_ioi": BenchmarkSpec(
        benchmark_id="mib_ioi",
        title="MIB Indirect Object Identification",
        source_dataset="mib-bench/ioi",
        source_config=None,
        split_map={"discovery": "train", "validation": "train", "test": "test"},
        adapter_module="analysis.operator_interpretability.benchmarks.mib_ioi",
        adapter_name="adapt_row",
        track="mib_adapted_operator_site_circuit",
        metric="logit_difference",
        counterfactual_column="s2_io_flip_counterfactual",
        causal_variables=("output_token", "output_position"),
    ),
    "mib_mcqa": BenchmarkSpec(
        benchmark_id="mib_mcqa",
        title="MIB CopyColors MCQA",
        source_dataset="mib-bench/copycolors_mcqa",
        source_config="4_answer_choices",
        split_map={
            "discovery": "train", "validation": "validation", "test": "test"},
        adapter_module="analysis.operator_interpretability.benchmarks.mib_mcqa",
        adapter_name="adapt_row",
        track="mib_adapted_operator_site_circuit",
        metric="logit_difference",
        counterfactual_column="symbol_counterfactual",
        causal_variables=("answer_pointer", "answer"),
    ),
    "mib_arithmetic": BenchmarkSpec(
        benchmark_id="mib_arithmetic",
        title="MIB two-digit arithmetic addition",
        source_dataset="mib-bench/arithmetic_addition",
        source_config=None,
        split_map={"discovery": "train", "validation": "train", "test": "test"},
        adapter_module="analysis.operator_interpretability.benchmarks.mib_arithmetic",
        adapter_name="adapt_row",
        track="mib_adapted_operator_site_circuit",
        metric="logit_difference",
        counterfactual_column="random_counterfactual",
        causal_variables=("ones_result", "tens_result", "tens_carry"),
    ),
    "mib_arc": BenchmarkSpec(
        benchmark_id="mib_arc",
        title="MIB ARC Easy",
        source_dataset="mib-bench/arc_easy",
        source_config=None,
        split_map={
            "discovery": "train", "validation": "validation", "test": "test"},
        adapter_module="analysis.operator_interpretability.benchmarks.mib_arc",
        adapter_name="adapt_row",
        track="mib_adapted_operator_site_circuit",
        metric="logit_difference",
        counterfactual_column="symbol_counterfactual",
        causal_variables=("answer_pointer", "answer"),
    ),
    "ravel": BenchmarkSpec(
        benchmark_id="ravel",
        title="RAVEL city attribute mediation",
        source_dataset="mib-bench/ravel",
        source_config=None,
        split_map={"discovery": "train", "validation": "val", "test": "test"},
        adapter_module="analysis.operator_interpretability.benchmarks.ravel",
        adapter_name="adapt_pairs",
        track="ravel_style_operator_contribution_mediation",
        metric="cause_and_isolation",
        counterfactual_column=None,
        causal_variables=("Continent", "Country", "Language"),
    ),
    "blimp": BenchmarkSpec(
        benchmark_id="blimp",
        title="BLiMP grammatical minimal pairs",
        source_dataset="https://github.com/alexwarstadt/blimp",
        source_config="data/*.jsonl",
        split_map={"discovery": "train", "validation": "train", "test": "train"},
        adapter_module="analysis.operator_interpretability.benchmarks.blimp",
        adapter_name="adapt_row",
        track="secondary_external_confirmation",
        metric="minimal_pair_log_probability",
        counterfactual_column="sentence_bad",
        causal_variables=("linguistic_phenomenon",),
    ),
    "counterfact": BenchmarkSpec(
        benchmark_id="counterfact",
        title="CounterFact factual recall",
        source_dataset="https://rome.baulab.info/data/dsets/counterfact.json",
        source_config=None,
        split_map={"discovery": "train", "validation": "train", "test": "train"},
        adapter_module="analysis.operator_interpretability.benchmarks.counterfact",
        adapter_name="adapt_row",
        track="secondary_external_confirmation",
        metric="true_vs_counterfactual_object_log_probability",
        counterfactual_column=None,
        causal_variables=("relation",),
    ),
}

PRIMARY_BENCHMARK_IDS = (
    "mib_ioi", "mib_mcqa", "mib_arithmetic", "mib_arc", "ravel",
)
SECONDARY_BENCHMARK_IDS = ("blimp", "counterfact")


def benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    key = str(benchmark_id).strip().lower()
    try:
        return BENCHMARK_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown interpretability benchmark {benchmark_id!r}; "
            f"known={','.join(BENCHMARK_SPECS)}") from exc


def assert_benchmark_support(benchmark_id: str, model_version: str) -> None:
    spec = benchmark_spec(benchmark_id)
    if model_version not in spec.supported_model_versions:
        raise ValueError(
            f"{benchmark_id} does not support {model_version}; "
            f"supported={','.join(spec.supported_model_versions)}")


def registry_record() -> dict[str, Any]:
    return {key: spec.to_dict() for key, spec in BENCHMARK_SPECS.items()}
