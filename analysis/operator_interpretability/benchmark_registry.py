"""Pinned source-form contracts for interpretability benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from analysis.operator_interpretability.protocol import SUPPORTED_MODEL_VERSIONS


MIB_REFERENCE_REPOSITORY = "https://github.com/aaronmueller/MIB"
MIB_REFERENCE_REVISION = "b69dabe9899251d4a8fe90789afa4d655afc84c7"
MIB_CIRCUIT_REFERENCE_REVISION = "b759df34433c9e31043ba9e02908ce0bf20e894f"


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    title: str
    source_dataset: str
    source_config: str | None
    source_revision: str | None
    phase_splits: Mapping[str, str]
    expected_split_rows: Mapping[str, int]
    required_columns: tuple[str, ...]
    counterfactual_columns: tuple[str, ...]
    adapter_module: str
    track: str
    metric: str
    causal_variables: tuple[str, ...]
    reference_repository: str
    reference_revision: str
    reference_path: str
    source_row_limit: int | None = None
    source_shuffle_seed: int | None = None
    supported_model_versions: tuple[str, ...] = SUPPORTED_MODEL_VERSIONS

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase_splits"] = dict(self.phase_splits)
        value["expected_split_rows"] = dict(self.expected_split_rows)
        return value


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "mib_ioi": BenchmarkSpec(
        benchmark_id="mib_ioi",
        title="MIB Indirect Object Identification",
        source_dataset="mib-bench/ioi",
        source_config=None,
        source_revision="e5f3468f3af4c0883be35cd3bced8c711c95d286",
        phase_splits={
            "discovery": "train", "validation": "validation", "test": "test"},
        expected_split_rows={"train": 10_000, "validation": 10_000, "test": 1_000},
        required_columns=(
            "template", "metadata", "prompt", "choices", "answerKey",
            "s2_io_flip_counterfactual",
        ),
        counterfactual_columns=("s2_io_flip_counterfactual",),
        adapter_module="analysis.operator_interpretability.benchmarks.mib_ioi",
        track="mib_circuit_localization",
        metric="candidate_log_probability_difference",
        causal_variables=("ioi_task_output",),
        reference_repository=MIB_REFERENCE_REPOSITORY,
        reference_revision=MIB_CIRCUIT_REFERENCE_REVISION,
        reference_path="MIB_circuit_track/dataset.py:HFEAPDataset",
    ),
    "mib_mcqa": BenchmarkSpec(
        benchmark_id="mib_mcqa",
        title="MIB CopyColors MCQA (4 choices)",
        source_dataset="mib-bench/copycolors_mcqa",
        source_config="4_answer_choices",
        source_revision="682676b0e80d3a80e847a21810157710c1f23e27",
        phase_splits={
            "discovery": "train", "validation": "validation", "test": "test"},
        expected_split_rows={"train": 110, "validation": 50, "test": 50},
        required_columns=(
            "dataset", "dataset_specific_id", "choices", "answerKey",
            "prompt", "symbol_counterfactual", "idx",
        ),
        counterfactual_columns=("symbol_counterfactual",),
        adapter_module="analysis.operator_interpretability.benchmarks.mib_mcqa",
        track="mib_circuit_localization",
        metric="candidate_log_probability_difference",
        causal_variables=("answer_symbol",),
        reference_repository=MIB_REFERENCE_REPOSITORY,
        reference_revision=MIB_CIRCUIT_REFERENCE_REVISION,
        reference_path="MIB_circuit_track/dataset.py:HFEAPDataset",
    ),
    "mib_arithmetic": BenchmarkSpec(
        benchmark_id="mib_arithmetic",
        title="MIB arithmetic addition",
        source_dataset="mib-bench/arithmetic_addition",
        source_config=None,
        source_revision="d56d68b7b1d8f9c7e9a262ec1f89331ed52c7516",
        phase_splits={
            "discovery": "train", "validation": "validation", "test": "test"},
        expected_split_rows={"train": 34_440, "validation": 4_920, "test": 1_000},
        required_columns=(
            "idx", "template", "prompt", "label", "operator", "num_digit",
            "random_counterfactual", "operand1", "operand2",
        ),
        counterfactual_columns=("random_counterfactual",),
        adapter_module="analysis.operator_interpretability.benchmarks.mib_arithmetic",
        track="mib_circuit_localization",
        metric="candidate_log_probability_difference",
        causal_variables=("sum_output",),
        reference_repository=MIB_REFERENCE_REPOSITORY,
        reference_revision=MIB_CIRCUIT_REFERENCE_REVISION,
        reference_path="MIB_circuit_track/dataset.py:HFEAPDataset",
    ),
    "mib_arc": BenchmarkSpec(
        benchmark_id="mib_arc",
        title="MIB ARC Easy",
        source_dataset="mib-bench/arc_easy",
        source_config=None,
        source_revision="be6999b975e8348387cc9a21f3c8b6be8021e7d6",
        phase_splits={
            "discovery": "train", "validation": "validation", "test": "test"},
        expected_split_rows={"train": 2_251, "validation": 570, "test": 1_188},
        required_columns=(
            "arc_id", "question", "choices", "prompt", "label", "answerKey",
            "symbol_counterfactual", "idx",
        ),
        counterfactual_columns=("symbol_counterfactual",),
        adapter_module="analysis.operator_interpretability.benchmarks.mib_arc",
        track="mib_circuit_localization",
        metric="candidate_log_probability_difference",
        causal_variables=("answer_symbol",),
        reference_repository=MIB_REFERENCE_REPOSITORY,
        reference_revision=MIB_CIRCUIT_REFERENCE_REVISION,
        reference_path="MIB_circuit_track/dataset.py:HFEAPDataset",
    ),
    "ravel": BenchmarkSpec(
        benchmark_id="ravel",
        title="MIB RAVEL city attributes",
        source_dataset="mib-bench/ravel",
        source_config=None,
        source_revision="53dc7a4dbe84276567b895d8fb608b3f169b9276",
        phase_splits={"discovery": "train", "validation": "val", "test": "test"},
        expected_split_rows={"train": 100_347, "val": 15_950, "test": 1_000},
        required_columns=(
            "template", "prompt", "entity", "attribute", "Continent",
            "Country", "Language", "attribute_counterfactual",
            "wikipedia_counterfactual",
        ),
        counterfactual_columns=(
            "attribute_counterfactual", "wikipedia_counterfactual"),
        adapter_module="analysis.operator_interpretability.benchmarks.ravel",
        track="ravel_causal_variable_localization",
        metric="cause_and_isolation_interchange",
        causal_variables=("Continent", "Country", "Language"),
        reference_repository=MIB_REFERENCE_REPOSITORY,
        reference_revision=MIB_REFERENCE_REVISION,
        reference_path="MIB-causal-variable-track/baselines/RAVEL_baselines.py",
        source_row_limit=10_000,
        source_shuffle_seed=42,
    ),
    "blimp": BenchmarkSpec(
        benchmark_id="blimp",
        title="BLiMP grammatical minimal pairs",
        source_dataset="https://github.com/alexwarstadt/blimp",
        source_config="data/*.jsonl",
        source_revision="3e56b06fcabca9b30822fc66435fca6b1aa40bb1",
        phase_splits={"discovery": "train", "validation": "train", "test": "train"},
        expected_split_rows={"train": 67_000},
        required_columns=("sentence_good", "sentence_bad"),
        counterfactual_columns=("sentence_bad",),
        adapter_module="analysis.operator_interpretability.benchmarks.blimp",
        track="secondary_external_confirmation",
        metric="minimal_pair_log_probability",
        causal_variables=("linguistic_phenomenon",),
        reference_repository="https://github.com/alexwarstadt/blimp",
        reference_revision="3e56b06fcabca9b30822fc66435fca6b1aa40bb1",
        reference_path="data/*.jsonl",
    ),
    "counterfact": BenchmarkSpec(
        benchmark_id="counterfact",
        title="CounterFact factual recall",
        source_dataset="https://rome.baulab.info/data/dsets/counterfact.json",
        source_config=None,
        source_revision=(
            "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"),
        phase_splits={"discovery": "train", "validation": "train", "test": "train"},
        expected_split_rows={"train": 21_919},
        required_columns=("case_id", "requested_rewrite"),
        counterfactual_columns=(),
        adapter_module="analysis.operator_interpretability.benchmarks.counterfact",
        track="secondary_external_confirmation",
        metric="true_vs_counterfactual_object_log_probability",
        causal_variables=("relation",),
        reference_repository="https://rome.baulab.info",
        reference_revision=(
            "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"),
        reference_path="data/dsets/counterfact.json",
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
