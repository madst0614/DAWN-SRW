"""Concrete item and preset registry for ``train_analysis_pool``.

An item is the smallest independently addressable analysis result.  Dataset
selection is part of the item identity; there is deliberately no orthogonal
``--benchmarks`` execution axis.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from analysis.operator_interpretability.protocol import SUPPORTED_MODEL_VERSIONS
from dawn.eval.zero_shot_protocol import PRIMARY_TASKS


PRIMARY_MECHANISTIC_BENCHMARKS = (
    "mib_ioi", "mib_mcqa", "mib_arithmetic", "mib_arc", "ravel",
)
MIB_CIRCUIT_BENCHMARKS = (
    "mib_ioi", "mib_mcqa", "mib_arithmetic", "mib_arc",
)
SECONDARY_MECHANISTIC_BENCHMARKS = ("blimp", "counterfact")


def _item(
        title: str, question: str, standard: str, *, backend: str,
        analysis_kind: str, benchmark_id: str | None = None,
        task_id: str | None = None, requires: Iterable[str] = (),
        claim_role: str = "scientific") -> dict[str, Any]:
    if (benchmark_id is None) == (task_id is None) and analysis_kind != "scientific_claims":
        raise ValueError(
            "an item must identify exactly one benchmark or zero-shot task")
    return {
        "title": title,
        "scientific_question": question,
        "standard": standard,
        "backend": backend,
        "analysis_kind": analysis_kind,
        "benchmark_id": benchmark_id,
        "task_id": task_id,
        "requires": tuple(requires),
        "claim_role": claim_role,
        "supported_model_versions": SUPPORTED_MODEL_VERSIONS,
    }


def _mechanistic_item(
        benchmark_id: str, analysis_kind: str, title: str, question: str,
        standard: str, *requires: str,
        claim_role: str = "scientific") -> dict[str, Any]:
    return _item(
        title, question, standard,
        backend="operator_interpretability",
        analysis_kind=analysis_kind,
        benchmark_id=benchmark_id,
        requires=requires,
        claim_role=claim_role,
    )


TRAIN_ANALYSIS_POOL_ITEMS: dict[str, dict[str, Any]] = {}


def _register(item_id: str, definition: Mapping[str, Any]) -> None:
    if item_id in TRAIN_ANALYSIS_POOL_ITEMS:
        raise ValueError(f"duplicate train_analysis_pool item: {item_id}")
    TRAIN_ANALYSIS_POOL_ITEMS[item_id] = dict(definition)


for _benchmark_id in (*PRIMARY_MECHANISTIC_BENCHMARKS,
                      *SECONDARY_MECHANISTIC_BENCHMARKS):
    _claim_role = (
        "auxiliary" if _benchmark_id in SECONDARY_MECHANISTIC_BENCHMARKS
        else "scientific")
    _contract = f"{_benchmark_id}.input_contract"
    _behavior = f"{_benchmark_id}.behavioral_eligibility"
    _register(_contract, _mechanistic_item(
        _benchmark_id, "input_contract",
        f"{_benchmark_id} immutable input contract",
        "Are this task's source, tokenizer, phases, checkpoint, and protocol exactly identified?",
        "Pinned source revision and hash-verified immutable prepared artifact.",
        claim_role=_claim_role))
    _register(_behavior, _mechanistic_item(
        _benchmark_id, "behavioral_eligibility",
        f"{_benchmark_id} behavioral eligibility",
        "Does the frozen checkpoint solve enough paired examples for mechanistic analysis?",
        "Base must be correct, and labeled source behavior must also be correct when defined; no optimizer or parameter update.",
        _contract, claim_role=_claim_role))

for _benchmark_id in MIB_CIRCUIT_BENCHMARKS:
    _behavior = f"{_benchmark_id}.behavioral_eligibility"
    _localization = f"{_benchmark_id}.operator_localization"
    _conditional = f"{_benchmark_id}.conditional_circuit_sufficiency"
    _autonomous = f"{_benchmark_id}.autonomous_circuit_sufficiency"
    _register(_localization, _mechanistic_item(
        _benchmark_id, "operator_localization",
        f"{_benchmark_id} operator-site localization",
        "Which layer-route-operator sites carry captured production contribution?",
        "Discovery-only production-precision contribution ranking with captured-mass and split-rank audits.",
        _behavior))
    _register(_conditional, _mechanistic_item(
        _benchmark_id, "conditional_circuit_sufficiency",
        f"{_benchmark_id} conditional circuit sufficiency",
        "Can a nested circuit execute with the full production admission denominator?",
        "Select the smallest validation circuit whose bootstrap CI lower bound passes; evaluate test once.",
        _localization))
    _register(_autonomous, _mechanistic_item(
        _benchmark_id, "autonomous_circuit_sufficiency",
        f"{_benchmark_id} autonomous circuit sufficiency",
        "Can the same circuit execute using only its own admission denominator?",
        "Selected numerator and denominator are both restricted to the frozen validation-selected circuit.",
        _localization))
    _register(f"{_benchmark_id}.circuit_necessity", _mechanistic_item(
        _benchmark_id, "circuit_necessity",
        f"{_benchmark_id} circuit necessity",
        "Does suppressing the validation-selected circuit reduce held-out task margin?",
        "Circuit-wide numerator suppression with the full production denominator.",
        _conditional))

_register("mib_ioi.native_operator_program", _mechanistic_item(
    "mib_ioi", "native_operator_program",
    "IOI native operator program surgery",
    "Can each paired-correct IOI decision be replayed, ablated, and "
    "counterfactually transferred by its answer-position multilayer RW "
    "operator program?",
    "Select the smallest compact decision-position program mass on "
    "validation, then evaluate the frozen threshold once on held-out test; "
    "specificity and ID-only transfer claims require paired effects "
    "to exceed answer-disjoint mismatched and complement-sampled controls.",
    "mib_ioi.behavioral_eligibility",
    claim_role="checkpoint_specific"))

_register("ravel.operator_localization", _mechanistic_item(
    "ravel", "operator_localization", "RAVEL operator-site localization",
    "Which sites carry captured contribution on independent official RAVEL base groups?",
    "Discovery-only production contribution ranking; independent cause groups are balanced by causal variable and official source, and every variable must pass split-rank stability.",
    "ravel.behavioral_eligibility"))
_register("ravel.operator_space_structure", _mechanistic_item(
    "ravel", "operator_space_structure", "RAVEL operator-space structure",
    "Do read/write maps form reproducible local functional families independent of address?",
    "Runs only after every causal variable passes localization stability; normalized rank-one RW similarity and reciprocal local neighborhoods hold address out.",
    "ravel.operator_localization"))
_register("ravel.causal_mediation", _mechanistic_item(
    "ravel", "ravel_causal_mediation", "RAVEL causal mediation",
    "Can a discovered family transfer a target variable while isolating non-targets?",
    "Route-native interchange; family must beat seed-only and contribution-matched disjoint controls after BH correction.",
    "ravel.operator_space_structure"))
_register("ravel.multilayer_trajectory", _mechanistic_item(
    "ravel", "multilayer_trajectory", "RAVEL held-out multilayer trajectory",
    "Are same-variable operator paths more similar than matched cross-variable paths?",
    "Disjoint official-base triplets, captured-mass-qualified weighted Jaccard, paired null, and bootstrap interval.",
    "ravel.operator_localization"))

_scientific_requires = []
for _benchmark_id in MIB_CIRCUIT_BENCHMARKS:
    _scientific_requires.extend((
        f"{_benchmark_id}.autonomous_circuit_sufficiency",
        f"{_benchmark_id}.circuit_necessity",
    ))
_scientific_requires.extend((
    "ravel.causal_mediation", "ravel.multilayer_trajectory",
))
_register("scientific_claims.primary", _item(
    "Fail-closed primary scientific claim ladder",
    "What is the strongest causal claim supported by every pre-registered prerequisite?",
    "Localization, necessity, both sufficiency modes, interchange, isolation, and held-out gates.",
    backend="operator_interpretability",
    analysis_kind="scientific_claims",
    requires=_scientific_requires,
    claim_role="scientific"))

for _task_id in PRIMARY_TASKS:
    _register(f"zero_shot.{_task_id}", _item(
        f"Stock zero-shot: {_task_id}",
        f"What frozen-checkpoint zero-shot performance is obtained on {_task_id}?",
        "Stock lm-eval==0.4.2 task definition, zero few-shot examples, no optimizer, and no checkpoint write.",
        backend="stock_zero_shot",
        analysis_kind="zero_shot_task",
        task_id=_task_id,
        claim_role="auxiliary"))


def _mib_circuit_preset(benchmark_id: str) -> tuple[str, ...]:
    return (
        f"{benchmark_id}.conditional_circuit_sufficiency",
        f"{benchmark_id}.autonomous_circuit_sufficiency",
        f"{benchmark_id}.circuit_necessity",
    )


ZERO_SHOT_ITEMS = tuple(f"zero_shot.{task}" for task in PRIMARY_TASKS)
MECHANISTIC_SCREEN_ITEMS = tuple(
    f"{benchmark}.behavioral_eligibility"
    for benchmark in PRIMARY_MECHANISTIC_BENCHMARKS)

TRAIN_ANALYSIS_POOL_PRESETS: dict[str, tuple[str, ...]] = {
    "contract": tuple(
        f"{benchmark}.input_contract"
        for benchmark in PRIMARY_MECHANISTIC_BENCHMARKS),
    "zero_shot": ZERO_SHOT_ITEMS,
    "mechanistic_screen": MECHANISTIC_SCREEN_ITEMS,
    "ioi_native_program": ("mib_ioi.native_operator_program",),
    "mib_ioi_circuit": _mib_circuit_preset("mib_ioi"),
    "mib_mcqa_circuit": _mib_circuit_preset("mib_mcqa"),
    "mib_arithmetic_circuit": _mib_circuit_preset("mib_arithmetic"),
    "mib_arc_circuit": _mib_circuit_preset("mib_arc"),
    "circuit": tuple(
        item
        for benchmark in MIB_CIRCUIT_BENCHMARKS
        for item in _mib_circuit_preset(benchmark)),
    "ravel_causal": (
        "ravel.causal_mediation", "ravel.multilayer_trajectory"),
    "causal": (
        "ravel.causal_mediation", "ravel.multilayer_trajectory"),
    "scientific": ("scientific_claims.primary",),
    "all": (
        "scientific_claims.primary",
        "blimp.behavioral_eligibility",
        "counterfact.behavioral_eligibility",
        *ZERO_SHOT_ITEMS,
    ),
}

DEFAULT_TRAIN_ANALYSIS_POOL_PRESET = "scientific"


def item_definition(item_id: str) -> dict[str, Any]:
    try:
        return TRAIN_ANALYSIS_POOL_ITEMS[str(item_id)]
    except KeyError as exc:
        raise ValueError(f"unknown train_analysis_pool item: {item_id}") from exc


def parse_train_analysis_pool_items(
        preset: str | None, items: str | None) -> list[str]:
    if items:
        requested = [part.strip() for part in items.split(",") if part.strip()]
        selected = list(TRAIN_ANALYSIS_POOL_ITEMS) if requested == ["all"] else requested
    else:
        key = str(preset or DEFAULT_TRAIN_ANALYSIS_POOL_PRESET).strip()
        if key not in TRAIN_ANALYSIS_POOL_PRESETS:
            raise ValueError(
                f"unknown train_analysis_pool preset {key!r}; "
                f"known={','.join(TRAIN_ANALYSIS_POOL_PRESETS)}")
        selected = list(TRAIN_ANALYSIS_POOL_PRESETS[key])
    unknown = [item for item in selected if item not in TRAIN_ANALYSIS_POOL_ITEMS]
    if unknown:
        raise ValueError(
            f"unknown train_analysis_pool items: {','.join(unknown)}")
    return list(dict.fromkeys(selected))


def dependency_closure(items: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    visiting: list[str] = []

    def visit(item: str) -> None:
        definition = item_definition(item)
        if item in visiting:
            cycle = " -> ".join((*visiting, item))
            raise ValueError(
                f"train_analysis_pool dependency cycle: {cycle}")
        if item in ordered:
            return
        visiting.append(item)
        for dependency in definition["requires"]:
            visit(dependency)
        visiting.pop()
        ordered.append(item)

    for item in items:
        visit(item)
    return ordered


def items_for_backend(items: Iterable[str], backend: str) -> list[str]:
    return [
        item for item in items
        if item_definition(item)["backend"] == backend
    ]


def benchmark_ids_for_items(items: Iterable[str]) -> tuple[str, ...]:
    values = []
    for item in items:
        benchmark_id = item_definition(item).get("benchmark_id")
        if benchmark_id is not None:
            values.append(str(benchmark_id))
    return tuple(dict.fromkeys(values))


def train_analysis_pool_catalog() -> dict[str, Any]:
    return {
        "items": {
            item: {
                **definition,
                "requires": list(definition["requires"]),
                "supported_model_versions": list(
                    definition["supported_model_versions"]),
            }
            for item, definition in TRAIN_ANALYSIS_POOL_ITEMS.items()
        },
        "presets": {
            name: list(items)
            for name, items in TRAIN_ANALYSIS_POOL_PRESETS.items()
        },
    }
