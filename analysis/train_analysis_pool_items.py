"""Canonical item registry for the DAWN operator interpretability pool."""

from __future__ import annotations

from typing import Any, Iterable

from analysis.operator_interpretability.protocol import SUPPORTED_MODEL_VERSIONS


def _item(title: str, question: str, standard: str, *requires: str) -> dict[str, Any]:
    return {
        "title": title,
        "scientific_question": question,
        "standard": standard,
        "requires": tuple(requires),
        "supported_model_versions": SUPPORTED_MODEL_VERSIONS,
    }


TRAIN_ANALYSIS_POOL_ITEMS: dict[str, dict[str, Any]] = {
    "benchmark_contract": _item(
        "Immutable benchmark contract",
        "Are sources, tokenizer, phases, checkpoint, and protocol exactly identified?",
        "Pinned official MIB/RAVEL sources and hash-verified immutable artifacts."),
    "behavioral_eligibility": _item(
        "Behavioral eligibility",
        "Does the checkpoint solve enough examples to support mechanistic claims?",
        "Base and source must both have positive clean-label margins; RAVEL counts official base groups.",
        "benchmark_contract"),
    "operator_localization": _item(
        "Operator-site localization",
        "Which layer-route-operator sites carry captured production contribution?",
        "Discovery-only ranking by pre-cancellation production-precision contribution-vector norm, with adaptive captured-mass and split-rank audits.",
        "behavioral_eligibility"),
    "conditional_circuit_sufficiency": _item(
        "Conditional circuit sufficiency",
        "Can a nested operator-site circuit execute with the production admission denominator?",
        "MIB-adapted fractions; choose the smallest validation circuit whose bootstrap CI lower bound passes, then freeze test evaluation.",
        "operator_localization"),
    "autonomous_circuit_sufficiency": _item(
        "Autonomous circuit sufficiency",
        "Can the same circuit execute using only its own admission denominator?",
        "MIB-adapted fractions; choose the smallest validation circuit whose bootstrap CI lower bound passes, then freeze test evaluation.",
        "operator_localization"),
    "circuit_necessity": _item(
        "Circuit necessity",
        "Does suppressing the validation-selected circuit reduce held-out task margin?",
        "Circuit-wide numerator suppression with the full production denominator.",
        "conditional_circuit_sufficiency"),
    "operator_space_structure": _item(
        "Operator-space structure",
        "Do read/write maps form reproducible local functional families independent of address?",
        "Normalized rank-one RW similarity; reciprocal local neighborhoods; address held out.",
        "operator_localization"),
    "ravel_causal_mediation": _item(
        "RAVEL-style causal mediation",
        "Can a discovered operator family transfer a causal variable while isolating non-targets?",
        "Variable-specific route-native interchange; family must beat seed-only and contribution-matched disjoint controls after six-test BH correction.",
        "operator_space_structure"),
    "multilayer_trajectory": _item(
        "Held-out multilayer trajectory",
        "Are same-variable operator paths more similar than matched cross-variable paths?",
        "Disjoint official-RAVEL-base triplets, captured-mass-qualified weighted Jaccard, paired null, and bootstrap interval.",
        "operator_localization"),
    "scientific_claims": _item(
        "Fail-closed scientific claim ladder",
        "What is the strongest causal claim supported by all prerequisites?",
        "Pre-registered localization, necessity, sufficiency, interchange, isolation, and held-out gates.",
        "circuit_necessity", "autonomous_circuit_sufficiency",
        "ravel_causal_mediation", "multilayer_trajectory"),
}


TRAIN_ANALYSIS_POOL_PRESETS: dict[str, tuple[str, ...]] = {
    "contract": ("benchmark_contract",),
    "circuit": (
        "benchmark_contract", "behavioral_eligibility",
        "operator_localization", "conditional_circuit_sufficiency",
        "autonomous_circuit_sufficiency", "circuit_necessity"),
    "causal": (
        "benchmark_contract", "behavioral_eligibility",
        "operator_localization", "operator_space_structure",
        "ravel_causal_mediation"),
    "scientific": tuple(TRAIN_ANALYSIS_POOL_ITEMS),
}

DEFAULT_TRAIN_ANALYSIS_POOL_PRESET = "scientific"


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
        if item not in TRAIN_ANALYSIS_POOL_ITEMS:
            raise ValueError(f"unknown train_analysis_pool item: {item}")
        if item in visiting:
            cycle = " -> ".join((*visiting, item))
            raise ValueError(
                f"train_analysis_pool dependency cycle: {cycle}")
        if item in ordered:
            return
        visiting.append(item)
        for dependency in TRAIN_ANALYSIS_POOL_ITEMS[item]["requires"]:
            visit(dependency)
        visiting.pop()
        ordered.append(item)

    for item in items:
        visit(item)
    return ordered


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
