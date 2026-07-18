"""Fail-closed claim ladder; metrics never silently become conclusions."""

from __future__ import annotations

from typing import Any, Mapping

from analysis.operator_interpretability.protocol import ProtocolConfig


def _ready(result: Mapping[str, Any] | None) -> bool:
    return bool(result) and result.get("status") in {"ready", "passed", "selected"}


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def evaluate_claims(results: Mapping[str, Any],
                    config: ProtocolConfig) -> dict[str, Any]:
    config.validate()
    capture = dict(results.get("capture") or {})
    localization = dict(results.get("localization") or {})
    necessity = dict(results.get("necessity") or {})
    conditional = dict(results.get("conditional_sufficiency") or {})
    autonomous = dict(results.get("autonomous_sufficiency") or {})
    interchange = dict(results.get("interchange") or {})
    held_out = dict(results.get("held_out") or {})
    spatial = dict(results.get("spatial_confirmation") or {})
    trajectory = dict(results.get("trajectory_confirmation") or {})

    capture_ok = (
        _ready(capture)
        and _number(capture.get("qualified_fraction"), 0.0)
        >= config.capture_threshold
        and _number(capture.get("rank_stability"), 0.0)
        >= config.rank_stability_min)
    claims: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, prerequisites: list[str], evidence: Any) -> None:
        unmet = [item for item in prerequisites if not claims[item]["passed"]]
        claims[name] = {
            "passed": bool(passed and not unmet),
            "unmet_prerequisites": unmet,
            "evidence": evidence,
        }

    claims["localization"] = {
        "passed": bool(capture_ok and _ready(localization)),
        "unmet_prerequisites": [],
        "evidence": localization,
    }
    add(
        "necessity",
        _ready(necessity) and _number(
            necessity.get("mean_margin_drop"), 0.0) > 0.0
        and necessity.get("all_significant_after_bh") is True,
        ["localization"], necessity)
    add(
        "conditional_sufficiency",
        _ready(conditional) and _number(
            conditional.get("test_faithfulness"), 0.0)
        >= config.circuit_faithfulness_min,
        ["necessity"], conditional)
    add(
        "autonomous_sufficiency",
        _ready(autonomous) and _number(
            autonomous.get("test_faithfulness"), 0.0)
        >= config.circuit_faithfulness_min,
        ["conditional_sufficiency"], autonomous)
    add(
        "interchange_causality",
        _ready(interchange)
        and _number(interchange.get("cause_success_fraction"), 0.0)
        >= config.interchange_success_min
        and _number(
            (interchange.get("cause_effect_ci") or {}).get("ci_low"),
            float("-inf")) > 0.0
        and interchange.get("all_variables_causal_after_bh") is True,
        ["autonomous_sufficiency"], interchange)
    add(
        "non_target_isolation",
        _ready(interchange)
        and _number(
            interchange.get("isolation_absolute_effect_mean"), float("inf"))
        <= config.isolation_max_absolute_effect
        and _number(
            (interchange.get("isolation_effect_ci") or {}).get("ci_high"),
            float("inf")) <= config.isolation_max_absolute_effect
        and interchange.get("all_variables_isolated") is True,
        ["interchange_causality"], interchange)
    add(
        "held_out_generalization",
        _ready(held_out) and held_out.get("selection_phase") == "validation"
        and held_out.get("evaluation_phase") == "test"
        and held_out.get("test_used_for_selection") is False,
        ["non_target_isolation"], held_out)
    add(
        "spatial_trajectory_confirmation",
        _ready(spatial) and _ready(trajectory)
        and spatial.get("address_used_for_discovery") is False
        and int(spatial.get("family_count", 0)) > 0
        and interchange.get(
            "all_variables_family_advantage_after_bh") is True
        and _number(trajectory.get("same_minus_cross_mean"), 0.0) > 0.0
        and _number(
            (trajectory.get("effect_ci") or {}).get("ci_low"),
            float("-inf")) > 0.0
        and _number(
            (trajectory.get("paired_null") or {}).get("p_value_two_sided"),
            1.0) <= config.alpha,
        ["held_out_generalization"], {
            "spatial": spatial, "trajectory": trajectory})

    strongest = "descriptive_only"
    for name in (
            "localization", "necessity", "conditional_sufficiency",
            "autonomous_sufficiency", "interchange_causality",
            "non_target_isolation", "held_out_generalization",
            "spatial_trajectory_confirmation"):
        if claims[name]["passed"]:
            strongest = name
    return {
        "status": "ready",
        "claims": claims,
        "strongest_supported_claim": strongest,
        "checkpoint_scope": "checkpoint_specific",
        "cross_checkpoint_claim": False,
        "suppression_interpreted_as": "necessity_only",
    }
