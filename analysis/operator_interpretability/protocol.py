"""Pre-registered scientific contract for DAWN operator interpretability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


PROTOCOL_ID = "dawn_operator_interpretability"
PROTOCOL_SCHEMA_VERSION = 10
ANALYSIS_ENGINE = "train_analysis_pool"
SUPPORTED_MODEL_VERSIONS = (
    "spatial-r1-v4.1.7.1",
    "spatial-r1-v4.1.7.2",
)
PHASES = ("discovery", "validation", "test")
UNIT_LEVELS = (
    "operator_site",
    "address_neighborhood",
    "functional_family",
    "multilayer_trajectory",
    "paired_operator_trajectory",
)
RETENTION_MODES = (
    "conditional_execution_sufficiency",
    "autonomous_subcircuit_sufficiency",
)
CLAIM_LADDER = (
    "localization",
    "necessity",
    "conditional_sufficiency",
    "autonomous_sufficiency",
    "interchange_causality",
    "non_target_isolation",
    "held_out_generalization",
    "spatial_trajectory_confirmation",
)
NATIVE_PROGRAM_CLAIM_LADDER = (
    "descriptive_decision_program",
    "compact_dynamic_sufficiency",
    "specific_causal_decision_program",
    "counterfactual_operator_selection_transfer",
    "counterfactual_contribution_transplant",
)
PAIRED_TRAJECTORY_ALGORITHM_VERSION = "paired_s2_operator_trajectory_v1"

# MIB circuit-track fractions, including the mandatory full-model endpoint.
CIRCUIT_FRACTIONS = (
    0.001, 0.002, 0.005, 0.01, 0.02,
    0.05, 0.10, 0.20, 0.50, 1.00,
)


@dataclass(frozen=True)
class ProtocolConfig:
    """Settings that must be fixed before validation or test is inspected."""

    seed: int = 4172
    max_examples_per_phase: int = 128
    ravel_max_examples_per_phase: int = 512
    capture_threshold: float = 0.95
    capture_topk_qk: int = 512
    capture_topk_v: int = 2048
    capture_topk_rst: int = 4096
    capture_max_topk_qk: int = 2048
    capture_max_topk_v: int = 8192
    capture_max_topk_rst: int = 8192
    space_max_operators: int = 2048
    bootstrap_samples: int = 2000
    permutation_samples: int = 2000
    alpha: float = 0.05
    minimum_known_correct: int = 32
    minimum_pairs_per_causal_variable: int = 8
    family_neighbor_k: int = 16
    family_similarity_quantile: float = 0.99
    rank_stability_min: float = 0.80
    circuit_faithfulness_min: float = 0.80
    interchange_success_min: float = 0.60
    isolation_max_absolute_effect: float = 0.10
    program_mass_candidates: tuple[float, ...] = (
        0.50, 0.70, 0.80, 0.90, 0.95)
    program_replay_faithfulness_min: float = 0.80
    program_replay_agreement_min: float = 0.90
    program_compact_fraction_max: float = 0.20
    program_id_transfer_flip_min: float = 0.50
    program_transplant_flip_min: float = 0.50
    program_position_scope: str = "answer_position_only"
    program_routes: tuple[str, ...] = ("q", "k", "v", "rst")
    program_denominator_policy: str = "full_production_denominator"
    program_mismatch_matching: str = (
        "same_template_answer_disjoint_nearest_site_count_seeded")
    program_random_sampling: str = (
        "selected_complement_first_without_replacement")
    program_faithfulness_endpoint: str = (
        "paired_counterfactual_source_prompt_base_answer_margin")
    trajectory_deep_examples: int = 8
    trajectory_discovery_examples: int = 128
    trajectory_validation_examples: int = 128
    trajectory_seed: int = 4172
    trajectory_test_enabled: bool = False
    trajectory_capture_topk_qk: int = 1024
    trajectory_capture_topk_v: int = 2048
    trajectory_capture_topk_rst: int = 4096
    trajectory_max_candidate_sites: int = 32
    trajectory_max_candidates_per_route: int = 8
    trajectory_intervention_batch_size: int = 4
    trajectory_max_patch_sites_per_variant: int = 8
    trajectory_max_operator_followup_sites: int = 8
    trajectory_individual_operator_followup_limit: int = 0
    trajectory_max_path_sites: int = 8
    trajectory_path_prefix_batch_size: int = 4
    trajectory_state_identity_atol: float = 1.0e-5
    trajectory_replay_atol: float = 5.0e-4
    trajectory_replay_rtol: float = 5.0e-4
    trajectory_divergence_epsilon: float = 1.0e-6

    def validate(self) -> "ProtocolConfig":
        if self.max_examples_per_phase <= 0:
            raise ValueError("max_examples_per_phase must be positive")
        if self.ravel_max_examples_per_phase <= 0:
            raise ValueError(
                "ravel_max_examples_per_phase must be positive")
        if not 0.0 < self.capture_threshold <= 1.0:
            raise ValueError("capture_threshold must be in (0, 1]")
        widths = (
            self.capture_topk_qk, self.capture_topk_v,
            self.capture_topk_rst, self.capture_max_topk_qk,
            self.capture_max_topk_v, self.capture_max_topk_rst,
            self.space_max_operators,
        )
        if any(int(value) <= 0 for value in widths):
            raise ValueError("capture/space widths must all be positive")
        for initial, maximum, route in (
                (self.capture_topk_qk, self.capture_max_topk_qk, "qk"),
                (self.capture_topk_v, self.capture_max_topk_v, "v"),
                (self.capture_topk_rst, self.capture_max_topk_rst, "rst")):
            if initial > maximum:
                raise ValueError(
                    f"capture initial width exceeds maximum for {route}")
        if self.bootstrap_samples < 100 or self.permutation_samples < 100:
            raise ValueError("scientific resampling counts must each be >= 100")
        if not 0.0 < self.alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5)")
        if self.minimum_known_correct <= 0:
            raise ValueError("minimum_known_correct must be positive")
        if self.minimum_pairs_per_causal_variable < 2:
            raise ValueError(
                "minimum_pairs_per_causal_variable must be >= 2")
        if self.family_neighbor_k <= 0:
            raise ValueError("family_neighbor_k must be positive")
        if not 0.0 < self.family_similarity_quantile < 1.0:
            raise ValueError("family_similarity_quantile must be in (0, 1)")
        masses = tuple(float(value) for value in self.program_mass_candidates)
        if (not masses or tuple(sorted(set(masses))) != masses
                or any(not 0.0 < value <= 1.0 for value in masses)):
            raise ValueError(
                "program_mass_candidates must be unique, increasing values "
                "in (0, 1]")
        for name, value in (
                ("program_replay_faithfulness_min",
                 self.program_replay_faithfulness_min),
                ("program_replay_agreement_min",
                 self.program_replay_agreement_min),
                ("program_compact_fraction_max",
                 self.program_compact_fraction_max),
                ("program_id_transfer_flip_min",
                 self.program_id_transfer_flip_min),
                ("program_transplant_flip_min",
                 self.program_transplant_flip_min)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.program_position_scope != "answer_position_only":
            raise ValueError(
                "program_position_scope must be 'answer_position_only'")
        if tuple(self.program_routes) != ("q", "k", "v", "rst"):
            raise ValueError("program_routes must be exactly q,k,v,rst")
        if self.program_denominator_policy != "full_production_denominator":
            raise ValueError(
                "program_denominator_policy must preserve the full "
                "production denominator")
        if self.program_mismatch_matching != (
                "same_template_answer_disjoint_nearest_site_count_seeded"):
            raise ValueError("unsupported program_mismatch_matching")
        if self.program_random_sampling != (
                "selected_complement_first_without_replacement"):
            raise ValueError("unsupported program_random_sampling")
        if self.program_faithfulness_endpoint != (
                "paired_counterfactual_source_prompt_base_answer_margin"):
            raise ValueError("unsupported program_faithfulness_endpoint")
        trajectory_positive = {
            "trajectory_deep_examples": self.trajectory_deep_examples,
            "trajectory_discovery_examples": self.trajectory_discovery_examples,
            "trajectory_validation_examples": self.trajectory_validation_examples,
            "trajectory_capture_topk_qk": self.trajectory_capture_topk_qk,
            "trajectory_capture_topk_v": self.trajectory_capture_topk_v,
            "trajectory_capture_topk_rst": self.trajectory_capture_topk_rst,
            "trajectory_max_candidate_sites": self.trajectory_max_candidate_sites,
            "trajectory_max_candidates_per_route": (
                self.trajectory_max_candidates_per_route),
            "trajectory_intervention_batch_size": (
                self.trajectory_intervention_batch_size),
            "trajectory_max_patch_sites_per_variant": (
                self.trajectory_max_patch_sites_per_variant),
            "trajectory_max_operator_followup_sites": (
                self.trajectory_max_operator_followup_sites),
            "trajectory_max_path_sites": self.trajectory_max_path_sites,
            "trajectory_path_prefix_batch_size": (
                self.trajectory_path_prefix_batch_size),
        }
        invalid = [
            name for name, value in trajectory_positive.items()
            if isinstance(value, bool) or int(value) <= 0
        ]
        if invalid:
            raise ValueError(
                "trajectory positive integer settings are invalid: "
                + ",".join(invalid))
        if self.trajectory_discovery_examples < self.trajectory_deep_examples:
            raise ValueError(
                "trajectory discovery examples must cover the deep cohort")
        if self.trajectory_max_candidate_sites < (
                self.trajectory_max_candidates_per_route):
            raise ValueError(
                "trajectory global candidate cap is smaller than route cap")
        if self.trajectory_max_path_sites > (
                self.trajectory_max_patch_sites_per_variant):
            raise ValueError(
                "trajectory path length exceeds fixed patch-slot capacity")
        if self.trajectory_max_operator_followup_sites > (
                self.trajectory_max_candidate_sites):
            raise ValueError(
                "trajectory operator follow-up cap exceeds site cap")
        if self.trajectory_individual_operator_followup_limit != 0:
            raise ValueError(
                "paired trajectory v1 forbids individual operator leave-one-out")
        if self.trajectory_test_enabled:
            raise ValueError("paired trajectory v1 must not evaluate test")
        for name, value in (
                ("trajectory_state_identity_atol",
                 self.trajectory_state_identity_atol),
                ("trajectory_replay_atol", self.trajectory_replay_atol),
                ("trajectory_replay_rtol", self.trajectory_replay_rtol),
                ("trajectory_divergence_epsilon",
                 self.trajectory_divergence_epsilon)):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["program_mass_candidates"] = list(
            self.program_mass_candidates)
        value["program_routes"] = list(self.program_routes)
        return value

    def max_examples_for(self, benchmark_id: str) -> int:
        self.validate()
        if str(benchmark_id) == "ravel":
            return int(self.ravel_max_examples_per_phase)
        return int(self.max_examples_per_phase)


def validate_model_version(model_version: str) -> str:
    value = str(model_version)
    if value not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(
            f"operator interpretability does not support {value!r}; "
            f"supported={','.join(SUPPORTED_MODEL_VERSIONS)}")
    return value


def protocol_record(config: ProtocolConfig, *, model_version: str,
                    benchmark_manifest_hash: str, checkpoint_identity: str,
                    model_config_hash: str) -> dict[str, Any]:
    if not benchmark_manifest_hash or not checkpoint_identity or not model_config_hash:
        raise ValueError(
            "benchmark, checkpoint, and model-config identities are required")
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "analysis_engine": ANALYSIS_ENGINE,
        "model_version": validate_model_version(model_version),
        "benchmark_manifest_hash": str(benchmark_manifest_hash),
        "checkpoint_identity": str(checkpoint_identity),
        "model_config_hash": str(model_config_hash),
        "phases": list(PHASES),
        "unit_levels": list(UNIT_LEVELS),
        "retention_modes": list(RETENTION_MODES),
        "claim_ladder": list(CLAIM_LADDER),
        "native_program_claim_ladder": list(NATIVE_PROGRAM_CLAIM_LADDER),
        "paired_trajectory_algorithm_version": (
            PAIRED_TRAJECTORY_ALGORITHM_VERSION),
        "circuit_fractions": list(CIRCUIT_FRACTIONS),
        "config": config.to_dict(),
        "test_selection_forbidden": True,
        "address_excluded_from_functional_family_discovery": True,
    }


def require_protocol_match(record: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Fail when a protocol-bound artifact is read under another protocol."""
    keys = sorted(set(record) | set(expected))
    differences = [key for key in keys if record.get(key) != expected.get(key)]
    if differences:
        raise ValueError(
            "analysis artifact protocol mismatch: " + ",".join(differences))
