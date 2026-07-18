"""Pre-registered scientific contract for DAWN operator interpretability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


PROTOCOL_ID = "dawn_operator_interpretability"
PROTOCOL_SCHEMA_VERSION = 5
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
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

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
