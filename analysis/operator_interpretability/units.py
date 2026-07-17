"""Canonical operator-site, family, trajectory, and circuit representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from analysis.operator_interpretability.benchmark_schema import canonical_hash


ROUTES = ("q", "k", "v", "rst")
ROUTE_INDEX = {name: index for index, name in enumerate(ROUTES)}


@dataclass(frozen=True, order=True)
class OperatorSite:
    layer: int
    route: str
    operator_id: int

    def validate(self, shape: "OperatorSpaceShape") -> "OperatorSite":
        if not 0 <= self.layer < shape.n_layers:
            raise ValueError(f"operator-site layer is out of range: {self}")
        if self.route not in ROUTES:
            raise ValueError(f"unknown operator-site route: {self.route!r}")
        if not 0 <= self.operator_id < shape.pool_size(self.route):
            raise ValueError(f"operator-site id is out of range: {self}")
        return self


@dataclass(frozen=True)
class OperatorSpaceShape:
    n_layers: int
    n_qk: int
    n_v: int
    n_rst: int

    def pool_size(self, route: str) -> int:
        return {
            "q": self.n_qk, "k": self.n_qk,
            "v": self.n_v, "rst": self.n_rst,
        }[route]

    @property
    def total_sites(self) -> int:
        return self.n_layers * (2 * self.n_qk + self.n_v + self.n_rst)

    @classmethod
    def from_model_cfg(cls, cfg: Mapping[str, Any]) -> "OperatorSpaceShape":
        value = cls(
            n_layers=int(cfg["n_layers"]),
            n_qk=int(cfg["n_qk"]),
            n_v=int(cfg["n_v"]),
            n_rst=int(cfg.get("n_rst", cfg.get("n_know"))),
        )
        if min(value.n_layers, value.n_qk, value.n_v, value.n_rst) <= 0:
            raise ValueError("operator-space dimensions must all be positive")
        return value


@dataclass(frozen=True)
class RankedSite:
    site: OperatorSite
    importance: float
    discovery_count: int
    captured_mass_mean: float


@dataclass(frozen=True)
class OperatorCircuit:
    sites: tuple[OperatorSite, ...]
    discovery_benchmark: str
    discovery_phase: str = "discovery"
    implicit_zero_site_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, shape: OperatorSpaceShape) -> "OperatorCircuit":
        if self.discovery_phase != "discovery":
            raise ValueError("circuits may only be discovered on the discovery phase")
        if len(set(self.sites)) != len(self.sites):
            raise ValueError("operator circuit contains duplicate sites")
        if self.implicit_zero_site_count < 0:
            raise ValueError("implicit_zero_site_count cannot be negative")
        if len(self.sites) + self.implicit_zero_site_count > shape.total_sites:
            raise ValueError("operator circuit is larger than the operator space")
        for site in self.sites:
            site.validate(shape)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "sites": [asdict(site) for site in self.sites],
            "discovery_benchmark": self.discovery_benchmark,
            "discovery_phase": self.discovery_phase,
            "implicit_zero_site_count": self.implicit_zero_site_count,
            "implicit_zero_site_order": "layer_route_operator_lexicographic",
            "metadata": dict(self.metadata),
        }

    @property
    def circuit_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def site_count(self) -> int:
        return len(self.sites) + int(self.implicit_zero_site_count)

    def _all_selected_sites(self, shape: OperatorSpaceShape):
        explicit = set(self.sites)
        yield from self.sites
        remaining = int(self.implicit_zero_site_count)
        if remaining <= 0:
            return
        for layer in range(shape.n_layers):
            for route in ROUTES:
                for operator_id in range(shape.pool_size(route)):
                    site = OperatorSite(layer, route, operator_id)
                    if site in explicit:
                        continue
                    yield site
                    remaining -= 1
                    if remaining == 0:
                        return
        raise RuntimeError("implicit circuit site expansion exhausted the space")

    def dense_masks(self, shape: OperatorSpaceShape) -> dict[str, np.ndarray]:
        self.validate(shape)
        qk = np.zeros((shape.n_layers, 2, shape.n_qk), dtype=np.bool_)
        v = np.zeros((shape.n_layers, shape.n_v), dtype=np.bool_)
        rst = np.zeros((shape.n_layers, shape.n_rst), dtype=np.bool_)
        for site in self._all_selected_sites(shape):
            if site.route in ("q", "k"):
                qk[site.layer, ROUTE_INDEX[site.route], site.operator_id] = True
            elif site.route == "v":
                v[site.layer, site.operator_id] = True
            else:
                rst[site.layer, site.operator_id] = True
        return {"qk": qk, "v": v, "rst": rst}


def nested_circuits(ranked_sites: Sequence[RankedSite], *,
                    shape: OperatorSpaceShape, benchmark_id: str,
                    fractions: Sequence[float]) -> list[tuple[float, OperatorCircuit]]:
    ordered = sorted(
        ranked_sites,
        key=lambda row: (-abs(float(row.importance)), row.site),
    )
    if not ordered:
        raise ValueError("cannot construct a circuit without ranked sites")
    if len({row.site for row in ordered}) != len(ordered):
        raise ValueError("ranked operator sites contain duplicates")
    circuits: list[tuple[float, OperatorCircuit]] = []
    for fraction in fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"invalid circuit fraction: {fraction}")
        requested = int(float(fraction) * shape.total_sites)
        selected = tuple(row.site for row in ordered[:requested])
        implicit_count = max(0, requested - len(selected))
        circuits.append((
            float(fraction),
            OperatorCircuit(
                sites=selected,
                discovery_benchmark=benchmark_id,
                implicit_zero_site_count=implicit_count,
                metadata={
                    "selection": "absolute_discovery_importance",
                    "requested_sites": requested,
                    "available_ranked_sites": len(ordered),
                    "total_operator_sites": shape.total_sites,
                    "unobserved_zero_importance_sites_added": implicit_count,
                    "candidate_funnel_limited": implicit_count > 0,
                },
            ).validate(shape),
        ))
    return circuits
