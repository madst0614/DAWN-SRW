# DAWN-SRW: Atomic Unit Composition for Language Modeling

[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.19615584)
[![Predecessor](https://img.shields.io/badge/DAWN-Predecessor-green)](https://doi.org/10.5281/zenodo.19599530)

DAWN-SRW extends the [DAWN framework](https://github.com/madst0614/DAWN) to rank-1 atomic units. Every projection in a Transformer (Query, Key, Value, feedforward) is dynamically composed from a shared pool of Select-Read-Write units — the minimal possible computational atom.

## Key Results

| Model | Params | Val Loss ↓ | PPL ↓ |
|-------|--------|-----------|-------|
| Dense Transformer | 40.4M | 3.720 | 41.3 |
| **DAWN-SRW** | 40.2M | **3.656** | **38.7** |

*Trained on C4 (5B tokens). DAWN-SRW surpasses the dense baseline after a reorganization phase (crossover at 33% of training).*

## Key Findings

- **Directional Redirection**: Knowledge-pool units read one direction and write in the opposite (cos(r,w) = −0.32). Stronger redirectors are selected more frequently (r = −0.62).
- **Perfect Additivity**: Unit contributions sum exactly (cos = 1.0 before normalization). No nonlinear interactions between units.
- **171° Drift**: The residual stream rotates 171° across 12 layers, with prediction emerging at layer 4 (top-1 35.6% at final layer).
- **Emergent Specialization**: 96.4% of QK units specialize for Q or K. Individual units show up to 21.6× POS selectivity.
- **Three-Phase Dynamics**: Fast capture → reorganization → specialization. Phase 2 cost shrinks with scale (26% at 40M → 17% at 400M).

## Architecture

Each atomic unit has three vectors:
- **Embedding (e)**: signature — *when* this unit is selected
- **Read (r)**: extracts a scalar from input — *what* it senses
- **Write (w)**: contributes a direction to residual — *where* it pushes

```
out = Σ gate_i · (x · r_i) · w_i / max(Σ gate_i, 1.0) × scale
```

Three global pools shared across all layers:
- **QK pool** (1,580 units): Q and K share pool, route independently
- **V pool** (2,600 units): Value projection
- **Know pool** (25,200 units): Replaces FFN

## Relation to DAWN

| | DAWN | DAWN-SRW |
|---|---|---|
| Unit rank | r = 64 | **r = 1** |
| Operation | Feature-Restore pair (d→r→d) | Read-Write pair (rank-1) |
| Routing | softmax → top-k → renorm | threshold → GELU gate → sum-norm |
| Signature | Learnable embedding | Learnable embedding *(same)* |
| Abstract structure | identity = signature + operation | identity = signature + operation *(same)* |

## Installation

```bash
git clone https://github.com/madst0614/dawn-srw
cd dawn-srw
pip install -r requirements.txt
```

## Checkpoints

| Model | Params | Val Loss | Status |
|-------|--------|----------|--------|
| DAWN-SRW 40M | 40.2M | 3.656 | Available |
| DAWN-SRW 400M | — | — | Training in progress |

## Citation

```bibtex
@misc{choi2026dawnsrw,
  title={DAWN-SRW: Atomic Unit Composition for Language Modeling},
  author={Choi, Seungho},
  year={2026},
  doi={10.5281/zenodo.19615584},
  url={https://github.com/madst0614/dawn-srw}
}
```

### Predecessor

```bibtex
@misc{choi2025dawn,
  title={DAWN: Dynamic Subspace Composition for Language Modeling},
  author={Choi, Seungho},
  year={2025},
  doi={10.5281/zenodo.19599530},
  url={https://doi.org/10.5281/zenodo.19599530}
}
```

## License

MIT