#!/usr/bin/env python3
"""
DAWN Parameter Calculator
=========================
Calculate total parameters and recommend neuron counts for target parameter budgets.

Usage:
    python -m scripts.tools.calc_params --config configs/train_config_v18_2_20M_r16_c4_500M.yaml
    python -m scripts.tools.calc_params --config configs/train_config_v18_2_20M_r16_c4_500M.yaml --target 40M
    python -m scripts.tools.calc_params --d_model 256 --n_layers 8 --rank 16 --target 20M
"""

import argparse
import yaml
import math
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load config from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('model', config)


def calc_embedding_params(vocab_size: int, d_model: int, max_seq_len: int) -> dict:
    """Calculate embedding layer parameters."""
    token_emb = vocab_size * d_model
    pos_emb = max_seq_len * d_model
    return {
        'token_emb': token_emb,
        'pos_emb': pos_emb,
        'total': token_emb + pos_emb,
    }


def calc_ssm_params(d_model: int, state_dim: int) -> dict:
    """Calculate Global SSM parameters."""
    A_log = d_model * state_dim
    W_delta = d_model * d_model  # Linear, no bias
    W_B = d_model * state_dim
    W_C = d_model * state_dim
    ssm_norm = 2 * d_model  # LayerNorm (weight + bias)
    context_proj = d_model * d_model
    importance_proj = d_model * d_model
    context_scale = 1

    total = A_log + W_delta + W_B + W_C + ssm_norm + context_proj + importance_proj + context_scale
    return {
        'A_log': A_log,
        'W_delta': W_delta,
        'W_B': W_B,
        'W_C': W_C,
        'ssm_norm': ssm_norm,
        'context_proj': context_proj,
        'importance_proj': importance_proj,
        'total': total,
    }


def calc_shared_neurons_params(
    d_model: int,
    rank: int,
    knowledge_rank: int,
    n_feature_qk: int,
    n_feature_v: int,
    n_restore_qk: int,
    n_restore_v: int,
    n_feature_know: int,
    n_restore_know: int,
) -> dict:
    """Calculate SharedNeurons parameters."""
    # Attention neurons
    f_neurons = (n_feature_qk + n_feature_v) * d_model * rank
    r_neurons = (n_restore_qk + n_restore_v) * rank * d_model

    # Knowledge neurons
    feature_know = n_feature_know * d_model * knowledge_rank
    restore_know = n_restore_know * knowledge_rank * d_model

    return {
        'f_neurons': f_neurons,
        'r_neurons': r_neurons,
        'feature_know': feature_know,
        'restore_know': restore_know,
        'total': f_neurons + r_neurons + feature_know + restore_know,
    }


def calc_router_params(
    d_model: int,
    d_space: int,
    n_feature_qk: int,
    n_feature_v: int,
    n_restore_qk: int,
    n_restore_v: int,
    n_feature_know: int,
    n_restore_know: int,
    learnable_tau: bool = True,
    version: str = '18.2',
) -> dict:
    """Calculate GlobalRouters parameters."""
    total_neurons = (n_feature_qk + n_feature_v + n_restore_qk + n_restore_v +
                     n_feature_know + n_restore_know)

    # UnifiedNeuronRouter
    proj_all = d_model * (d_space * 6) + (d_space * 6)  # 6 projections + bias
    proj_feature_know = d_model * d_space + d_space
    proj_restore_know = d_model * d_space + d_space
    neuron_emb = total_neurons * d_space

    router_params = {
        'proj_all': proj_all,
        'proj_feature_know': proj_feature_know,
        'proj_restore_know': proj_restore_know,
        'neuron_emb': neuron_emb,
    }

    # v18.2 specific: LayerNorms for each projection
    if version == '18.2':
        norm_params = 6 * (2 * d_space)  # 6 norms (fqk_Q, fqk_K, fv, rqk_Q, rqk_K, rv)
        router_params['norms'] = norm_params

    # Tau projection (if learnable)
    if learnable_tau:
        if version == '18.2':
            tau_proj = d_model * 8 + 8  # 8 pools for v18.2 (Q/K separated)
        else:
            tau_proj = d_model * 6 + 6  # 6 pools for v18.1
        router_params['tau_proj'] = tau_proj

    router_params['total'] = sum(router_params.values())
    return router_params


def calc_layer_params(d_model: int, n_heads: int) -> dict:
    """Calculate per-layer parameters (excluding shared neurons)."""
    # LayerNorms
    norm1 = 2 * d_model
    norm2 = 2 * d_model

    # AttentionCircuit: expand_O
    expand_O = d_model * d_model

    return {
        'norm1': norm1,
        'norm2': norm2,
        'expand_O': expand_O,
        'total': norm1 + norm2 + expand_O,
    }


def calc_total_params(config: dict, verbose: bool = True) -> dict:
    """Calculate total parameters for DAWN model."""
    # Extract config values
    vocab_size = config.get('vocab_size', 30000)
    d_model = config.get('d_model', 256)
    n_layers = config.get('n_layers', 8)
    n_heads = config.get('n_heads', 4)
    rank = config.get('rank', 16)
    knowledge_rank = config.get('knowledge_rank', 16)
    max_seq_len = config.get('max_seq_len', 512)
    state_dim = config.get('state_dim', 64)
    d_space = config.get('d_space', 256)

    # Neuron counts
    n_feature_qk = config.get('n_feature_qk', 64)
    n_feature_v = config.get('n_feature_v', 264)
    n_restore_qk = config.get('n_restore_qk', 64)
    n_restore_v = config.get('n_restore_v', 264)
    n_feature_know = config.get('n_feature_know', 160)
    n_restore_know = config.get('n_restore_know', 160)

    learnable_tau = config.get('learnable_tau', True)
    use_ssm = not (config.get('attention_token_routing', False) and
                   config.get('knowledge_token_routing', False))
    version = str(config.get('model_version', '18.2'))

    # Calculate each component
    emb = calc_embedding_params(vocab_size, d_model, max_seq_len)
    ssm = calc_ssm_params(d_model, state_dim) if use_ssm else {'total': 0}
    neurons = calc_shared_neurons_params(
        d_model, rank, knowledge_rank,
        n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
        n_feature_know, n_restore_know
    )
    router = calc_router_params(
        d_model, d_space,
        n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
        n_feature_know, n_restore_know,
        learnable_tau, version
    )
    layer = calc_layer_params(d_model, n_heads)

    # Final norm + lm_head (tied with token_emb, so only norm counts)
    final_norm = 2 * d_model

    # Total
    total = (emb['total'] + ssm['total'] + neurons['total'] +
             router['total'] + layer['total'] * n_layers + final_norm)

    result = {
        'embedding': emb,
        'ssm': ssm if use_ssm else None,
        'shared_neurons': neurons,
        'router': router,
        'per_layer': layer,
        'final_norm': final_norm,
        'total': total,
        'total_M': total / 1e6,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("DAWN Parameter Breakdown")
        print("=" * 60)
        print(f"\nConfig: d_model={d_model}, n_layers={n_layers}, rank={rank}")
        print(f"        n_feature_qk={n_feature_qk}, n_feature_v={n_feature_v}")
        print(f"        n_restore_qk={n_restore_qk}, n_restore_v={n_restore_v}")
        print(f"        n_feature_know={n_feature_know}, n_restore_know={n_restore_know}")

        print(f"\n{'Component':<25} {'Params':>15} {'%':>8}")
        print("-" * 50)
        print(f"{'Embedding':<25} {emb['total']:>15,} {100*emb['total']/total:>7.1f}%")
        if use_ssm:
            print(f"{'Global SSM':<25} {ssm['total']:>15,} {100*ssm['total']/total:>7.1f}%")
        print(f"{'Shared Neurons':<25} {neurons['total']:>15,} {100*neurons['total']/total:>7.1f}%")
        print(f"  {'- Attention (F+R)':<23} {neurons['f_neurons'] + neurons['r_neurons']:>15,}")
        print(f"  {'- Knowledge (F+R)':<23} {neurons['feature_know'] + neurons['restore_know']:>15,}")
        print(f"{'Router':<25} {router['total']:>15,} {100*router['total']/total:>7.1f}%")
        layer_label = f"Layers (×{n_layers})"
        print(f"{layer_label:<25} {layer['total'] * n_layers:>15,} {100*layer['total']*n_layers/total:>7.1f}%")
        print(f"{'Final Norm':<25} {final_norm:>15,} {100*final_norm/total:>7.1f}%")
        print("-" * 50)
        print(f"{'TOTAL':<25} {total:>15,} {'100.0%':>8}")
        print(f"{'':25} {total/1e6:>15.2f}M")

        # Per-neuron breakdown
        total_neurons = (n_feature_qk + n_feature_v + n_restore_qk + n_restore_v +
                        n_feature_know + n_restore_know)
        print(f"\nTotal neurons: {total_neurons}")
        print(f"Params per neuron (neurons only): {neurons['total'] / total_neurons:,.0f}")
        print(f"Params per neuron (total model): {total / total_neurons:,.0f}")

    return result


def recommend_neurons(
    target_params_M: float,
    d_model: int = 256,
    n_layers: int = 8,
    rank: int = 16,
    knowledge_rank: int = 16,
    vocab_size: int = 30000,
    max_seq_len: int = 512,
    d_space: int = 256,
    state_dim: int = 64,
    use_ssm: bool = False,
    qk_ratio: float = 0.2,  # n_feature_qk as fraction of n_feature_v
    know_ratio: float = 0.6,  # n_feature_know as fraction of n_feature_v
) -> dict:
    """
    Recommend neuron counts for target parameter budget.

    Args:
        target_params_M: Target parameters in millions
        qk_ratio: n_feature_qk = n_feature_v * qk_ratio
        know_ratio: n_feature_know = n_feature_v * know_ratio

    Returns:
        Recommended config
    """
    target = target_params_M * 1e6

    # Fixed overhead (embedding, SSM, layers, final norm)
    emb_params = vocab_size * d_model + max_seq_len * d_model
    ssm_params = (d_model * state_dim + d_model * d_model * 3 +
                  2 * d_model + d_model * d_model * 2 + 1) if use_ssm else 0
    layer_params = (4 * d_model + d_model * d_model) * n_layers  # norms + expand_O
    final_params = 2 * d_model

    fixed_overhead = emb_params + ssm_params + layer_params + final_params

    # Available for neurons + router
    available = target - fixed_overhead

    # Solve for n_feature_v
    # Let x = n_feature_v
    # n_feature_qk = n_restore_qk = x * qk_ratio
    # n_restore_v = x
    # n_feature_know = n_restore_know = x * know_ratio
    #
    # Neuron params:
    # f_neurons = (qk + v) * d_model * rank = x * (qk_ratio + 1) * d_model * rank
    # r_neurons = (qk + v) * rank * d_model = same
    # feature_know = know * d_model * knowledge_rank = x * know_ratio * d_model * knowledge_rank
    # restore_know = same
    #
    # Router params (approximately):
    # proj_all ≈ d_model * d_space * 6
    # proj_know ≈ d_model * d_space * 2
    # neuron_emb = total_neurons * d_space = x * (2 * qk_ratio + 2 + 2 * know_ratio) * d_space
    # tau_proj ≈ d_model * 8
    # norms ≈ 6 * 2 * d_space

    # Simplified: params_per_x = coefficient for x in total formula
    attn_coef = 2 * (1 + qk_ratio) * d_model * rank  # f + r neurons for attention
    know_coef = 2 * know_ratio * d_model * knowledge_rank  # f + r neurons for knowledge
    emb_coef = (2 * qk_ratio + 2 + 2 * know_ratio) * d_space  # neuron embeddings

    params_per_x = attn_coef + know_coef + emb_coef

    # Router fixed overhead
    router_fixed = (d_model * d_space * 6 + d_space * 6 +  # proj_all
                    d_model * d_space * 2 + d_space * 2 +  # proj_know
                    d_model * 8 + 8 +  # tau_proj
                    6 * 2 * d_space)  # norms

    available_for_neurons = available - router_fixed
    n_feature_v = int(available_for_neurons / params_per_x)

    # Calculate all neuron counts
    n_feature_qk = max(16, int(n_feature_v * qk_ratio))
    n_restore_qk = n_feature_qk
    n_restore_v = n_feature_v
    n_feature_know = max(16, int(n_feature_v * know_ratio))
    n_restore_know = n_feature_know

    # Round to nice numbers (multiples of 8)
    n_feature_v = (n_feature_v // 8) * 8
    n_restore_v = n_feature_v
    n_feature_qk = (n_feature_qk // 8) * 8
    n_restore_qk = n_feature_qk
    n_feature_know = (n_feature_know // 8) * 8
    n_restore_know = n_feature_know

    recommended = {
        'd_model': d_model,
        'n_layers': n_layers,
        'rank': rank,
        'knowledge_rank': knowledge_rank,
        'n_feature_qk': n_feature_qk,
        'n_feature_v': n_feature_v,
        'n_restore_qk': n_restore_qk,
        'n_restore_v': n_restore_v,
        'n_feature_know': n_feature_know,
        'n_restore_know': n_restore_know,
        'd_space': d_space,
        'max_seq_len': max_seq_len,
        'vocab_size': vocab_size,
    }

    # Verify
    actual = calc_total_params(recommended, verbose=False)

    print("\n" + "=" * 60)
    print(f"Recommended Config for {target_params_M:.1f}M Parameters")
    print("=" * 60)
    print(f"\nNeuron counts:")
    print(f"  n_feature_qk:   {n_feature_qk:4d}  (Q/K attention)")
    print(f"  n_feature_v:    {n_feature_v:4d}  (V attention)")
    print(f"  n_restore_qk:   {n_restore_qk:4d}")
    print(f"  n_restore_v:    {n_restore_v:4d}")
    print(f"  n_feature_know: {n_feature_know:4d}  (knowledge)")
    print(f"  n_restore_know: {n_restore_know:4d}")
    print(f"\nTotal neurons: {n_feature_qk + n_feature_v + n_restore_qk + n_restore_v + n_feature_know + n_restore_know}")
    print(f"Actual params: {actual['total_M']:.2f}M (target: {target_params_M:.1f}M)")
    print(f"Difference: {(actual['total_M'] - target_params_M):+.2f}M")

    return recommended


def main():
    parser = argparse.ArgumentParser(
        description='DAWN Parameter Calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate params from config
  python -m scripts.tools.calc_params --config configs/train_config_v18_2_20M_r16_c4_500M.yaml

  # Recommend neurons for target params
  python -m scripts.tools.calc_params --config configs/train_config_v18_2_20M_r16_c4_500M.yaml --target 40

  # Manual config
  python -m scripts.tools.calc_params --d_model 256 --n_layers 8 --rank 16 --target 20
        """
    )
    parser.add_argument('--config', type=str, help='Path to config yaml file')
    parser.add_argument('--target', type=float, help='Target parameters in millions (e.g., 40 for 40M)')

    # Manual config options
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--knowledge_rank', type=int, default=16)
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--d_space', type=int, default=256)

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        print(f"Loaded config: {args.config}")
    else:
        config = {
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'rank': args.rank,
            'knowledge_rank': args.knowledge_rank,
            'vocab_size': args.vocab_size,
            'max_seq_len': args.max_seq_len,
            'd_space': args.d_space,
            'n_feature_qk': 64,
            'n_feature_v': 264,
            'n_restore_qk': 64,
            'n_restore_v': 264,
            'n_feature_know': 160,
            'n_restore_know': 160,
            'learnable_tau': True,
            'model_version': '18.2',
        }

    # Calculate current params
    result = calc_total_params(config)

    # Recommend neurons if target specified
    if args.target:
        recommend_neurons(
            target_params_M=args.target,
            d_model=config.get('d_model', args.d_model),
            n_layers=config.get('n_layers', args.n_layers),
            rank=config.get('rank', args.rank),
            knowledge_rank=config.get('knowledge_rank', args.knowledge_rank),
            vocab_size=config.get('vocab_size', args.vocab_size),
            max_seq_len=config.get('max_seq_len', args.max_seq_len),
            d_space=config.get('d_space', args.d_space),
            use_ssm=not (config.get('attention_token_routing', False) and
                         config.get('knowledge_token_routing', False)),
        )


if __name__ == '__main__':
    main()
