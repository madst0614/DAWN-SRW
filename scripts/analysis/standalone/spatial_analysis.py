#!/usr/bin/env python3
"""
DAWN-Spatial v3 Analysis Script
================================
Standalone analysis for DAWN-Spatial SRW checkpoints (JAX/Flax).

Features:
  D1: Model Info — params, architecture, FLOPs
  D2: Validation — loss, perplexity, accuracy
  D3: Neuron Health — activation rates, dead neurons, gate distribution
  D4: Generation — autoregressive text generation
  D5: Weight Analysis — embedding similarity, effective rank

Usage:
    python scripts/analysis/standalone/spatial_analysis.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/.../run_XXX \
        --config configs/train_config_spatial_r1_v3_40M_c4_5B.yaml \
        --output results/spatial_analysis

    # Quick generation only:
    python scripts/analysis/standalone/spatial_analysis.py \
        --checkpoint gs://...  --config configs/... \
        --only generate --prompt "The meaning of life is"
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time
import math
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization as serialization
import yaml


# ============================================================
# GCS / file helpers (copied from train_jax for standalone use)
# ============================================================

def _is_gcs(path):
    return str(path).startswith("gs://")


def _open_file(path, mode="rb"):
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            return fs.open(path_str, mode)
        except ImportError:
            pass
        try:
            import tensorflow as tf
            return tf.io.gfile.GFile(path_str, mode)
        except ImportError:
            raise ImportError("GCS requires 'gcsfs' or 'tensorflow'.")
    else:
        p = Path(path_str)
        if "w" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, mode)


def _file_exists(path):
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            return gcsfs.GCSFileSystem().exists(path_str)
        except ImportError:
            import tensorflow as tf
            return tf.io.gfile.exists(path_str)
    return Path(path_str).exists()


def _list_dir(path):
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            return [f"gs://{f}" for f in fs.ls(path_str)]
        except ImportError:
            import tensorflow as tf
            return tf.io.gfile.listdir(path_str)
    return [str(p) for p in Path(path_str).iterdir()]


# ============================================================
# Model building
# ============================================================

def build_model(cfg):
    """Build DAWN-Spatial v3 model from config dict."""
    from models.dawn_spatial_v3 import DAWN as DAWN_SpatialV3
    mcfg = cfg['model']
    return DAWN_SpatialV3(
        vocab_size=mcfg.get('vocab_size', 30522),
        d_model=mcfg.get('d_model', 384),
        n_layers=mcfg.get('n_layers', 12),
        n_heads=mcfg.get('n_heads', 6),
        max_seq_len=mcfg.get('max_seq_len', 512),
        d_bottleneck=mcfg.get('d_bottleneck', 128),
        n_qk=mcfg.get('n_qk', 1580),
        n_v=mcfg.get('n_v', 2600),
        n_know=mcfg.get('n_know', 25200),
        max_k_qk=mcfg.get('max_k_qk', 158),
        max_k_v=mcfg.get('max_k_v', 260),
        max_k_know=mcfg.get('max_k_know', 1810),
        dropout_rate=mcfg.get('dropout', 0.1),
        router_dropout=mcfg.get('router_dropout', 0.1),
        gradient_checkpointing=False,
        n_chunks_know=cfg.get('training', {}).get('n_chunks_know', 1),
        n_chunks_qk=cfg.get('training', {}).get('n_chunks_qk', 1),
        n_chunks_v=cfg.get('training', {}).get('n_chunks_v', 1),
    )


def load_checkpoint_params(ckpt_path, model, cfg):
    """Load checkpoint and return params."""
    # Find latest checkpoint in directory
    if not ckpt_path.endswith('.msgpack') and not ckpt_path.endswith('.ckpt'):
        # It's a directory — find latest checkpoint file
        files = _list_dir(ckpt_path)
        flax_files = sorted([f for f in files if f.endswith('.flax')])
        # Prefer best_model.flax, else latest checkpoint_step*.flax
        best = [f for f in flax_files if 'best_model' in f]
        if best:
            ckpt_path = best[0]
        elif flax_files:
            ckpt_path = flax_files[-1]
        else:
            raise FileNotFoundError(
                f"No .flax checkpoints in {ckpt_path}\n"
                f"  Files: {[os.path.basename(f) for f in files]}")
        print(f"  Selected: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    # Init model to get param structure
    rng = jax.random.PRNGKey(0)
    max_seq = cfg['model'].get('max_seq_len', 512)
    dummy = jnp.ones((1, max_seq), dtype=jnp.int32)
    variables = model.init(
        {'params': rng, 'dropout': rng}, dummy, deterministic=True)
    target_params = variables['params']

    # Load checkpoint — use flax msgpack_restore for proper numpy array handling,
    # then restore only params (skip opt_state to avoid shape mismatch)
    with _open_file(ckpt_path, 'rb') as f:
        bytes_data = f.read()

    raw = serialization.msgpack_restore(bytes_data)
    raw_params = raw['params']

    # Restore params into model structure (validates shapes)
    params = serialization.from_state_dict(target_params, raw_params)

    step = int(raw.get('step', 0))
    epoch = int(raw.get('epoch', 0))
    print(f"  Step: {step}, Epoch: {epoch}")

    ckpt = {'params': params, 'step': step, 'epoch': epoch}
    return ckpt['params'], ckpt


def count_params(params):
    """Count total parameters."""
    return sum(x.size for x in jax.tree.leaves(params) if hasattr(x, 'size'))


# ============================================================
# D1: Model Info
# ============================================================

def analyze_model_info(model, params, cfg):
    print("\n" + "="*60)
    print("D1: Model Info")
    print("="*60)

    n_params = count_params(params)
    mcfg = cfg['model']

    print(f"  Version:      {mcfg.get('model_version', 'unknown')}")
    print(f"  Parameters:   {n_params:,}")
    print(f"  d_model:      {mcfg.get('d_model')}")
    print(f"  n_layers:     {mcfg.get('n_layers')}")
    print(f"  n_heads:      {mcfg.get('n_heads')}")
    print(f"  d_bottleneck: {mcfg.get('d_bottleneck')}")
    print(f"  n_qk:         {mcfg.get('n_qk')}")
    print(f"  n_v:          {mcfg.get('n_v')}")
    print(f"  n_know:       {mcfg.get('n_know')}")
    print(f"  max_seq_len:  {mcfg.get('max_seq_len')}")

    # Parameter breakdown
    pool = {k: v for k, v in
            jax.tree.map_with_path(
                lambda path, x: ('/'.join(str(p) for p in path), x.size),
                params).items()} if False else {}

    # Simpler breakdown
    pool_params = sum(x.size for x in jax.tree.leaves(params.get('neuron_pool', {})))
    router_params = sum(x.size for x in jax.tree.leaves(params.get('router', {})))
    other = n_params - pool_params - router_params

    print(f"\n  Param breakdown:")
    print(f"    NeuronPool:  {pool_params:>12,} ({pool_params/n_params*100:.1f}%)")
    print(f"    Router:      {router_params:>12,} ({router_params/n_params*100:.1f}%)")
    print(f"    Other:       {other:>12,} ({other/n_params*100:.1f}%)")

    return {'n_params': n_params, 'version': mcfg.get('model_version')}


# ============================================================
# D2: Validation
# ============================================================

def analyze_validation(model, params, cfg, val_data_path, max_batches=200, batch_size=32):
    print("\n" + "="*60)
    print("D2: Validation Performance")
    print("="*60)

    max_seq = cfg['model'].get('max_seq_len', 512)

    # Load validation data
    print(f"  Loading: {val_data_path}")
    if _is_gcs(val_data_path):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(val_data_path, 'rb') as f:
            raw = f.read()
        tokens = np.frombuffer(raw, dtype=np.uint16).copy()
    else:
        tokens = np.memmap(val_data_path, dtype=np.uint16, mode='r')

    n_tokens = len(tokens)
    n_seqs = n_tokens // max_seq
    tokens = tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    print(f"  Tokens: {n_tokens:,}, Sequences: {n_seqs:,}")

    # eval_step returns weighted loss (loss*valid) to avoid per-batch sync
    @jax.jit
    def eval_step(params, input_ids):
        attention_mask = jnp.ones_like(input_ids)
        result = model.apply(
            {'params': params}, input_ids, labels=input_ids,
            attention_mask=attention_mask, deterministic=True,
            rngs={'dropout': jax.random.PRNGKey(0)})
        # Return weighted values — accumulate on device, sync once at end
        return (result['loss'] * result['valid_count'],
                result['correct'], result['valid_count'])

    n_batches = min(max_batches, n_seqs // batch_size)
    # Pre-load all batches to device at once
    total_seqs = n_batches * batch_size
    all_tokens = jnp.array(tokens[:total_seqs], dtype=jnp.int32)

    print(f"  Running {n_batches} batches (bs={batch_size}, "
          f"{total_seqs*max_seq:,} tokens on device)...")

    # Accumulate on device — no per-batch host sync
    acc_loss = jnp.float32(0.0)
    acc_correct = jnp.float32(0.0)
    acc_valid = jnp.float32(0.0)

    t0 = time.time()
    for i in range(n_batches):
        batch = jax.lax.dynamic_slice(
            all_tokens, (i * batch_size, 0), (batch_size, max_seq))
        wloss, correct, valid = eval_step(params, batch)
        acc_loss = acc_loss + wloss
        acc_correct = acc_correct + correct
        acc_valid = acc_valid + valid

        if (i+1) % 50 == 0:
            # Periodic sync for progress (only every 50 batches)
            avg = float(acc_loss) / float(acc_valid)
            print(f"    [{i+1}/{n_batches}] loss={avg:.4f} ppl={math.exp(avg):.2f}")

    # Single host sync at the end
    total_loss = float(acc_loss)
    total_correct = int(acc_correct)
    total_valid = int(acc_valid)
    elapsed = time.time() - t0

    avg_loss = total_loss / total_valid
    ppl = math.exp(avg_loss)
    acc = total_correct / total_valid * 100

    print(f"\n  Results:")
    print(f"    Loss:       {avg_loss:.4f}")
    print(f"    Perplexity: {ppl:.2f}")
    print(f"    Accuracy:   {acc:.2f}%")
    print(f"    Tokens:     {total_valid:,}")
    print(f"    Time:       {elapsed:.1f}s ({total_valid/elapsed:.0f} tok/s)")

    return {'loss': avg_loss, 'perplexity': ppl, 'accuracy': acc}


# ============================================================
# D3: Neuron Health
# ============================================================

def analyze_neuron_health(model, params, cfg):
    print("\n" + "="*60)
    print("D3: Neuron Health")
    print("="*60)

    pool = params['neuron_pool']
    results = {}

    # Compute all stats on device, collect into one dict, then single sync
    @jax.jit
    def compute_pool_stats(emb, read, write):
        emb_n = jnp.linalg.norm(emb, axis=-1)
        read_n = jnp.linalg.norm(read, axis=-1)
        write_n = jnp.linalg.norm(write, axis=-1)
        return {
            'emb_mean': emb_n.mean(), 'emb_std': emb_n.std(),
            'emb_dead': (emb_n < 1e-6).sum(),
            'read_mean': read_n.mean(), 'read_std': read_n.std(),
            'read_dead': (read_n < 1e-6).sum(),
            'write_mean': write_n.mean(), 'write_std': write_n.std(),
            'write_dead': (write_n < 1e-6).sum(),
        }

    for pool_name, emb_key in [('QK', 'qk_emb'), ('V', 'v_emb'), ('Know', 'know_emb')]:
        emb = pool[emb_key]
        read = pool[emb_key.replace('emb', 'read')]
        write = pool[emb_key.replace('emb', 'write')]
        N = emb.shape[0]

        # Single device call + single host sync via jax.device_get
        s = jax.device_get(compute_pool_stats(emb, read, write))

        print(f"\n  {pool_name} Pool (N={N}):")
        for w in ('emb', 'read', 'write'):
            print(f"    {w:5s} norm: mean={s[f'{w}_mean']:.4f}, "
                  f"std={s[f'{w}_std']:.4f}, dead={int(s[f'{w}_dead'])}")

        results[pool_name] = {
            'N': N,
            'emb_norm_mean': float(s['emb_mean']),
            'read_norm_mean': float(s['read_mean']),
            'write_norm_mean': float(s['write_mean']),
            'dead_emb': int(s['emb_dead']),
            'dead_read': int(s['read_dead']),
            'dead_write': int(s['write_dead']),
        }

    # Router tau bias — single sync
    router = params['router']
    tau_attn = jax.device_get(router['tau_attn']['bias'])
    tau_know = jax.device_get(router['tau_know']['bias'])
    print(f"\n  Tau bias:")
    print(f"    attn: [{', '.join(f'{v:.3f}' for v in tau_attn)}]")
    print(f"    know: [{tau_know[0]:.3f}]")

    return results


# ============================================================
# D4: Generation
# ============================================================

def generate(model, params, cfg, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    """Autoregressive generation with DAWN-Spatial.

    Uses fixed-length padded input to avoid JIT recompilation per token.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_seq = cfg['model'].get('max_seq_len', 512)
    gen_len = min(max_new_tokens, max_seq - len(input_ids))

    # Single jitted step: forward + sample, no host round-trip per token
    @jax.jit
    def generate_step(params, ids_padded, seq_pos, rng):
        logits = model.apply(
            {'params': params}, ids_padded[None, :],
            deterministic=True,
            rngs={'dropout': jax.random.PRNGKey(0)})['logits'][0, seq_pos, :]
        logits = logits / temperature
        top_vals, _ = jax.lax.top_k(logits, top_k)
        logits = jnp.where(logits >= top_vals[-1], logits, -1e10)
        rng, sample_rng = jax.random.split(rng)
        next_token = jax.random.categorical(sample_rng, logits)
        # Update ids in-place on device
        new_ids = jax.lax.dynamic_update_index_in_dim(
            ids_padded, next_token.astype(jnp.int32), seq_pos + 1, axis=0)
        return new_ids, next_token, rng

    print(f"\n  Prompt: \"{prompt}\" ({len(input_ids)} tokens)")
    print(f"  Generating up to {gen_len} tokens (temp={temperature}, top_k={top_k})...")
    print(f"  ---")

    # Build initial padded ids on device (single transfer)
    ids_padded = np.zeros(max_seq, dtype=np.int32)
    ids_padded[:len(input_ids)] = input_ids
    ids_dev = jnp.array(ids_padded)
    rng = jax.random.PRNGKey(42)

    # Warmup JIT
    _ = generate_step(params, ids_dev, jnp.int32(len(input_ids) - 1), rng)

    t0 = time.time()
    pos = len(input_ids) - 1
    stop_ids = {tokenizer.sep_token_id, tokenizer.pad_token_id}
    n_gen = 0

    for i in range(gen_len):
        if pos >= max_seq - 1:
            break
        ids_dev, next_token, rng = generate_step(params, ids_dev, jnp.int32(pos), rng)
        pos += 1
        n_gen += 1
        # Only sync to check stop token (cheap: single int)
        tok_id = int(next_token)
        if tok_id in stop_ids:
            break

    elapsed = time.time() - t0

    # Single sync: get all generated ids from device
    all_ids = jax.device_get(ids_dev)[:pos + 1].tolist()
    output_text = tokenizer.decode(all_ids, skip_special_tokens=True)
    gen_text = tokenizer.decode(all_ids[len(input_ids):], skip_special_tokens=True)

    print(f"  {output_text}")
    print(f"  ---")
    tok_s = n_gen / elapsed if elapsed > 0 else 0
    print(f"  Generated {n_gen} tokens in {elapsed:.1f}s ({tok_s:.1f} tok/s)")

    return {'prompt': prompt, 'output': output_text, 'generated': gen_text,
            'n_tokens': n_gen, 'tok_per_sec': tok_s}


# ============================================================
# D5: Weight Analysis
# ============================================================

def analyze_weights(params, cfg):
    print("\n" + "="*60)
    print("D5: Weight Analysis")
    print("="*60)

    pool = params['neuron_pool']
    results = {}

    for pool_name, emb_key in [('QK', 'qk_emb'), ('V', 'v_emb'), ('Know', 'know_emb')]:
        emb = np.array(pool[emb_key])
        N, d = emb.shape

        # Cosine similarity stats without materializing N×N matrix
        norms = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8
        emb_normed = emb / norms
        # mean |cos_sim| ≈ sample mean of |a·b| for random pairs
        n_sample = min(N, 1024)
        idx = np.random.default_rng(42).choice(N, n_sample, replace=False)
        sample = emb_normed[idx]
        gram = sample @ sample.T  # [1024, 1024] — manageable
        np.fill_diagonal(gram, 0)
        mean_sim = float(np.abs(gram).mean() * n_sample / (n_sample - 1))
        max_sim = float(np.abs(gram).max())

        # Effective rank (via SVD on small sample)
        sv = np.linalg.svd(sample, compute_uv=False)
        sv_norm = sv / (sv.sum() + 1e-8)
        entropy = -float((sv_norm * np.log(sv_norm + 1e-10)).sum())
        eff_rank = float(np.exp(entropy))

        print(f"\n  {pool_name} (N={N}, d={d}):")
        print(f"    Cosine sim: mean={mean_sim:.4f}, max={max_sim:.4f} (sampled {n_sample})")
        print(f"    Effective rank: {eff_rank:.1f} / {min(n_sample, d)}")
        print(f"    Top-5 SVs: {', '.join(f'{v:.2f}' for v in sv[:5])}")

        results[pool_name] = {
            'mean_cosine_sim': mean_sim,
            'max_cosine_sim': max_sim,
            'effective_rank': eff_rank,
        }

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DAWN-Spatial v3 Analysis")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (local or GCS)")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--val_data", default=None, help="Validation .bin path")
    parser.add_argument("--output", default="results/spatial_analysis", help="Output directory")
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--only", default=None, help="Run only: info,val,health,generate,weights")
    parser.add_argument("--prompt", default="The meaning of life is",
                        help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Build model
    model = build_model(cfg)

    # Load checkpoint
    params, ckpt_data = load_checkpoint_params(args.checkpoint, model, cfg)

    # Determine which analyses to run
    only = set(args.only.split(',')) if args.only else None
    results = {}

    if only is None or 'info' in only:
        results['model_info'] = analyze_model_info(model, params, cfg)

    if only is None or 'val' in only:
        val_path = args.val_data or cfg.get('data', {}).get('bin_val')
        if val_path:
            results['validation'] = analyze_validation(
                model, params, cfg, val_path, args.max_batches, args.batch_size)
        else:
            print("\n  Skipping validation (no --val_data or data.bin_val in config)")

    if only is None or 'health' in only:
        results['neuron_health'] = analyze_neuron_health(model, params, cfg)

    if only is None or 'generate' in only:
        results['generation'] = generate(
            model, params, cfg, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature)

    if only is None or 'weights' in only:
        results['weight_analysis'] = analyze_weights(params, cfg)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, 'analysis_results.json')

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        if isinstance(obj, (np.floating, jnp.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
