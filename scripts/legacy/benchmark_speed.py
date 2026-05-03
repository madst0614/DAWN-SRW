#!/usr/bin/env python3
"""
DAWN vs Vanilla Speed Benchmark
================================
Compares inference speed (ms/token) between DAWN and Vanilla transformers.

Supports all DAWN versions (auto-detection from checkpoint).

Usage:
    python benchmark_speed.py --folder checkpoints/  # Auto-discover
    python benchmark_speed.py --dawn_ckpt path/to/dawn.pt --vanilla_ckpt path/to/vanilla.pt
    python benchmark_speed.py --dawn_ckpt path/to/dawn.pt  # DAWN only

Auto-discovery looks for:
    - DAWN: files containing 'dawn', 'v18', 'v17', 'v16', 'v15', 'v14'
    - Vanilla: files containing 'vanilla', 'baseline', 'gpt', 'transformer'
    - Prefers 'best' or 'final' checkpoints
"""

import argparse
import sys
import os
import time
import glob
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_checkpoints(folder, pattern="*.pt"):
    """Auto-discover checkpoints in folder"""
    if not os.path.isdir(folder):
        return None, None

    all_ckpts = glob.glob(os.path.join(folder, "**", pattern), recursive=True)
    all_ckpts += glob.glob(os.path.join(folder, "**", "*.pth"), recursive=True)

    dawn_ckpt = None
    vanilla_ckpt = None

    for ckpt in all_ckpts:
        name = os.path.basename(ckpt).lower()
        if any(x in name for x in ['dawn', 'v18', 'v17', 'v16', 'v15', 'v14']):
            if dawn_ckpt is None or 'best' in name or 'final' in name:
                dawn_ckpt = ckpt
        elif any(x in name for x in ['vanilla', 'baseline', 'gpt', 'transformer']):
            if vanilla_ckpt is None or 'best' in name or 'final' in name:
                vanilla_ckpt = ckpt

    # Fallback: if no specific match, use most recent
    if dawn_ckpt is None and all_ckpts:
        # Sort by modification time
        all_ckpts.sort(key=os.path.getmtime, reverse=True)
        dawn_ckpt = all_ckpts[0]
        if len(all_ckpts) > 1:
            vanilla_ckpt = all_ckpts[1]

    return dawn_ckpt, vanilla_ckpt


def resolve_checkpoint_path(path):
    """Resolve path to actual checkpoint file (handles directories)"""
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        # Search for checkpoint files in directory
        patterns = ["*.pt", "*.pth", "checkpoint*.pt", "best*.pt", "final*.pt"]
        all_ckpts = []
        for pattern in patterns:
            all_ckpts.extend(glob.glob(os.path.join(path, pattern)))
            all_ckpts.extend(glob.glob(os.path.join(path, "**", pattern), recursive=True))

        if all_ckpts:
            # Prefer 'best' or 'final', otherwise use most recent
            for ckpt in all_ckpts:
                name = os.path.basename(ckpt).lower()
                if 'best' in name or 'final' in name:
                    print(f"  Found checkpoint: {ckpt}")
                    return ckpt

            # Sort by modification time, use most recent
            all_ckpts.sort(key=os.path.getmtime, reverse=True)
            print(f"  Found checkpoint: {all_ckpts[0]}")
            return all_ckpts[0]

        raise FileNotFoundError(f"No checkpoint files found in directory: {path}")

    raise FileNotFoundError(f"Path does not exist: {path}")


def load_dawn_model(ckpt_path, device='cuda'):
    """Load DAWN model from checkpoint with auto version detection (handles directory paths)"""
    from models import create_model_by_version, normalize_version

    # Resolve directory to actual checkpoint file
    ckpt_path = resolve_checkpoint_path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Get config and state_dict
    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Clean compiled model prefix
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Auto-detect version from config or state_dict keys
    version = config.get('model_version', None)
    if version is None:
        v18_2_keys = ['router.tau_proj.weight', 'router.neuron_router.norm_fqk_Q.weight']
        dawn_keys = ['shared_neurons.f_neurons', 'router.neuron_router.neuron_emb']

        if all(k in state_dict for k in v18_2_keys):
            version = '18.2'
        elif any(k in state_dict for k in dawn_keys):
            if config.get('learnable_tau', False) or config.get('max_paths'):
                version = '18.2'
            else:
                version = '17.1'
        else:
            version = 'baseline'

    version = normalize_version(version)
    print(f"  Detected version: {version}")

    model = create_model_by_version(version, config)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DAWN loaded: {n_params:,} params")
    print(f"  Config: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")

    return model, config


def load_vanilla_model(ckpt_path, device='cuda'):
    """Load Vanilla transformer from checkpoint (handles directory paths)"""
    # Resolve directory to actual checkpoint file
    ckpt_path = resolve_checkpoint_path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Clean compiled model prefix
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Try different model imports
    try:
        from models import VanillaTransformer
        model = VanillaTransformer(**config)
        model.load_state_dict(state_dict, strict=False)
    except (ImportError, Exception):
        # Fallback: try model_vanilla
        try:
            from models.model_vanilla import VanillaTransformer
            model = VanillaTransformer(**config)
            model.load_state_dict(state_dict, strict=False)
        except (ImportError, Exception):
            # Fallback: try to load as generic transformer
            try:
                from transformers import GPT2LMHeadModel, GPT2Config

                gpt2_config = GPT2Config(
                    vocab_size=config.get('vocab_size', 30522),
                    n_embd=config.get('d_model', 512),
                    n_layer=config.get('n_layers', 12),
                    n_head=config.get('n_heads', 8),
                )
                model = GPT2LMHeadModel(gpt2_config)
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Error loading vanilla model: {e}")
                return None, None

    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vanilla loaded: {n_params:,} params")

    return model, config


def benchmark_model(model, seq_len=512, batch_size=1, warmup=10, iterations=100, device='cuda'):
    """Benchmark model inference speed"""

    # Create dummy input
    vocab_size = 30522  # BERT vocab size
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)

    # Synchronize before timing
    if device != 'cpu':
        torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({iterations} iterations)...")
    times = []

    for _ in range(iterations):
        if device != 'cpu':
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(x)

        if device != 'cpu':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    total_tokens = batch_size * seq_len

    results = {
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'min_ms': times.min() * 1000,
        'max_ms': times.max() * 1000,
        'ms_per_token': times.mean() * 1000 / total_tokens,
        'tokens_per_sec': total_tokens / times.mean(),
        'iterations': iterations,
        'batch_size': batch_size,
        'seq_len': seq_len,
    }

    return results


def print_results(name, results):
    """Print benchmark results"""
    print(f"\n{'='*50}")
    print(f"{name} Results")
    print(f"{'='*50}")
    print(f"  Batch size: {results['batch_size']}, Seq length: {results['seq_len']}")
    print(f"  Mean time: {results['mean_ms']:.2f} ms (std: {results['std_ms']:.2f})")
    print(f"  Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
    print(f"  ms/token: {results['ms_per_token']:.4f}")
    print(f"  Tokens/sec: {results['tokens_per_sec']:,.0f}")


def compare_results(dawn_results, vanilla_results):
    """Compare DAWN vs Vanilla results"""
    print(f"\n{'='*50}")
    print("COMPARISON")
    print(f"{'='*50}")

    speed_ratio = dawn_results['ms_per_token'] / vanilla_results['ms_per_token']
    throughput_ratio = dawn_results['tokens_per_sec'] / vanilla_results['tokens_per_sec']

    print(f"  DAWN ms/token:    {dawn_results['ms_per_token']:.4f}")
    print(f"  Vanilla ms/token: {vanilla_results['ms_per_token']:.4f}")
    print(f"  Speed ratio: {speed_ratio:.2f}x (DAWN/Vanilla)")
    print(f"  Throughput ratio: {throughput_ratio:.2f}x (DAWN/Vanilla)")

    if speed_ratio < 1.1:
        print(f"\n  [VERDICT] DAWN is similar speed (~{speed_ratio:.1f}x)")
    elif speed_ratio < 1.5:
        print(f"\n  [VERDICT] DAWN is {(speed_ratio-1)*100:.0f}% slower")
    else:
        print(f"\n  [VERDICT] DAWN is {speed_ratio:.1f}x slower - tradeoff needed")

    return {
        'speed_ratio': speed_ratio,
        'throughput_ratio': throughput_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DAWN vs Vanilla speed")
    parser.add_argument("--folder", type=str, default=None, help="Folder to auto-discover checkpoints")
    parser.add_argument("--dawn_ckpt", type=str, default=None, help="Path to DAWN checkpoint")
    parser.add_argument("--vanilla_ckpt", type=str, default=None, help="Path to Vanilla checkpoint")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Auto-discover checkpoints if folder provided
    if args.folder:
        print(f"Searching for checkpoints in: {args.folder}")
        found_dawn, found_vanilla = find_checkpoints(args.folder)
        if found_dawn:
            print(f"  Found DAWN: {found_dawn}")
            args.dawn_ckpt = args.dawn_ckpt or found_dawn
        if found_vanilla:
            print(f"  Found Vanilla: {found_vanilla}")
            args.vanilla_ckpt = args.vanilla_ckpt or found_vanilla

    if not args.dawn_ckpt:
        parser.error("--dawn_ckpt required (or use --folder for auto-discovery)")

    print(f"\n{'='*50}")
    print("DAWN vs Vanilla Speed Benchmark")
    print(f"{'='*50}")
    print(f"Device: {args.device}")
    print(f"Seq length: {args.seq_len}, Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")

    # GPU info
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load and benchmark DAWN
    print(f"\n[1] Loading DAWN: {args.dawn_ckpt}")
    dawn_model, dawn_config = load_dawn_model(args.dawn_ckpt, args.device)

    print(f"\n[2] Benchmarking DAWN...")
    dawn_results = benchmark_model(
        dawn_model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device
    )
    print_results("DAWN", dawn_results)

    # Cleanup DAWN
    del dawn_model
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    # Load and benchmark Vanilla if provided
    if args.vanilla_ckpt:
        print(f"\n[3] Loading Vanilla: {args.vanilla_ckpt}")
        vanilla_model, vanilla_config = load_vanilla_model(args.vanilla_ckpt, args.device)

        if vanilla_model is not None:
            print(f"\n[4] Benchmarking Vanilla...")
            vanilla_results = benchmark_model(
                vanilla_model,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                warmup=args.warmup,
                iterations=args.iterations,
                device=args.device
            )
            print_results("Vanilla", vanilla_results)

            # Compare
            comparison = compare_results(dawn_results, vanilla_results)

            # Cleanup
            del vanilla_model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    else:
        print("\n[3] Vanilla checkpoint not provided, DAWN-only benchmark complete.")

    print(f"\n{'='*50}")
    print("Benchmark Complete")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
