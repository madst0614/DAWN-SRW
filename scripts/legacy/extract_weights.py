#!/usr/bin/env python3
"""Extract model weights and config from checkpoint (without optimizer state)."""

import argparse
import torch
from pathlib import Path


def extract_weights(checkpoint_path: str, output_path: str = None):
    """Extract model state_dict and config from checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Show what's in the checkpoint
    print(f"Checkpoint keys: {list(ckpt.keys())}")

    # Extract only what we need
    extracted = {}

    if 'model_state_dict' in ckpt:
        extracted['model_state_dict'] = ckpt['model_state_dict']
        print(f"  model_state_dict: {len(ckpt['model_state_dict'])} parameters")

    if 'config' in ckpt:
        extracted['config'] = ckpt['config']
        print(f"  config: {ckpt['config']}")

    if 'epoch' in ckpt:
        extracted['epoch'] = ckpt['epoch']

    if 'global_step' in ckpt:
        extracted['global_step'] = ckpt['global_step']

    # Default output path
    if output_path is None:
        output_path = checkpoint_path.parent / "dawn_24m_weights.pt"

    print(f"\nSaving to: {output_path}")
    torch.save(extracted, output_path)

    # Size comparison
    original_size = checkpoint_path.stat().st_size / (1024 * 1024)
    new_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nSize: {original_size:.1f}MB -> {new_size:.1f}MB ({new_size/original_size*100:.1f}%)")

    return extracted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract model weights from checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output path (default: dawn_24m_weights.pt in same dir)')

    args = parser.parse_args()
    extract_weights(args.checkpoint, args.output)
