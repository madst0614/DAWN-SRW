#!/usr/bin/env python3
"""Test extracted DAWN weights."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models import DAWN

parser = argparse.ArgumentParser()
parser.add_argument('weights', type=str, nargs='?', default='dawn_24m_weights.pt',
                   help='Path to weights file (default: dawn_24m_weights.pt)')
args = parser.parse_args()

# 1. Config (v17.1)
config = {
    'vocab_size': 30522,
    'd_model': 384,
    'n_layers': 12,
    'n_heads': 6,
    'rank': 64,
    'knowledge_rank': 128,
    'n_feature_qk': 120,
    'n_feature_v': 24,
    'n_restore_qk': 120,
    'n_restore_v': 24,
    'n_feature_know': 24,
    'n_restore_know': 24,
    'top_k_feature_qk': 20,
    'top_k_feature_v': 6,
    'top_k_restore_qk': 20,
    'top_k_restore_v': 6,
    'top_k_feature_know': 4,
    'top_k_restore_know': 4,
}

# 2. Load model
print(f"Loading: {args.weights}")
model = DAWN(**config)
ckpt = torch.load(args.weights, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"Loaded config: {ckpt.get('config')}")
print(f"Global step: {ckpt.get('global_step')}")

# 3. Forward pass test
dummy_input = torch.randint(0, 1000, (1, 64))
with torch.no_grad():
    output = model(dummy_input)
print(f"Output shape: {output.shape}")  # [1, 64, 30522]

# 4. Parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")
