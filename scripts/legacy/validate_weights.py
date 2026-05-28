#!/usr/bin/env python3
"""Validate DAWN weights on C4 validation set (streaming).

Uses same data processing as training:
- Concatenate all tokens into flat stream (no special tokens)
- Reshape to [N, seq_len] sequences
- No padding within sequences

Supports all DAWN versions (auto-detection from checkpoint).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm
import math

from models import create_model_by_version, normalize_version


def load_model_from_checkpoint(weights_path, device='cpu'):
    """Load model from checkpoint with auto version detection."""
    ckpt = torch.load(weights_path, map_location=device)

    # Get config and state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        config = ckpt.get('model_config', ckpt.get('config', {}))
    else:
        state_dict = ckpt
        config = {}

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

    return model, config, version


def validate(model, dataloader, device, max_batches=200):
    """Evaluate model - matches training eval exactly"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, (input_ids,) in enumerate(tqdm(dataloader, desc="Evaluating", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = input_ids.to(device)
            # No padding in this data → attention_mask is all 1s
            attention_mask = torch.ones_like(input_ids)

            # CLM evaluation
            logits = model(input_ids, attention_mask=attention_mask)

            # Shift for autoregressive loss (no padding to mask)
            B, S, V = logits.shape
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous().long()

            # All tokens are valid (no padding)
            valid_tokens = shift_labels.numel()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1)
            )

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, help='Path to weights file')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_batches', type=int, default=200,
                       help='Max batches to evaluate (default: 200)')
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # Load model with auto version detection
    print(f"Loading model from {args.weights}")
    model, config, version = load_model_from_checkpoint(args.weights, device='cpu')
    model = model.to(args.device)
    model.eval()

    # Load tokenizer (same as training: bert-base-uncased)
    print("Loading tokenizer (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Calculate how many tokens we need
    target_tokens = args.max_batches * args.batch_size * args.seq_len
    print(f"Target tokens: {target_tokens:,}")

    # Load C4 validation (streaming) - same as prepare_val_data.py
    print("Loading C4 validation set (streaming)...")
    dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)

    # Collect tokens - SAME as training data prep
    all_tokens = []
    pbar = tqdm(total=target_tokens, desc="Tokenizing", unit="tok", unit_scale=True)

    for example in dataset:
        # Same as prepare_val_data.py: encode without special tokens
        tokens = tokenizer.encode(example['text'], add_special_tokens=False)
        all_tokens.extend(tokens)
        pbar.update(len(tokens))

        if len(all_tokens) >= target_tokens:
            break

    pbar.close()

    # Trim and reshape - same as training
    all_tokens = all_tokens[:target_tokens]
    num_seqs = len(all_tokens) // args.seq_len
    tokens_tensor = torch.tensor(all_tokens[:num_seqs * args.seq_len], dtype=torch.long)
    tokens_tensor = tokens_tensor.view(num_seqs, args.seq_len)

    print(f"Prepared {num_seqs} sequences of length {args.seq_len}")

    # Create dataloader
    dataset = TensorDataset(tokens_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Validate
    avg_loss, ppl = validate(model, dataloader, args.device, args.max_batches)

    print(f"\n{'='*40}")
    print(f"Validation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  PPL:  {ppl:.2f}")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
