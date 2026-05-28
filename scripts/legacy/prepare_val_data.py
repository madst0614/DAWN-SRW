#!/usr/bin/env python3
"""
Prepare validation data from C4

Creates a separate validation set from C4 validation split.

Usage:
    python prepare_val_data.py --output-dir ./data/val/c4

    # Custom size
    python prepare_val_data.py --output-dir ./data/val/c4 --target-tokens 50_000_000
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare C4 validation dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save validation data"
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=50_000_000,
        help="Total validation tokens (default: 50M)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace tokenizer (default: bert-base-uncased)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DAWN Validation Data Preparation")
    print("=" * 60)
    print(f"Output: {save_dir}")
    print(f"Target tokens: {args.target_tokens:,}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load C4 validation split (streaming)
    print(f"\nLoading C4 validation split...")
    dataset = load_dataset(
        "allenai/c4", "en",
        split="validation",
        streaming=True
    )

    # Collect tokens
    all_tokens = []
    examples = 0

    pbar = tqdm(total=args.target_tokens, desc="Tokenizing", unit="tok", unit_scale=True)

    for example in dataset:
        tokens = tokenizer.encode(example['text'], add_special_tokens=False)
        all_tokens.extend(tokens)
        examples += 1
        pbar.update(len(tokens))

        if len(all_tokens) >= args.target_tokens:
            break

    pbar.close()

    # Trim to exact size
    all_tokens = all_tokens[:args.target_tokens]

    # Save
    output_file = save_dir / f'c4_val_{args.target_tokens // 1_000_000}M.pt'
    torch.save({
        'tokens': torch.tensor(all_tokens, dtype=torch.int32),
        'total_examples': examples
    }, output_file)

    print(f"\n" + "=" * 60)
    print(f"Saved: {output_file}")
    print(f"  Tokens: {len(all_tokens):,}")
    print(f"  Examples: {examples:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
