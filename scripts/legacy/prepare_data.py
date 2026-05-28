#!/usr/bin/env python3
"""
DAWN Data Preparation Script

Prepares C4 dataset for training DAWN and baseline models.
Downloads from HuggingFace and tokenizes into chunks.

Usage:
    # Default: 5B tokens for full training
    python prepare_data.py --output-dir ./data/train/c4

    # Quick start: 500M tokens for testing
    python prepare_data.py --output-dir ./data/train/c4 --target-tokens 500_000_000

    # Resume interrupted download
    python prepare_data.py --output-dir ./data/train/c4 --resume

Requirements:
    pip install datasets transformers torch tqdm
"""

import argparse
import os
import gc
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare C4 dataset for DAWN training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save tokenized chunks"
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=5_000_000_000,
        help="Total tokens to collect (default: 5B)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000_000,
        help="Tokens per chunk file (default: 500M)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace tokenizer to use (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing chunks"
    )
    return parser.parse_args()


def find_resume_point(save_dir: Path):
    """Find existing chunks and calculate resume point"""
    existing_chunks = sorted([
        f for f in os.listdir(save_dir)
        if f.startswith('c4_raw_') and f.endswith('.pt')
    ])

    if not existing_chunks:
        return 0, 0

    chunk_num = len(existing_chunks)
    last_chunk_path = save_dir / existing_chunks[-1]
    last_chunk = torch.load(last_chunk_path)
    skip_examples = last_chunk['total_examples']
    del last_chunk
    gc.collect()

    return chunk_num, skip_examples


def main():
    args = parse_args()

    # Setup
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DAWN Data Preparation")
    print("=" * 60)
    print(f"Output directory: {save_dir}")
    print(f"Target tokens: {args.target_tokens:,}")
    print(f"Chunk size: {args.chunk_size:,}")
    print(f"Tokenizer: {args.tokenizer}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size:,}")

    # Check for resume
    chunk_num = 0
    skip_examples = 0

    if args.resume:
        chunk_num, skip_examples = find_resume_point(save_dir)
        if chunk_num > 0:
            print(f"\nResuming from chunk {chunk_num}, skipping {skip_examples:,} examples")

    # Calculate expected chunks
    expected_chunks = args.target_tokens // args.chunk_size
    if chunk_num >= expected_chunks:
        print(f"\nAlready completed! {chunk_num} chunks exist.")
        return

    # Load C4 streaming
    print(f"\nLoading C4 dataset (streaming)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    if skip_examples > 0:
        print(f"Skipping {skip_examples:,} examples...")
        dataset = dataset.skip(skip_examples)

    # Tokenize and save
    all_tokens = []
    examples_processed = skip_examples
    initial_tokens = chunk_num * args.chunk_size

    pbar = tqdm(
        total=args.target_tokens,
        initial=initial_tokens,
        desc="Tokenizing",
        unit="tok",
        unit_scale=True
    )

    try:
        for example in dataset:
            # Tokenize
            tokens = tokenizer.encode(example['text'], add_special_tokens=False)
            all_tokens.extend(tokens)
            examples_processed += 1
            pbar.update(len(tokens))

            # Save chunk when full
            while len(all_tokens) >= args.chunk_size:
                chunk_tokens = all_tokens[:args.chunk_size]
                all_tokens = all_tokens[args.chunk_size:]

                filename = save_dir / f'c4_raw_{chunk_num:03d}.pt'
                torch.save({
                    'tokens': torch.tensor(chunk_tokens, dtype=torch.int32),
                    'total_examples': examples_processed
                }, filename)

                tqdm.write(f"Saved: {filename.name} ({examples_processed:,} examples)")
                chunk_num += 1

                # Memory cleanup
                del chunk_tokens
                gc.collect()

            # Check if done
            if chunk_num >= expected_chunks:
                break

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Progress saved at chunk {chunk_num}")
        print(f"Run with --resume to continue")

    finally:
        pbar.close()

    print(f"\n" + "=" * 60)
    print(f"Done!")
    print(f"  Chunks saved: {chunk_num}")
    print(f"  Examples processed: {examples_processed:,}")
    print(f"  Output directory: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
