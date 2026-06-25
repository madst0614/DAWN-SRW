"""
Pretokenize C4 dataset for DAWN TPU training.

Streams C4 from HuggingFace, tokenizes with bert-base-uncased,
packs into fixed-length sequences (no padding), and saves as
flat numpy binary (.bin) with uint16 dtype.

Designed for TPU training pipeline — outputs are consumed by
utils/data_jax.py (pure numpy/JAX data loader, no PyTorch).

Supports shard splitting for large datasets (--num_shards).
Supports --resume to continue from a previous run (reads existing _meta.json,
skips already-processed documents, and appends new shards).

Usage:
    # Single file
    python scripts/pretokenize_c4.py --split train --num_tokens 20_000_000_000 --output gs://my-bucket/data/c4_train.bin

    # Sharded (20 shards, ~1B tokens each)
    python scripts/pretokenize_c4.py --split train --num_tokens 20_000_000_000 --num_shards 20 --output gs://my-bucket/data/c4_train

    # Validation
    python scripts/pretokenize_c4.py --split validation --num_tokens 10_000_000 --output ./data/c4_val.bin

    # Resume: extend 40B -> 60B (skips first 40B worth of docs, appends new shards)
    python scripts/pretokenize_c4.py --split train --num_tokens 60_000_000_000 --num_shards 60 --output gs://my-bucket/data/c4_train --resume
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def open_output(path: str, mode: str = "wb"):
    """Open a file for writing, supporting both local and GCS paths."""
    if is_gcs_path(path):
        try:
            from google.cloud import storage
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            return fs.open(path, mode)
        except ImportError:
            try:
                import tensorflow as tf
                return tf.io.gfile.GFile(path, mode)
            except ImportError:
                raise ImportError(
                    "GCS output requires 'gcsfs' or 'tensorflow'. "
                    "Install with: pip install gcsfs  OR  pip install tensorflow"
                )
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return open(path, mode)


def output_exists(path: str) -> bool:
    """Check whether an output path exists, supporting both local and GCS paths."""
    if is_gcs_path(path):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            return fs.exists(path)
        except ImportError:
            try:
                import tensorflow as tf
                return tf.io.gfile.exists(path)
            except ImportError:
                raise ImportError(
                    "GCS output requires 'gcsfs' or 'tensorflow'. "
                    "Install with: pip install gcsfs  OR  pip install tensorflow"
                )
    return os.path.exists(path)


def open_new_output(path: str):
    """Open a new binary output file, refusing to overwrite existing data."""
    if output_exists(path):
        raise RuntimeError(
            f"Refusing to overwrite existing output file: {path}. "
            "Metadata and shards may be inconsistent, or a previous resume partially ran."
        )
    return open_output(path, "wb")


def read_json(path: str) -> dict:
    """Read JSON from local or GCS."""
    if is_gcs_path(path):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(path, 'r') as f:
                return json.load(f)
        except ImportError:
            import tensorflow as tf
            with tf.io.gfile.GFile(path, 'r') as f:
                return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def write_json(path: str, data: dict):
    """Write JSON metadata, supporting both local and GCS paths."""
    content = json.dumps(data, indent=2)
    if is_gcs_path(path):
        with open_output(path, "w") as f:
            f.write(content)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)


def shard_path(base: str, shard_idx: int) -> str:
    """Generate shard file path: base_000.bin, base_001.bin, ..."""
    return f"{base}_{shard_idx:03d}.bin"


def require_nonnegative_int(metadata: dict, key: str) -> int:
    if key not in metadata:
        raise RuntimeError(f"Resume metadata missing required field '{key}'")
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            f"Resume metadata field '{key}' must be a non-negative integer, got {value!r}"
        )
    return value


def resume_docs_from_metadata(metadata: dict) -> int:
    if "docs_consumed" in metadata:
        return require_nonnegative_int(metadata, "docs_consumed")
    if "docs_processed" in metadata:
        return require_nonnegative_int(metadata, "docs_processed")
    raise RuntimeError(
        "Resume metadata missing docs_consumed/docs_processed; refusing to resume."
    )


def main():
    parser = argparse.ArgumentParser(description="Pretokenize C4 for DAWN training")
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation"],
                        help="Dataset split to process")
    parser.add_argument("--num_tokens", type=int, required=True,
                        help="Target number of tokens (e.g. 20_000_000_000 for 20B)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path (.bin for single file, or base name for shards)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of shards to split into (default: 1 = single file)")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length for packing (default: 512)")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased",
                        help="HuggingFace tokenizer name (default: bert-base-uncased)")
    parser.add_argument("--flush_every", type=int, default=100_000,
                        help="Flush buffer to disk every N sequences (default: 100000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run. Reads existing _meta.json, "
                             "skips already-processed docs, appends new shards.")
    args = parser.parse_args()

    seq_len = args.seq_len
    target_tokens = args.num_tokens
    target_seqs = target_tokens // seq_len
    num_shards = args.num_shards

    # Determine output base name
    output_base = args.output
    if output_base.endswith(".bin"):
        output_base = output_base[:-4]  # strip .bin

    is_sharded = num_shards > 1
    seqs_per_shard = target_seqs // num_shards if is_sharded else target_seqs

    # ---- Resume: load previous state ----
    resume_docs = 0
    resume_tokens = 0
    resume_seqs = 0
    resume_shards = 0
    prev_shard_infos = []
    prev_tail_tokens = []  # leftover token_buffer from previous run

    meta_path = output_base + "_meta.json"

    if args.resume:
        try:
            prev_meta = read_json(meta_path)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"--resume requested but metadata not found at {meta_path}; "
                "refusing to start from scratch."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"--resume requested but metadata at {meta_path} could not be read; "
                f"refusing to start from scratch: {e}"
            ) from e
        if not isinstance(prev_meta, dict):
            raise RuntimeError(
                f"--resume requested but metadata at {meta_path} is not a JSON object; "
                "refusing to start from scratch."
            )

        resume_docs = resume_docs_from_metadata(prev_meta)
        resume_tokens = require_nonnegative_int(prev_meta, "total_tokens")
        resume_seqs = require_nonnegative_int(prev_meta, "total_sequences")
        resume_shards = require_nonnegative_int(prev_meta, "num_shards")

        prev_seq_len = prev_meta.get("seq_len", seq_len)
        if prev_seq_len != seq_len:
            raise RuntimeError(
                f"Resume metadata seq_len={prev_seq_len} does not match --seq_len={seq_len}"
            )
        if resume_tokens != resume_seqs * seq_len:
            raise RuntimeError(
                f"Resume metadata total_tokens={resume_tokens:,} does not match "
                f"total_sequences*seq_len={resume_seqs * seq_len:,}"
            )

        prev_shard_infos = prev_meta.get("shards", [])
        if not isinstance(prev_shard_infos, list):
            raise RuntimeError("Resume metadata field 'shards' must be a list")
        if len(prev_shard_infos) != resume_shards:
            raise RuntimeError(
                f"Resume metadata says num_shards={resume_shards}, but has "
                f"{len(prev_shard_infos)} shard records; refusing to guess next shard."
            )

        if "tail_token_buffer" in prev_meta:
            prev_tail_tokens = prev_meta["tail_token_buffer"]
            if not isinstance(prev_tail_tokens, list):
                raise RuntimeError("Resume metadata field 'tail_token_buffer' must be a list")
        else:
            prev_tail_tokens = []

        print(f"{'='*60}")
        print(f"RESUME MODE")
        print(f"{'='*60}")
        print(f"Previous run: {resume_tokens:,} tokens, {resume_docs:,} docs, {resume_shards} shards")
        print(f"Remaining: {target_tokens - resume_tokens:,} tokens to collect")
        print(f"Will skip {resume_docs:,} docs then continue...")
        if prev_tail_tokens:
            print(f"Restoring {len(prev_tail_tokens)} leftover tokens from previous run")
        elif "tail_token_buffer" not in prev_meta and "discarded_tail_tokens" in prev_meta:
            discarded = prev_meta["discarded_tail_tokens"]
            print(
                "Legacy metadata has no tail_token_buffer; "
                f"discarded_tail_tokens={discarded} will remain discarded."
            )
        print()

        if resume_tokens >= target_tokens:
            print(f"Already have {resume_tokens:,} >= target {target_tokens:,}. Nothing to do.")
            return
        if is_sharded and resume_shards >= num_shards:
            raise RuntimeError(
                f"Resume metadata already has {resume_shards} shards, but --num_shards={num_shards}; "
                "increase --num_shards or verify metadata before resuming."
            )

    print(f"{'='*60}")
    print(f"Pretokenize C4 (TPU)")
    print(f"{'='*60}")
    print(f"Split: {args.split}")
    print(f"Target tokens: {target_tokens:,}")
    print(f"Target sequences: {target_seqs:,} (seq_len={seq_len})")
    if args.resume and resume_tokens > 0:
        print(f"Already collected: {resume_tokens:,} tokens ({resume_tokens/target_tokens*100:.1f}%)")
    if is_sharded:
        print(f"Total shards: {num_shards}")
        print(f"Sequences per shard: ~{seqs_per_shard:,}")
        print(f"Output pattern: {output_base}_XXX.bin")
    else:
        print(f"Output: {output_base}.bin")
    print(f"Tokenizer: {args.tokenizer}")

    # ---- Load tokenizer ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    assert vocab_size <= 65535, f"Vocab size {vocab_size} exceeds uint16 max (65535)"

    # ---- Stream C4 dataset ----
    from datasets import load_dataset
    print(f"\nLoading C4 ({args.split}) in streaming mode...")
    dataset = load_dataset("allenai/c4", "en", split=args.split, streaming=True)

    # ---- Skip docs if resuming ----
    data_iter = iter(dataset)
    if args.resume and resume_docs > 0:
        print(f"Skipping {resume_docs:,} previously processed docs...")
        skip_start = time.time()
        for i in range(resume_docs):
            try:
                next(data_iter)
            except StopIteration:
                print(f"ERROR: Dataset exhausted after {i:,} docs (expected {resume_docs:,})")
                return
            if (i + 1) % 1_000_000 == 0:
                elapsed = time.time() - skip_start
                rate = (i + 1) / elapsed
                remaining = (resume_docs - i - 1) / rate
                print(f"  Skipped {i+1:,}/{resume_docs:,} docs "
                      f"({(i+1)/resume_docs*100:.1f}%, "
                      f"{rate:.0f} docs/s, ~{remaining/60:.0f}min remaining)")
        skip_elapsed = time.time() - skip_start
        print(f"  Skip complete: {resume_docs:,} docs in {skip_elapsed:.0f}s "
              f"({resume_docs/skip_elapsed:.0f} docs/s)")

    # ---- Tokenize and pack ----
    print("Starting tokenization and packing...")

    token_buffer = list(prev_tail_tokens)  # restore leftover from previous run
    total_tokens_written = resume_tokens
    total_seqs_written = resume_seqs
    docs_processed = resume_docs
    start_time = time.time()
    last_log_tokens = total_tokens_written

    flush_every = args.flush_every
    pending_seqs = []

    # Shard tracking (continue from where we left off)
    current_shard = resume_shards
    shard_seqs_written = 0
    shard_files = []
    shard_infos = list(prev_shard_infos)  # keep previous shard records

    # Open next output file
    if is_sharded:
        current_path = shard_path(output_base, current_shard)
    else:
        current_path = output_base + ".bin"
    f_out = open_new_output(current_path)

    def flush_pending():
        nonlocal pending_seqs
        if pending_seqs:
            chunk = np.stack(pending_seqs)
            f_out.write(chunk.tobytes())
            pending_seqs = []

    def close_shard():
        """Close current shard, record metadata."""
        nonlocal f_out, shard_seqs_written
        flush_pending()
        f_out.close()
        shard_files.append(current_path)
        shard_infos.append({
            "path": current_path,
            "sequences": shard_seqs_written,
            "tokens": shard_seqs_written * seq_len,
        })
        print(f"  Shard {current_shard} closed: {current_path} "
              f"({shard_seqs_written:,} seqs, {shard_seqs_written * seq_len:,} tokens)")
        shard_seqs_written = 0

    def open_next_shard():
        """Open next shard file."""
        nonlocal current_shard, f_out, current_path
        current_shard += 1
        current_path = shard_path(output_base, current_shard)
        f_out = open_new_output(current_path)

    try:
        for example in data_iter:
            text = example.get("text", "")
            if not text or len(text.strip()) == 0:
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue

            token_buffer.extend(token_ids)
            docs_processed += 1

            while len(token_buffer) >= seq_len:
                seq = token_buffer[:seq_len]
                token_buffer = token_buffer[seq_len:]

                pending_seqs.append(np.array(seq, dtype=np.uint16))
                total_seqs_written += 1
                total_tokens_written += seq_len
                shard_seqs_written += 1

                if len(pending_seqs) >= flush_every:
                    flush_pending()

                # Progress logging every 1B tokens
                if total_tokens_written - last_log_tokens >= 1_000_000_000:
                    elapsed = time.time() - start_time
                    new_tokens = total_tokens_written - resume_tokens
                    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0
                    progress = total_tokens_written / target_tokens * 100
                    shard_info = f" shard={current_shard}" if is_sharded else ""
                    remaining_tokens = target_tokens - total_tokens_written
                    eta = remaining_tokens / tok_per_sec if tok_per_sec > 0 else 0
                    print(
                        f"  [{progress:6.2f}%] {total_tokens_written/1e9:.1f}B tokens, "
                        f"{total_seqs_written:,} seqs, {docs_processed:,} docs, "
                        f"{tok_per_sec/1e6:.2f}M tok/s, "
                        f"elapsed={elapsed/3600:.1f}h, ETA={eta/3600:.1f}h{shard_info}"
                    )
                    last_log_tokens = total_tokens_written

                # Shard boundary check
                if is_sharded and shard_seqs_written >= seqs_per_shard and current_shard < num_shards - 1:
                    close_shard()
                    open_next_shard()

                if total_tokens_written >= target_tokens:
                    break

            if total_tokens_written >= target_tokens:
                break

        # Close final shard/file
        close_shard()

    except Exception:
        # Make sure we close on error too
        try:
            flush_pending()
            f_out.close()
        except Exception:
            pass
        raise

    elapsed = time.time() - start_time

    # ---- Write metadata ----
    metadata = {
        "total_tokens": total_tokens_written,
        "total_sequences": total_seqs_written,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "tokenizer": args.tokenizer,
        "dtype": "uint16",
        "split": args.split,
        "num_shards": len(shard_infos),
        "shards": shard_infos,
        "docs_consumed": docs_processed,
        "docs_processed": docs_processed,
        "tail_token_buffer": token_buffer[:seq_len],  # save leftover for resume
        "discarded_tail_tokens": len(token_buffer),
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(
            (total_tokens_written - resume_tokens) / elapsed, 1
        ) if elapsed > 0 else 0,
    }
    write_json(meta_path, metadata)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}")
    print(f"  Tokens written: {total_tokens_written:,}")
    if resume_tokens > 0:
        print(f"  New tokens this run: {total_tokens_written - resume_tokens:,}")
    print(f"  Sequences: {total_seqs_written:,}")
    print(f"  Docs consumed: {docs_processed:,}")
    print(f"  Discarded tail: {len(token_buffer)} tokens")
    print(f"  Shards: {len(shard_infos)}")
    for i, info in enumerate(shard_infos):
        marker = " (new)" if i >= resume_shards else ""
        print(f"    [{i:03d}] {info['path']} ({info['sequences']:,} seqs){marker}")
    print(f"  Metadata: {meta_path}")
    print(f"  Total size: ~{total_seqs_written * seq_len * 2 / 1e9:.2f} GB")
    new_tokens = total_tokens_written - resume_tokens
    print(f"  Elapsed: {elapsed:.0f}s ({new_tokens / elapsed / 1e6:.2f}M tok/s)")


if __name__ == "__main__":
    main()
