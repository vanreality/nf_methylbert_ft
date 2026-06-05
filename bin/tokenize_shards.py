#!/usr/bin/env python3
"""Tokenize preprocessed parquet into model-ready parquet shards (once).

Each input read is encoded a single time with the official MethylBERT k-mer
vocabulary into fixed-length ``input_ids`` / ``token_type_ids`` plus its labels.
Raw sequences are dropped so shards stay small, and downstream training/inference
read these shards directly instead of re-tokenizing.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import mb_lib  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Preprocessed *_seq.parquet")
    p.add_argument("--prefix", required=True, help="Shard file prefix (e.g. train)")
    p.add_argument("--outdir", default=".", help="Shard output directory")
    p.add_argument("--seq-len", type=int, default=mb_lib.DEFAULT_SEQ_LEN)
    p.add_argument("--shard-size", type=int, default=200_000)
    p.add_argument("--batch-rows", type=int, default=50_000)
    return p.parse_args()


def main():
    args = parse_args()
    vocab = mb_lib.get_vocab(3)
    n_shards, total = mb_lib.tokenize_parquet_to_shards(
        in_parquet=args.input,
        out_dir=args.outdir,
        prefix=args.prefix,
        vocab=vocab,
        seq_len=args.seq_len,
        shard_size=args.shard_size,
        batch_rows=args.batch_rows,
    )
    print(f"Tokenized {total} reads from {args.input} into {n_shards} shard(s) "
          f"under {args.outdir} (prefix={args.prefix}, seq_len={args.seq_len})")


if __name__ == "__main__":
    main()
