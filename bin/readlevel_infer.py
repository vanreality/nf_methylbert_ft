#!/usr/bin/env python3
"""Read-level probability inference (no tumour deconvolution).

Loads a fine-tuned MethylBERT model and runs a forward pass over tokenized
parquet shards to emit a per-read classification probability table. This
deliberately replaces the default sample-level deconvolution output: we only
keep ``P(class)`` for each read.

If the shards carry ground-truth ``ctype_label`` (e.g. the fine-tune test set),
a metric summary (accuracy / balanced accuracy / AUC / PR-AUC) is also written.
"""

import argparse
import os
import sys

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import mb_lib  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True, help="tokenized shard dir/glob")
    p.add_argument("--model", required=True, help="fine-tuned model directory")
    p.add_argument("--dmr", default=None, help="filtered_dmr.bed to set n_dmrs")
    p.add_argument("--output", required=True, help="output parquet path")
    p.add_argument("--metrics", default=None, help="optional metric summary csv")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seq-len", type=int, default=mb_lib.DEFAULT_SEQ_LEN)
    p.add_argument("--loss", default="bce", choices=["bce", "focal_bce"])
    p.add_argument("--cache-shards", type=int, default=8)
    return p.parse_args()


def n_dmrs_from_bed(path):
    mx = -1
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                mx = max(mx, int(line.split("\t")[3]))
    return mx + 1


EMPTY_SCHEMA = {
    "name": pl.Series([], dtype=pl.Utf8),
    "dmr_label": pl.Series([], dtype=pl.Int64),
    "prob_class_0": pl.Series([], dtype=pl.Float32),
    "prob_class_1": pl.Series([], dtype=pl.Float32),
    "pred": pl.Series([], dtype=pl.Int64),
}


def write_empty(output):
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    pl.DataFrame(EMPTY_SCHEMA).write_parquet(output, compression="zstd")
    print(f"No reads to score; wrote empty predictions -> {output}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # No tokenized shards (e.g. zero reads overlapped the DMR index): emit empty.
    if not mb_lib.list_shards(args.shards):
        write_empty(args.output)
        if args.metrics:
            pl.DataFrame({"n": [0]}).write_csv(args.metrics)
        return

    n_dmrs = n_dmrs_from_bed(args.dmr) if args.dmr else None
    model = mb_lib.load_finetuned_model(args.model, seq_len=args.seq_len,
                                        loss=args.loss, n_dmrs=n_dmrs)
    model.to(device)

    dataset = mb_lib.ShardDataset(args.shards, seq_len=args.seq_len,
                                  cache_shards=args.cache_shards)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    res = mb_lib.run_read_level_inference(
        model, loader, device, precision=args.precision,
        with_labels=True, progress=True)

    out = {
        "name": res["name"],
        "dmr_label": res["dmr_label"],
        "prob_class_0": res["prob_class_0"],
        "prob_class_1": res["prob_class_1"],
        "pred": res["pred"],
    }
    if "ctype_label" in res:
        out["ctype_label"] = res["ctype_label"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    pl.DataFrame(out).write_parquet(args.output, compression="zstd")
    print(f"Wrote {len(res['name'])} read predictions -> {args.output}")

    if args.metrics and "ctype_label" in res:
        m = mb_lib.classification_metrics(res["ctype_label"], res["prob_class_1"],
                                          res["pred"])
        pl.DataFrame({k: [v] for k, v in m.items()}).write_csv(args.metrics)
        print(f"Test metrics -> {args.metrics}: {m}")


if __name__ == "__main__":
    main()
