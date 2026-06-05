#!/usr/bin/env python3
"""Fine-tune preprocessing (Polars / streaming, no pandas iterrows / intervaltree).

Reads train/val/test BED reads + a DMR BED, then:

1. assigns each read its overlapping DMR (vectorised searchsorted, no per-row
   Python loops over a tree);
2. keeps only DMRs that are present in *all* provided splits and re-indexes the
   ``dmr_label`` sequentially (identical semantics to the previous pandas code);
3. converts each kept read to MethylBERT k-mer / methylation strings;
4. writes one parquet per split (``*_seq.parquet``) plus ``filtered_dmr.bed``
   (chr, start, end, dmr_label) so the infer pipeline can reuse the exact same
   DMR index.
"""

import argparse
import os

import numpy as np
import polars as pl

BED_COLS = ["chr", "start", "end", "seq", "name", "ctype"]


def read_bed(path: str) -> pl.DataFrame:
    """Stream a 6-column BED-like reads file with Polars."""
    return (
        pl.scan_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=BED_COLS,
            schema_overrides={"chr": pl.Utf8, "start": pl.Int64, "end": pl.Int64,
                              "seq": pl.Utf8, "name": pl.Utf8, "ctype": pl.Utf8},
        )
        .collect(streaming=True)
    )


def read_dmr(path: str) -> pl.DataFrame:
    dmr = pl.read_csv(
        path, separator="\t", has_header=False,
        columns=[0, 1, 2], new_columns=["chr", "start", "end"],
        schema_overrides={"chr": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
    )
    return dmr.with_row_index("dmr_label").select(
        ["chr", "start", "end", pl.col("dmr_label").cast(pl.Int64)]
    )


def build_dmr_index(dmr_df: pl.DataFrame):
    """Per-chromosome arrays sorted by start (DMRs are non-overlapping regions)."""
    index = {}
    for chrom, sub in dmr_df.partition_by("chr", as_dict=True).items():
        chrom = chrom[0] if isinstance(chrom, tuple) else chrom
        sub = sub.sort("start")
        index[chrom] = (
            sub["start"].to_numpy(),
            sub["end"].to_numpy(),
            sub["dmr_label"].to_numpy(),
        )
    return index


def assign_overlapping_label(reads_df: pl.DataFrame, dmr_index) -> np.ndarray:
    """Return the overlapping (original) dmr_label per read, or -1 if none."""
    n = reads_df.height
    out = np.full(n, -1, dtype=np.int64)
    chrom_np = reads_df["chr"].to_numpy()
    start_np = reads_df["start"].to_numpy()
    end_np = reads_df["end"].to_numpy()

    for chrom in np.unique(chrom_np):
        if chrom not in dmr_index:
            continue
        d_start, d_end, d_label = dmr_index[chrom]
        sel = np.nonzero(chrom_np == chrom)[0]
        rs = start_np[sel]
        re = end_np[sel]
        pos = np.searchsorted(d_start, rs, side="right") - 1
        assigned = np.full(sel.shape[0], -1, dtype=np.int64)
        # Check a small neighbourhood; DMRs are non-overlapping so a read can
        # only touch the interval starting at/just before it (and the next one).
        for c in (-1, 0, 1):
            idx = pos + c
            valid = (idx >= 0) & (idx < d_start.shape[0]) & (assigned < 0)
            if not valid.any():
                continue
            vi = np.nonzero(valid)[0]
            di = idx[vi]
            overlap = (d_start[di] < re[vi]) & (d_end[di] > rs[vi])
            hit = vi[overlap]
            assigned[hit] = d_label[di[overlap]]
        out[sel] = assigned
    return out


def seq_to_kmer(seq: str, k: int = 3):
    converted, methyl = [], []
    for i in range(len(seq) - k):
        token = seq[i:i + k]
        mid = token[1]
        m = 0 if mid == "C" else (1 if mid == "M" else 2)
        converted.append(token)
        methyl.append(str(m))
    return " ".join(converted), "".join(methyl)


def kmerize(seqs):
    dna, methyl = [], []
    for s in seqs:
        a, b = seq_to_kmer(s)
        dna.append(a)
        methyl.append(b)
    return dna, methyl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=None)
    p.add_argument("--val", default=None)
    p.add_argument("--test", default=None)
    p.add_argument("--dmr", required=True)
    p.add_argument("--target", default="T")
    p.add_argument("--background", default="N")
    p.add_argument("--outdir", default=".")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dmr_df = read_dmr(args.dmr)
    dmr_index = build_dmr_index(dmr_df)

    splits = {}
    for split, path in [("train", args.train), ("val", args.val), ("test", args.test)]:
        if path:
            splits[split] = read_bed(path)

    # Per-split overlapping labels, then DMRs common to ALL splits.
    split_labels = {}
    per_split_present = []
    for split, df in splits.items():
        lab = assign_overlapping_label(df, dmr_index)
        split_labels[split] = lab
        per_split_present.append(set(int(x) for x in np.unique(lab) if x >= 0))

    common = set.intersection(*per_split_present) if per_split_present else set()
    kept = dmr_df.filter(pl.col("dmr_label").is_in(sorted(common))).sort(["chr", "start"])
    remap = {old: new for new, old in enumerate(kept["dmr_label"].to_list())}
    print(f"DMR filtering: {dmr_df.height} -> {kept.height} "
          f"(dropped {dmr_df.height - kept.height} not present in all splits)")

    # Save the filtered DMR index (shared with the infer pipeline).
    filtered_path = os.path.join(args.outdir, "filtered_dmr.bed")
    (
        kept.with_columns(
            pl.col("dmr_label").replace_strict(remap, default=None).alias("dmr_label")
        )
        .select(["chr", "start", "end", "dmr_label"])
        .write_csv(filtered_path, separator="\t", include_header=False)
    )

    target, background = args.target, args.background
    for split, out_name in [("train", "train_seq.parquet"),
                            ("val", "val_seq.parquet"),
                            ("test", "test_seq.parquet")]:
        if split not in splits:
            continue
        df = splits[split]
        labels = split_labels[split]
        new_labels = np.array([remap.get(int(x), -1) for x in labels], dtype=np.int64)
        keep_mask = new_labels >= 0

        df = df.with_columns(pl.Series("dmr_label", new_labels)).filter(
            pl.Series(keep_mask)
        )
        print(f"{split}: {labels.shape[0]} -> {df.height} reads after DMR overlap filter")

        dna, methyl = kmerize(df["seq"].to_list())
        ctype_norm = (
            df["ctype"]
            .map_elements(
                lambda c: "T" if c == target else ("N" if c == background else c),
                return_dtype=pl.Utf8,
            )
        )
        out = pl.DataFrame({
            "name": df["name"],
            "dna_seq": dna,
            "methyl_seq": methyl,
            "dmr_label": df["dmr_label"],
            "ctype": ctype_norm,
            "dmr_ctype": pl.Series(["T"] * df.height),
        }).with_columns(
            (pl.col("ctype") == "T").cast(pl.Int8).alias("ctype_label")
        )
        out.write_parquet(os.path.join(args.outdir, out_name), compression="zstd")
        print(f"Wrote {out.height} reads -> {out_name}")


if __name__ == "__main__":
    main()
