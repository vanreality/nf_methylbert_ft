#!/usr/bin/env python3
import argparse

import pandas as pd
from intervaltree import Interval, IntervalTree


def build_interval_tree(dmr_df):
    trees = {}
    for chrom, group in dmr_df.groupby('chr'):
        tree = IntervalTree()
        for _, row in group.iterrows():
            tree[row['start']:row['end']] = row['dmr_label']
        trees[chrom] = tree
    return trees


def get_overlapping_dmr_labels(dmr_trees, data_df):
    """Return the set of dmr_labels that overlap with at least one read in data_df."""
    labels = set()
    for _, row in data_df.iterrows():
        chrom = row['chr']
        if chrom in dmr_trees:
            for ov in dmr_trees[chrom].overlap(row['start'], row['end']):
                labels.add(ov.data)
    return labels


def filter_dmr_by_overlap(dmr_df, data_dfs):
    """Keep only DMRs that overlap with reads in ALL splits, then reassign sequential dmr_label."""
    dmr_trees = build_interval_tree(dmr_df)

    per_split_labels = [get_overlapping_dmr_labels(dmr_trees, df) for df in data_dfs]
    common_labels = set.intersection(*per_split_labels)

    filtered = dmr_df[dmr_df['dmr_label'].isin(common_labels)].reset_index(drop=True)
    filtered['dmr_label'] = range(filtered.shape[0])
    print(
        f"DMR filtering: {dmr_df.shape[0]} -> {filtered.shape[0]} "
        f"(dropped {dmr_df.shape[0] - filtered.shape[0]} DMRs not present in all splits)"
    )
    return filtered


def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter DMRs by overlap with reads in all data splits.")
    parser.add_argument("--train", type=str, help="Training data file path", default=None)
    parser.add_argument("--val", type=str, help="Validation data file path", default=None)
    parser.add_argument("--test", type=str, help="Test data file path", default=None)
    parser.add_argument("--dmr", type=str, required=True, help="DMR file path")
    parser.add_argument("--output", type=str, required=True, help="Output filtered DMR file path")
    return parser.parse_args()


def main():
    args = parse_arguments()

    dmr_df = pd.read_csv(args.dmr, sep='\t', usecols=[0, 1, 2], names=['chr', 'start', 'end'])
    dmr_df['dmr_label'] = range(dmr_df.shape[0])

    bed_cols = ['chr', 'start', 'end', 'seq', 'name', 'ctype']
    data_dfs = []
    for path in (args.train, args.val, args.test):
        if path:
            data_dfs.append(pd.read_csv(path, sep='\t', names=bed_cols))

    if not data_dfs:
        raise ValueError("At least one of --train, --val, or --test must be provided.")

    filtered_dmr = filter_dmr_by_overlap(dmr_df, data_dfs)
    filtered_dmr.to_csv(args.output, sep='\t', header=True, index=False)


if __name__ == "__main__":
    main()
