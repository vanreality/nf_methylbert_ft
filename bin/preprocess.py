#!/usr/bin/env python3
import pandas as pd
from intervaltree import Interval, IntervalTree
import os
import argparse


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
    print(f"DMR filtering: {dmr_df.shape[0]} -> {filtered.shape[0]} (dropped {dmr_df.shape[0] - filtered.shape[0]} DMRs not present in all splits)")
    return filtered


def filter_sequences_by_dmr(data_df, dmr_df):
    """Keep only reads that overlap with at least one retained DMR."""
    dmr_trees = build_interval_tree(dmr_df)
    keep = []
    for idx, row in data_df.iterrows():
        chrom = row['chr']
        if chrom in dmr_trees and dmr_trees[chrom].overlap(row['start'], row['end']):
            keep.append(idx)
    filtered = data_df.loc[keep].reset_index(drop=True)
    print(f"Sequence filtering: {data_df.shape[0]} -> {filtered.shape[0]} reads")
    return filtered


def assign_dmr_label(data_df, dmr_df):
    dmr_tree = build_interval_tree(dmr_df)

    for idx, row in data_df.iterrows():
        if row['chr'] in dmr_tree:
            overlaps = dmr_tree[row['chr']].overlap(row['start'], row['end'])
            count = len(overlaps)
            if count == 1:
                data_df.at[idx, 'dmr_label'] = list(overlaps)[0].data
            elif count > 1:
                print(f'{row} overlaps with multiple dmrs')
                data_df.at[idx, 'dmr_label'] = list(overlaps)[0].data

    data_df['dmr_label'] = data_df['dmr_label'].astype(int)
    return data_df


def seq_to_kmer(seq, k=3):
    converted_seq = list()
    methyl_seq = list()
    for seq_idx in range(len(seq)-k):
        token = seq[seq_idx:seq_idx+k]
        if token[1] == 'C':
            m = 0
        elif token[1] == 'M':
            m = 1
        else:
            m = 2
            
        converted_seq.append(token)
        methyl_seq.append(str(m))

    return " ".join(converted_seq), "".join(methyl_seq)


def convert_to_methylbert_format(data_df, dmr_df, format_df, target_label, background_label):
    data_df = assign_dmr_label(data_df, dmr_df)
    data_df[['dna_seq', 'methyl_seq']] = data_df['seq'].apply(lambda x: pd.Series(seq_to_kmer(x, k=3)))
    data_df = data_df.rename(columns={"chr":"ref_name", 
                                      "start":"ref_pos", 
                                      "end":"length"})
    data_df['length'] = data_df['length'] - data_df['ref_pos']
    data_df['ctype'] = data_df['ctype'].replace({target_label: 'T', background_label: 'N'})
    data_df['dmr_ctype'] = 'T'

    # Ensure data_df has all columns from format_df
    missing_columns = set(format_df.columns) - set(data_df.columns)
    
    # Add missing columns to data_df with values filled as '='
    for col in missing_columns:
        data_df[col] = '='

    data_df = data_df[format_df.columns]
        
    return data_df


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Training data file path", default=None)
    parser.add_argument("--val", type=str, help="Validation data file path", default=None)
    parser.add_argument("--test", type=str, help="Test data file path", default=None)
    parser.add_argument("--dmr", type=str, help="DMR file path")
    parser.add_argument("--target", type=str, help="Target label", default="T")
    parser.add_argument("--background", type=str, help="Background label", default="N")
    args = parser.parse_args()

    return (args.train, args.val, args.test, args.dmr, args.target, args.background)


def main():
    # Parse input parameters
    (train_file, val_file, test_file, dmr_file, target_label, background_label) = parse_arguments()

    # Methylbert data format
    methylbert_header = [
        'name', 'flag', 'ref_name', 'ref_pos', 'map_quality',
        'cigar', 'next_ref_name', 'next_ref_pos', 'length', 'seq',
        'qual', 'MD', 'PG', 'XG', 'NM', 'XM', 'XR',
        'dna_seq', 'methyl_seq', 'dmr_ctype', 'dmr_label', 'ctype'
    ]
    methylbert_format = pd.DataFrame(columns=methylbert_header)
    
    # DMR
    dmr_df = pd.read_csv(dmr_file, sep='\t', usecols=[0, 1, 2], names=['chr', 'start', 'end'])
    dmr_df['dmr_label'] = range(dmr_df.shape[0])

    bed_cols = ['chr', 'start', 'end', 'seq', 'name', 'ctype']
    data = {}
    if train_file:
        data['train'] = pd.read_csv(train_file, sep='\t', names=bed_cols)
    if val_file:
        data['val'] = pd.read_csv(val_file, sep='\t', names=bed_cols)
    if test_file:
        data['test'] = pd.read_csv(test_file, sep='\t', names=bed_cols)

    dmr_df = filter_dmr_by_overlap(dmr_df, list(data.values()))

    for split, output_name in [('train', 'train_seq.csv'), ('val', 'val_seq.csv'), ('test', 'test_seq.csv')]:
        if split in data:
            data[split] = filter_sequences_by_dmr(data[split], dmr_df)
            converted = convert_to_methylbert_format(data[split], dmr_df, methylbert_format, target_label, background_label)
            converted.to_csv(output_name, sep='\t', header=True, index=None)


if __name__ == "__main__":
    main()

