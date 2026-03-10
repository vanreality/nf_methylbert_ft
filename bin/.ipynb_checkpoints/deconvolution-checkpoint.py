from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.data.vocab import MethylVocab
from methylbert.deconvolute import deconvolute
from methylbert.trainer import MethylBertFinetuneTrainer
from torch.utils.data import DataLoader

import pandas as pd

import argparse
import os
import re


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--train", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--cores", type=int, default=52)
    args = parser.parse_args()

    return args.input, args.train, args.model, args.output, args.cores


def stat_meth_token(s: str) -> int:
    # Count occurrences of "MG" and "CG"
    count_mg = s.count("MG")
    count_cg = s.count("CG")
    # Count occurrences of "M" not followed by "G" using regex
    count_m_not_g = len(re.findall(r'M(?!G)', s))
    # Total count
    total_count = count_mg + count_cg + count_m_not_g
    # Total length
    total_length = len(s)
    # Calculate ratio
    total_ratio = total_count / total_length if total_length > 0 else 0
    return total_count, total_ratio


def modify_res(res_file_path):
    res_df = pd.read_csv(res_file_path, sep='\t')
    res_df[['meth_count', 'meth_ratio']] = res_df['dna_seq'].apply(
        lambda x: pd.Series(stat_meth_token(x))
    )
    res_df.to_csv(res_file_path, sep='\t', index=False)


if __name__ == "__main__":
    # Parse user input
    (input_path, train, model_dir, out_dir, n_cores) = parse_arguments()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tokenizer = MethylVocab(k=3)
    dataset = MethylBertFinetuneDataset(input_path, tokenizer, seq_len=150)
    data_loader = DataLoader(dataset, batch_size=128, num_workers=n_cores)
    df_train = pd.read_csv(train, sep="\t")

    trainer = MethylBertFinetuneTrainer(
        len(tokenizer),
        train_dataloader=data_loader,
        test_dataloader=data_loader,
    )
    trainer.load(model_dir)

    deconvolute(
            trainer=trainer,
            tokenizer=tokenizer,
            data_loader=data_loader,
            output_path=out_dir,
            df_train=df_train,
            adjustment=True,
    )

    print("Deconvolution done!")

    modify_res(os.path.join(out_dir, 'res.csv'))

    