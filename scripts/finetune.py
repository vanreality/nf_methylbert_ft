from methylbert.data import finetune_data_generate as fdg
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.data.vocab import MethylVocab
from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.deconvolute import deconvolute

import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

import os
import re
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--step", type=int, default=4000)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--cores", type=int, default=64)
    parser.add_argument("--savefreq", type=int, default=500)
    parser.add_argument("--logfreq", type=int, default=10)
    parser.add_argument("--evalfreq", type=int, default=100)
    parser.add_argument("--logpath", type=str)
    args = parser.parse_args()

    return (args.input, args.model, args.output, args.step, args.bs, args.cores, args.savefreq, args.logfreq, args.evalfreq, args.logpath)


def load_data(train_dataset: str, test_dataset: str, batch_size: int, num_workers: int):
    tokenizer = MethylVocab(k=3)

    # Load data sets
    train_dataset = MethylBertFinetuneDataset(train_dataset, tokenizer, seq_len=150)
    test_dataset = MethylBertFinetuneDataset(test_dataset, tokenizer, seq_len=150)

    # Create a data loader
    print("Creating Dataloader")

    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size,
                                   num_workers=num_workers, 
                                   pin_memory=False, 
                                   shuffle=True)

    test_data_loader = DataLoader(test_dataset, 
                                  batch_size=batch_size,
                                  num_workers=num_workers, 
                                  pin_memory=True,
                                  shuffle=False) if test_dataset is not None else None

    return tokenizer, train_data_loader, test_data_loader


def finetune(tokenizer: MethylVocab,
             save_path: str,
             train_data_loader: DataLoader,
             test_data_loader: DataLoader,
             pretrain_model: str,
             steps: int, 
             save_freq: int, 
             log_freq: int, 
             eval_freq: int):
    trainer = MethylBertFinetuneTrainer(vocab_size=len(tokenizer),
                                        save_path=save_path + "/bert.model/",
                                        train_dataloader=train_data_loader,
                                        test_dataloader=test_data_loader,
                                        with_cuda=True,
                                        lr=1e-5,
                                        max_grad_norm=1.0,
                                        gradient_accumulation_steps=1,
                                        warmup_step=100,
                                        decrease_steps=200,
                                        save_freq=save_freq,
                                        log_freq=log_freq,
                                        eval_freq=eval_freq,
                                        beta=(0.9,0.98),
                                        weight_decay=0.1)
    trainer.load(pretrain_model)
    trainer.train(steps)

    print("Finetuning done!")


def main():
    # Parse input parameters
    (input_dir, pre_model, out_dir, n_steps, batch_size, 
     n_cores, savefreq, logfreq, evalfreq, log_path) = parse_arguments()

    # Create output directories
    out_finetune = os.path.join(out_dir, "1.finetune")
    out_plot = os.path.join(out_dir, "2.plot")
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    if not os.path.exists(out_finetune): os.mkdir(out_finetune)
    if not os.path.exists(out_plot): os.mkdir(out_plot)

    # Prepare data loader
    tokenizer, train_data_loader, test_data_loader = \
        load_data(train_dataset=os.path.join(input_dir, "train_seq.csv"),
                  test_dataset=os.path.join(input_dir, "val_seq.csv"),
                  batch_size=batch_size,
                  num_workers=n_cores)

    # Finetune the model
    finetune(tokenizer, 
             out_finetune, 
             train_data_loader, 
             test_data_loader, 
             pre_model, 
             n_steps, 
             savefreq, 
             logfreq, 
             evalfreq)


if __name__ == "__main__":
    main()
