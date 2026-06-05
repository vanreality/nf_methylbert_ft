# MethylBERT read-classifier fine-tuning pipeline (v2.0.2)

A Nextflow pipeline that fine-tunes [MethylBERT](https://github.com/CompEpigen/methylbert)
for **read-level methylation classification** using real multi-GPU training.

```
PREPROCESS  ->  TOKENIZE_SHARDS  ->  FINETUNE (DDP / torchrun)  ->  READLEVEL_INFER
```

## What changed vs. the previous version

| Area | Before | Now |
| --- | --- | --- |
| Container | `methylbert_v2.0.1.sif` | `methylbert_v2.0.2.sif` |
| Multi-GPU | requested 4 GPUs but used `nn.DataParallel` (effectively 1 process) | true **DDP via `torchrun`**, one process per GPU |
| Preprocessing | `pandas` + `intervaltree`, `iterrows`, whole-file loads | **Polars / streaming**, vectorised DMR overlap |
| Tokenization | re-done on every dataset construction | done **once** into parquet shards (`TOKENIZE_SHARDS`) |
| Dataset | whole file read into a list of dicts | **streaming `ShardDataset`** (bounded LRU shard cache) |
| Validation | every step | **once per epoch** (or every `--val_every_steps`) |
| Metrics | loss + accuracy | + **AUC / PR-AUC / balanced accuracy**, per-checkpoint table, best-ckpt selection |
| Final stage | tumour deconvolution | **read-level probability inference** on the test set |
| DMR index | not persisted | `filtered_dmr.bed` is published and reused by the infer pipeline |

DDP training supports mixed precision (`--precision fp16|bf16|fp32`), gradient
accumulation, a fixed seed, and resume from checkpoint. Checkpoints, the best
model, metrics and plots are written on rank 0 only; the training loss is reduced
and validation metrics are gathered across all ranks.

## Repository layout

```
.
├── assets/                         # example samplesheets
├── data/
│   └── pretrained_hg19_12l/        # pretrained MethylBERT MLM (config.json + pytorch_model.bin)
├── images/methylbert_v2.0.2.sif    # container
├── bin/
│   ├── mb_lib.py                   # shared: tokenizer, ShardWriter/ShardDataset, model build/save/load, metrics
│   ├── preprocess.py               # Polars/streaming preprocessing -> *_seq.parquet + filtered_dmr.bed
│   ├── tokenize_shards.py          # tokenize once -> parquet shards
│   ├── finetune_ddp.py             # DDP/torchrun fine-tuning
│   └── readlevel_infer.py          # read-level probability inference + metrics
├── main.nf
└── nextflow.config
```

## Samplesheet

Tab-separated, one fine-tune run per row:

```
name    train    validation    test    dmr    target    background
```

`train` / `validation` / `test` are 6-column BED reads files
(`chr  start  end  seq  name  ctype`); `dmr` is a BED file of candidate regions;
`target` / `background` are the `ctype` values mapped to the positive / negative
class.

## Usage

```bash
nextflow run main.nf -profile singularity \
    --samplesheet data/220k_data/samplesheet_downsampled.tsv \
    --outdir test
```

### Key parameters (`nextflow.config`)

| Param | Default | Meaning |
| --- | --- | --- |
| `epoch` | 10 | training epochs |
| `batch_size_per_gpu` | 64 | per-GPU micro batch size |
| `gradient_accumulation_steps` | 1 | gradient accumulation |
| `precision` | `bf16` | `fp16` / `bf16` / `fp32` |
| `val_every_steps` | 0 | 0 = validate once per epoch |
| `best_metric` | `auc` | checkpoint selection metric |
| `shard_size` | 200000 | reads per tokenized shard |
| `ft_gpus` | 4 | GPUs per fine-tune job (`--gres=gpu:N`, `nproc_per_node=N`) |

## Outputs (`<outdir>/<name>/`)

```
filtered_dmr.bed                       # shared DMR index for the infer pipeline
1.finetuned_model/
    bert.model/                        # best model (config.json, model.safetensors, *.pickle)
    bert.model_step*/                  # per-validation checkpoints
    metrics_per_checkpoint.csv         # AUC / PR-AUC / balanced acc / loss per checkpoint (+ is_best)
    train_log.csv
    training_curves.png
2.read_level_inference/
    test_predictions.parquet           # per-read probabilities on the test set
    test_metrics.csv                   # AUC / PR-AUC / balanced acc / accuracy
```
