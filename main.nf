#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
 * MethylBERT read-classifier fine-tuning pipeline (v2.0.2)
 *
 *   PREPROCESS  ->  TOKENIZE_SHARDS  ->  FINETUNE (DDP/torchrun)  ->  READLEVEL_INFER
 *
 * Streaming Polars preprocessing emits a shared filtered DMR index; tokenization
 * happens exactly once into parquet shards; training uses real multi-GPU DDP;
 * the test set gets read-level probability inference (no deconvolution).
 */

process PREPROCESS {
    tag "${name}"
    publishDir "${params.outdir}/${name}", mode: 'copy', pattern: 'filtered_dmr.bed'

    input:
    tuple val(name), path(train), path(validation), path(test), path(dmr),
          val(target), val(background)

    output:
    tuple val(name),
          path('train_seq.parquet'),
          path('val_seq.parquet'),
          path('test_seq.parquet'),
          path('filtered_dmr.bed')

    script:
    """
    preprocess.py \\
        --train ${train} \\
        --val ${validation} \\
        --test ${test} \\
        --dmr ${dmr} \\
        --target ${target} \\
        --background ${background} \\
        --outdir .
    """
}

process TOKENIZE_SHARDS {
    tag "${name}"

    input:
    tuple val(name), path(train_seq), path(val_seq), path(test_seq), path(dmr)

    output:
    tuple val(name),
          path('train_shards'),
          path('val_shards'),
          path('test_shards'),
          path(dmr)

    script:
    """
    tokenize_shards.py --input ${train_seq} --prefix train --outdir train_shards \\
        --shard-size ${params.shard_size}
    tokenize_shards.py --input ${val_seq}   --prefix val   --outdir val_shards \\
        --shard-size ${params.shard_size}
    tokenize_shards.py --input ${test_seq}  --prefix test  --outdir test_shards \\
        --shard-size ${params.shard_size}
    """
}

process FINETUNE {
    tag "${name}"
    publishDir "${params.outdir}/finetuned_model", mode: 'copy'

    input:
    tuple val(name), path(train_shards), path(val_shards), path(test_shards), path(dmr)

    output:
    tuple val(name),
          path('bert.model'),
          path('test_shards'),
          path(dmr),                          emit: model
    path('metrics_per_checkpoint.csv')
    path('train_log.csv')
    path('training_curves.png'),              optional: true
    path('bert.model_step*'),                 optional: true

    script:
    """
    export OMP_NUM_THREADS=${Math.max(1, (task.cpus as int).intdiv(params.ft_gpus))}
    export TOKENIZERS_PARALLELISM=false

    python3 -m torch.distributed.run \\
        --standalone --nnodes=1 --nproc_per_node=${params.ft_gpus} \\
        \$(command -v finetune_ddp.py) \\
        --train-shards train_shards \\
        --val-shards val_shards \\
        --pretrained ${params.pretrained_model} \\
        --dmr ${dmr} \\
        --output . \\
        --batch-size ${params.batch_size_per_gpu} \\
        --epochs ${params.epoch} \\
        --lr ${params.lr} \\
        --warmup-ratio ${params.warmup_ratio} \\
        --decay-start-ratio ${params.decay_start_ratio} \\
        --precision ${params.precision} \\
        --gradient-accumulation-steps ${params.gradient_accumulation_steps} \\
        --num-workers ${params.dataloader_workers} \\
        --val-every-steps ${params.val_every_steps} \\
        --best-metric ${params.best_metric} \\
        --seed ${params.seed}
    """
}

process READLEVEL_INFER {
    tag "${name}"
    publishDir "${params.outdir}/${name}/2.read_level_inference", mode: 'copy'

    input:
    tuple val(name), path(bert_model), path(test_shards), path(dmr)

    output:
    tuple val(name), path('test_predictions.parquet'), path('test_metrics.csv')

    script:
    """
    readlevel_infer.py \\
        --shards test_shards \\
        --model ${bert_model} \\
        --dmr ${dmr} \\
        --output test_predictions.parquet \\
        --metrics test_metrics.csv \\
        --batch-size ${params.infer_batch_size} \\
        --precision ${params.precision} \\
        --num-workers ${params.dataloader_workers}
    """
}

workflow {
    Channel
        .fromPath(params.samplesheet)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            tuple(row.name,
                  file(row.train),
                  file(row.validation),
                  file(row.test),
                  file(row.dmr),
                  row.target,
                  row.background)
        }
        .set { ch_meta_rows }

    PREPROCESS(ch_meta_rows)
    TOKENIZE_SHARDS(PREPROCESS.out)
    FINETUNE(TOKENIZE_SHARDS.out)
    READLEVEL_INFER(FINETUNE.out.model)
}
