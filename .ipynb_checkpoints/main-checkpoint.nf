#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

process data_preprocess {
    input:
    tuple(
        val(name), 
        path(train), 
        path(validation), 
        path(test),
        path(dmr),
        val(target),
        val(background)
    )

    output:
    tuple(
        val(name),
        path('train_seq.csv'),
        path('val_seq.csv'),
        path('test_seq.csv'),
        path('workflow.log')
    )

    script:
    """
    python3 ${workflow.projectDir}/scripts/preprocess.py \\
      --train ${train} \\
      --val ${validation} \\
      --test ${test} \\
      --dmr ${dmr} \\
      --target ${target} \\
      --background ${background} &>> workflow.log
    """
}

process methylbert_finetune {
    input:
    tuple(
        val(name),
        path(train),
        path(validation),
        path(test),
        path(log)
    )

    output:
    tuple(
        val(name),
        path("${name}"),
        path(train),
        path(test)
    )

    script:
    """
    total_reads_count=\$(awk 'END {print NR}' ${train})
    evalfreq=\$((total_reads_count / ${params.batch_size}))
    step=\$((evalfreq * ${params.epoch}))

    mkdir -p ${name}/1.finetuned_model
    python3 ${workflow.projectDir}/scripts/finetune.py \\
      --input . \\
      --model ${params.pretrained_model} \\
      --output ${name}/1.finetuned_model \\
      --bs ${params.batch_size} \\
      --step \${step} \\
      --cores ${task.cpus} \\
      --logfreq 1 \\
      --savefreq \${evalfreq} \\
      --evalfreq \${evalfreq} &>> ${log}
    cp ${log} ${name}/
    """
}

process methylbert_deconvolution {
    input:
    tuple(
        val(name),
        path(output),
        path(train),
        path(test)
    )

    output:
    path("${name}")

    script:
    """
    mkdir -p ${name}/2.deconvolution
    python3 ${workflow.projectDir}/scripts/deconvolution.py \\
      --input ${test} \\
      --train ${train} \\
      --model ${name}/1.finetuned_model/bert.model \\
      --output ${name}/2.deconvolution \\
      --cores ${task.cpus} &>> ${name}/workflow.log
    """
}

//process plot_curves {
//}

workflow {
    // TODO Refine the workflow with a process for meta_param_parsing, add error handling
    Channel
        .fromPath(params.meta)
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

    data_preprocess(ch_meta_rows)
        | set { ch_preprocessed_data }

    methylbert_finetune(ch_preprocessed_data)
        | set { ch_finetune_output }

    methylbert_deconvolution(ch_finetune_output)
}