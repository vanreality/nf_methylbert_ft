#!/usr/bin/bash
#SBATCH -J filter_dmr
#SBATCH --partition=cn-long
#SBATCH -o filter_dmr.log
#SBATCH -e filter_dmr.log
#SBATCH --cpus-per-task=8

singularity exec -B /lustre1,/lustre2,/appsnew /lustre1/cqyi/syfan/images/common_tools.sif \
python3 filter_dmr.py \
--train /lustre1/cqyi/syfan/nf_methylbert_ft/data/220k_data/train_reads_downsampled.bed \
--val /lustre1/cqyi/syfan/nf_methylbert_ft/data/220k_data/val_reads_downsampled.bed \
--test /lustre1/cqyi/syfan/nf_methylbert_ft/data/220k_data/test_reads_downsampled.bed \
--dmr /lustre1/cqyi/syfan/nf_methylbert_ft/data/220k_data/dmr.bed \
--output /lustre1/cqyi/syfan/nf_methylbert_ft/results/220k_data/data_preprocess/filtered_dmr.bed