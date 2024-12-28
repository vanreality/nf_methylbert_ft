# MethylBERT read classifier fine-tuning pipeline

This Nextflow pipeline automates the process of training large language model using `MethylBERT`, which make use of DNA sequences and methylation information for read classification tasks. The pipeline processes multiple datasets in parallel, performs model training, and outputs the results along with log files for each training run.

The workflow is designed to be scalable, reproducible, and efficient, leveraging Nextflow’s ability to handle job scheduling and parallelization.

## Requirements
To use the scripts, ensure the following dependencies are installed:

- [Nextflow](https://www.nextflow.io/docs/latest/index.html)
- [Singularity](https://docs.sylabs.io/guides/4.2/user-guide/) (Optional – for containerized environments)
- [MethylBERT](https://github.com/CompEpigen/methylbert) (Optional – if not using Singularity)

Install Nextflow
```bash
curl -s https://get.nextflow.io | bash
```

## Pipeline Structure
```
.                                   # Project root directory
├── config                          # Directory for configuration files
│   └── meta.tsv                    # Metadata table, describing sample information and file paths
├── data                            # Data directory, containing pretrained models and test data
│   ├── pretrained_hg19_12l         # Pretrained model (hg19 version, 12 layers)
│   │   ├── config.json             # Model configuration file, specifying model parameters and architecture
│   │   └── pytorch_model.bin       # PyTorch model binary file, containing model weights
│   └── test_data                   # Test data directory
│       ├── data_1
│       │   ├── dmr.bed             # Differentially methylated region (DMR) file
│       │   ├── test.bed            # Test set
│       │   ├── train.bed           # Training set
│       │   └── val.bed             # Validation set
│       └── data_2
│           ├── dmr.bed             
│           ├── test.bed            
│           ├── train.bed           
│           └── val.bed             
├── images                          # Directory for container image files
│   ├── base.sif                    # Basic Singularity image file
│   └── methylbert_v2.0.1.sif       # Singularity image for MethylBERT v2.0.1
├── main.nf                         # Main Nextflow pipeline script, defining core logic of the analysis workflow
├── nextflow.config                 # Nextflow configuration file, specifying workflow parameters and resource allocation
├── README.md                       # Project documentation, describing background, installation, and usage
└── scripts                         # Directory for custom scripts
    ├── deconvolution.py            # Read-level deconvolution script
    ├── finetune.py                 # Fine-tuning script, for training pretrained models on read-classification task
    ├── plot.py                     # Plotting script, generating learning curves
    └── preprocess.py               # Data preprocessing script for format conversion
```

## Usage
### 1. Prepare Data Files
Ensure your raw methylation data is formatted according to the expected format. The format should contain **six columns** with the following headers:
```
chr    start    end    seq    tag    label
```
- **chr** – Chromosome  
- **start** – Start position  
- **end** – End position  
- **seq** – DNA sequence with or without base 'M'
- **tag** – Additional metadata or identifier  
- **label** – Class or label for the sequence  

Place the prepared data files in the appropriate directory for processing.

### 2. Prepare Metadata File
Edit the `config/meta.tsv` file to include the datasets and parameters required for each run.

### 3. Execute the Pipeline
```bash
nextflow run main.nf -profile singularity -bg
```

## Cleaning Up Temporary Files
After execution, clean up intermediate files to free disk space:
```bash
nextflow clean -f
```

## Notes
- Ensure that Nextflow and Singularity are properly installed and configured before running the pipeline.
- Review the logs in the output directory for troubleshooting and performance evaluation.
