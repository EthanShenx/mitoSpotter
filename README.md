# An easy *mitoSpotter* running pipeline for markers <img src='./webui/static/logo.png' align="right" alt="" width="120" />

Welcome to the *mitoSpotter* marker analysis pipeline! This notebook provides a step-by-step guide to identifying mitochondrial and nuclear gene markers using Hidden Markov Models (HMMs). Whether you are new to bioinformatics or an experienced researcher, this pipeline is designed to be accessible, reproducible, and efficient.

All the data, scripts, and output from scripts are already in-place in our well-structured directory. Detailed file connection illustration is as follows.

![All scripts documentation](./webui/static/All_scripts_documentation.jpg)

All the preprocessed dataset (e.g., output from 01-03 scripts have been deposited in out_dir directory, but if you want to run this part, you can still do it!).

NOTE: This .md provides the exact same workflow and instrcution as the one in `easyMarking.ipynb`, but in a pure command-line format. So if you wanna replace the code and run directly, we highly recommand you to follow the jupyter notebook version.

## At a glimpse: Overall script structure

Below is the directory structure of the `scripts/` folder, which contains all Python scripts used in this pipeline. The flow is sequential and each script performs a specific task. The output of one script serves as the input for the next.

/mitoSpotter/scripts  
├── 01_from_gtf_extract_id.py  
├── 02_fasta_split_by_id.py  
├── 03_sequence2unit_nt.py  
├── 04_train_hmm_nt.py  
└── 05_decode_path_nt.py  

## Before we start

Before executing any cells in this notebook, please ensure the following:

1. **Environment Setup**:  
   You should have configured the conda or virtual environment as described in the documentation. Also, ensure that the correct kernel is selected in your Jupyter environment.

2. **Download Required Data**:  
   The pipeline depends on two essential genomic files:
   - **CDS genome file**: [Homo_sapiens.GRCh38.cds.all.fa.gz](https://42basepairs.com/browse/web/ensembl/release-82/fasta/homo_sapiens/cds?file=Homo_sapiens.GRCh38.cds.all.fa.gz&preview=)
   - **GTF annotation file**: [Homo_sapiens.GRCh38.115.chr.gtf.gz](https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.chr.gtf.gz)  
   Download and place them in the `/mitoSpotter/data` directory. *Note: This download may take several minutes depending on your connection speed.*

## STEP 1: EXTRACT NUCLEAR AND MITOCHONDRIA GENE ID

**Input**: GTF file  
**Output**: Two text files containing mitochondrial and nuclear gene IDs.  
**Key parameter**: `--protein_coding_only` ensures we only retain protein-coding genes, which are relevant for marker identification.

```bash
python scripts/01_from_gtf_extract_id.py \
    --gtf data/Homo_sapiens.GRCh38.115.chr.gtf \
    --outdir out_dir/01_ids \
    --prefix human_protein_coding_marker_testing_ \
    --protein_coding_only
```

### Expected Output

![01 output](./webui/static/01_output.png)

## 02 SPLIT FASTA BY ID

In this step, we want to get the sequence with these ids.

**Input**: A complete *5'UTR + CDS + 3' UTR* genome sequence file (FASTA format, from the previous step)

**Output**: Two FASTA files containing mitochondrial and nuclear gene sequences

**Key parameters**:

--mito_ids: Path to file containing mitochondrial gene IDs

--nuc_ids: Path to file containing nuclear gene IDs

--prefix: Custom prefix for output filenames

```bash
python scripts/02_fasta_split_by_id.py \
    --fasta data/Homo_sapiens.GRCh38.cds.all.fa \
    --mito_ids out_dir/01_ids/human_protein_coding_marker_testing_ids_mito.txt \
    --nuc_ids out_dir/01_ids/human_protein_coding_marker_testing_ids_nuclear.txt \
    --outdir out_dir/02_split_fasta \
    --prefix human_marker_testing_
```

### Expected Output

![02 output](./webui/static/02_output.png)

## 03 TOKENIZATION AND SPLIT INTO TRAINING AND HOLDOUT DATASETS

**Input**: Separated FASTA files for mitochondrial and nuclear genes (from the previous step)

**Output**:

Training dataset files (TSV format)

Holdout dataset files (TSV format)

**Key parameters**:

--mode: Tokenization mode (1nt, 2nt, or 3nt)

--train_frac: Fraction of data to use for training (default: 0.7)

Multiple mode combinations are processed automatically for comparative analysis

```bash
# Nuclear CDS with 3nt mode
python scripts/03_sequence2unit_nt.py \
    --fasta out_dir/02_split_fasta/human_nuclear_cds.fa \
    --mode 3nt \
    --train_tsv out_dir/03_unit/train/human_marker_testing_nuclear_3nt_train.tsv \
    --holdout_tsv out_dir/03_unit/holdout/human_marker_testing_nuclear_3nt_holdout.tsv \
    --train_frac 0.7

# Nuclear CDS with 2nt mode
python scripts/03_sequence2unit_nt.py \
    --fasta out_dir/02_split_fasta/human_nuclear_cds.fa \
    --mode 2nt \
    --train_tsv out_dir/03_unit/train/human_marker_testing_nuclear_2nt_train.tsv \
    --holdout_tsv out_dir/03_unit/holdout/human_marker_testing_nuclear_2nt_holdout.tsv \
    --train_frac 0.7

# Nuclear CDS with 1nt mode
python scripts/03_sequence2unit_nt.py \
    --fasta out_dir/02_split_fasta/human_nuclear_cds.fa \
    --mode 1nt \
    --train_tsv out_dir/03_unit/train/human_marker_testing_nuclear_1nt_train.tsv \
    --holdout_tsv out_dir/03_unit/holdout/human_marker_testing_nuclear_1nt_holdout.tsv \
    --train_frac 0.7

# Mito CDS with 3nt mode
python scripts/03_sequence2unit_nt.py \
    --fasta out_dir/02_split_fasta/human_mito_cds.fa \
    --mode 3nt \
    --train_tsv out_dir/03_unit/train/human_marker_testing_mito_3nt_train.tsv \
    --holdout_tsv out_dir/03_unit/holdout/human_marker_testing_mito_3nt_holdout.tsv \
    --train_frac 0.7

# Mito CDS with 2nt mode
python scripts/03_sequence2unit_nt.py \
    --fasta out_dir/02_split_fasta/human_mito_cds.fa \
    --mode 2nt \
    --train_tsv out_dir/03_unit/train/human_marker_testing_mito_2nt_train.tsv \
    --holdout_tsv out_dir/03_unit/holdout/human_marker_testing_mito_2nt_holdout.tsv \
    --train_frac 0.7

# Mito CDS with 1nt mode
python scripts/03_sequence2unit_nt.py \
    --fasta out_dir/02_split_fasta/human_mito_cds.fa \
    --mode 1nt \
    --train_tsv out_dir/03_unit/train/human_marker_testing_mito_1nt_train.tsv \
    --holdout_tsv out_dir/03_unit/holdout/human_marker_testing_mito_1nt_holdout.tsv \
    --train_frac 0.7
```

### Expected Output

![03 output](./webui/static/03_output.png)

## 04 HMM TRAINING

Yeahh! Finally comes into training phase! Let me take 3-nt level for example.

![04 param](./webui/static/04_param.png)

### EM run

```bash
python scripts/04_train_hmm_nt.py \
    --nuclear_nt_tsv out_dir/03_unit/train/human_marker_testing_nuclear_3nt_train.tsv \
    --mito_nt_tsv out_dir/03_unit/train/human_marker_testing_mito_3nt_train.tsv \
    --ngram 3 \
    --train_method em \
    --learn et \
    --n_em_iter 20 \
    --out_model_json out_dir/04_model/human_marker_testing_3nt_model_em.json \
    --out_vocab_json out_dir/04_model/human_marker_testing_3nt_vocab_em.json \
    --out_states_json out_dir/04_model/human_marker_testing_3nt_states_em.json \
    --sample 0.0001 \
    --track_memory \
    --n_workers 2
```

> **Note**: The `--n_workers 2` option is recommended for Mac or Linux. Remove this line if you are using Windows.

> **Note**: `--sample 0.0001` is used for downsampling for faster training, expected to take less than 1 minute.

### Viterbi

```bash
python scripts/04_train_hmm_nt.py \
    --nuclear_nt_tsv out_dir/03_unit/train/human_marker_testing_nuclear_3nt_train.tsv \
    --mito_nt_tsv out_dir/03_unit/train/human_marker_testing_mito_3nt_train.tsv \
    --ngram 3 \
    --train_method viterbi \
    --n_viterbi_iter 20 \
    --out_model_json out_dir/04_model/human_marker_testing_3nt_model_viterbi.json \
    --out_vocab_json out_dir/04_model/human_marker_testing_3nt_vocab_viterbi.json \
    --out_states_json out_dir/04_model/human_marker_testing_3nt_states_viterbi.json \
    --sample 0.0001 \
    --track_memory \
    --n_workers 2
```

> **Note**: The `--n_workers 2` option is recommended for Mac or Linux. Remove this line if you are using Windows.

> **Note**: `--sample 0.0001` is used for downsampling for faster training, expected to take less than 1 minute.

### Hybrid

```bash
# Let's make em 0.5 and viterbi 0.5 (1:1)
python scripts/04_train_hmm_nt.py \
    --nuclear_nt_tsv out_dir/03_unit/train/human_marker_testing_nuclear_3nt_train.tsv \
    --mito_nt_tsv out_dir/03_unit/train/human_marker_testing_mito_3nt_train.tsv \
    --ngram 3 \
    --train_method hybrid \
    --n_em_iter 10 \
    --n_viterbi_iter 10 \
    --learn et \
    --out_model_json out_dir/04_model/human_marker_testing_3nt_model_hybird.json \
    --out_vocab_json out_dir/04_model/human_marker_testing_3nt_vocab_hybird.json \
    --out_states_json out_dir/04_model/human_marker_testing_3nt_states_hybird.json \
    --sample 0.0001 \
    --track_memory \
    --n_workers 2
```

> **Note**: The `--n_workers 2` option is recommended for Mac or Linux. Remove this line if you are using Windows.

> **Note**: `--sample 0.0001` is used for downsampling for faster training, expected to take less than 1 minute.

> **Note**: Use `--learn et` to learn both emission and transition - this is essential!

## 05 SEQUENCE DECODING

![05 param](./webui/static/05_param.png)

### FASTA

```bash
python scripts/05_decode_path_nt.py \
    --model_json out_dir/04_model/human_marker_testing_3nt_model_em.json \
    --vocab_json out_dir/04_model/human_marker_testing_3nt_vocab_em.json \
    --states_json out_dir/04_model/human_marker_testing_3nt_states_em.json \
    --ngram 3 \
    --fasta ./webui/static/Rickettsia_prowazekii_str_Madrid_E.fa \
    --min_len 10 \
    --out_tsv out_dir/05_res/human_marker_testing_fasta_res.tsv \
    --plotting \
    --track_memory
```

> **Note**: The example FASTA file is from a bacteria (*Rickettsia prowazekii*) which has a relatively small phylogenetic distance to mitochondria genes.

> **Note**: `--min_len 10` ensures we don't decode sequences shorter than 10 nucleotides.

> **Note**: `--plotting` generates result plots which will be saved in the same directory where you run the script.

### ARGUMENT SEQUENCE PASSING

```bash
python scripts/05_decode_path_nt.py \
    --model_json out_dir/04_model/human_marker_testing_3nt_model_em.json \
    --vocab_json out_dir/04_model/human_marker_testing_3nt_vocab_em.json \
    --states_json out_dir/04_model/human_marker_testing_3nt_states_em.json \
    --ngram 3 \
    --min_len 10 \
    --seq "ATACCCATGGCCAACCTCCTACTCCTCATTGTACCCATTCTAATCGCAATGGCATTCCTAATGCTTACCGAACGAAAAATTCTAGGCTATATACAACTACGCAAAGGCCCCAACGTTGTAGGCCCCTACGGGCTACTACAACCCTTCGCTGACGCCATAAAACTCTTCACCAAAGAGCCCCTAAAACCCGCCACATCTACCATCACCCTCTACATCACCGCCCCGACCTTAGCTCTCACCATCGCTCTTCTACTATGAACCCCCCTCCCCATACCCAACCCCCTGGTCAACCTCAACCTAGGCCTCCTATTTATTCTAGCCACCTCTAGCCTAGCCGTTTACTCAATCCTCTGATCAGGGTGAGCATCAAACTCAAACTACGCCCTGATCGGCGCACTGCGAGCAGTAGCCCAAACAATCTCATATGAAGTCA" \
    --seq_id "marker_test_1" \
    --seq "GTCACCCTAGCCATCATTCTACTATCAACATTACTAATAAGTGGCTCCTTTAACCTCTCCACCCTTATCACAACACAAGAACACCTCTGATTACTCCTGCCATCATGACCCTTGGCCATAATATGATTTATCTCCACACTAGCAGAGACCAACCGAACCCCCTTCGACCTTGCCGAAGGGGAGTCCGAACTAGTCTCAGGCTTCAACATCGAATACGCCGCAGGCCCCTTCGCCCTATTCTTCATAGCCGAATACACAAACATTATTATAATAAACACCCTCACCACTACAATCTTCCTAGGAACAACATATGACGCACTCTCCCCTGAACTCTACACAACATATTTTGTCACCAAGACCCTACTTCTAACCTCCCTGTTCTTATGAATTCGAACAGCATACCCCCGATTCCGCTACGACCAACTCATACACCTCCTATGAAAAAACTTCCTACCACTCACCCTAGCATTACTTATATGATATGTCTCCATACCCATTACAATCTCCAGCATTCCCCCTCAAACC" \
    --seq_id "marker_test_2" \
    --out_tsv out_dir/05_res/human_marker_testing_arg_res.tsv \
    --plotting \
    --track_memory
```

> **Note**: You can pass multiple sequences directly via command line using multiple `--seq` and `--seq_id` arguments.

## Visualization of Results

The decoding step also generates several informative plots (if you pass "--plotting") to help you interpret prediction results from the model and the compositional features of your sequences:

- **Classification Counts**: Shows the proportion of sequences classified as nuclear vs. mitochondrial.  
  ![classification_counts](./webui/static/Plot_example/classification_counts.png)

- **GC Content per Sequence**: Shows the GC content proportion across sequences, which often differs between nuclear and mitochondrial genes.
  ![gc_content_stacked_bar](./webui/static/Plot_example/gc_content_stacked_bar.png)

- **Log-Likelihood Distribution**: Displays the log-likelihood scores for all decoded sequences; higher values indicate better model fit.
  ![loglikelihood_distribution](./webui/static/Plot_example/loglikelihood_distribution.png)

- **Hidden State Proportions**: Illustrates the proportion of "time" each sequence spends in the "nuclear" or "mitochondrial" hidden state.
  ![state_proportions_stacked_bar](./webui/static/Plot_example/state_proportions_stacked_bar.png)
