# mitoSpotter CLI Workflow (Scripts)

This guide walks through the end‑to‑end command‑line workflow for preparing data, training HMMs, decoding sequences, and benchmarking. All commands are copy‑paste ready and assume execution from the project root unless noted.

## Prerequisites

- Python 3.9+
- Recommended packages:
  - `pip install biopython numpy matplotlib tqdm`
- Input data (examples included in `data/`):
  - Ensembl GTF per species (e.g., human, mouse, rat)
  - cDNA or CDS FASTA per species

Directory conventions used below:
- IDs, tokenized data, and models: `out_dir/`
- Scripts: `scripts/`

## 1) Build transcript ID lists from GTF

Extract transcript IDs (mitochondrial vs nuclear). By default only `protein_coding` is kept.

```bash
python scripts/01_from_gtf_extract_id.py \
  --gtf data/Homo_sapiens.GRCh38.115.chr.gtf \
  --outdir out_dir/01_ids \
  --prefix human_protein_coding_
```

Outputs in `out_dir/01_ids/`:
- `human_protein_coding_ids_mito.txt`
- `human_protein_coding_ids_nuclear.txt`

Tips:
- Disable the protein‑coding filter: add `--no-protein_coding_only`.
- The `--prefix` is optional but keeps files clearly labeled.

## 2) Split FASTA into mitochondrial vs nuclear sets

Split cDNA or CDS FASTA into two files using the ID lists.

cDNA example:
```bash
python scripts/02_fasta_split_by_id.py \
  --fasta data/Homo_sapiens.GRCh38.cdna.all.fa \
  --mito_ids out_dir/01_ids/human_protein_coding_ids_mito.txt \
  --nuc_ids  out_dir/01_ids/human_protein_coding_ids_nuclear.txt \
  --outdir   out_dir/02_split_fasta \
  --mode     cdna \
  --prefix   human_
```

CDS example:
```bash
python scripts/02_fasta_split_by_id.py \
  --fasta data/Homo_sapiens.GRCh38.cds.all.fa \
  --mito_ids out_dir/01_ids/human_protein_coding_ids_mito.txt \
  --nuc_ids  out_dir/01_ids/human_protein_coding_ids_nuclear.txt \
  --outdir   out_dir/02_split_fasta \
  --mode     cds \
  --prefix   human_
```

Outputs (examples):
- `out_dir/02_split_fasta/human_mito_cdna.fa`, `out_dir/02_split_fasta/human_nuclear_cdna.fa`
- `out_dir/02_split_fasta/human_mito_cds.fa`,  `out_dir/02_split_fasta/human_nuclear_cds.fa`

## 3) Tokenize sequences into 1‑nt, 2‑nt, or 3‑nt units

Generate tokenized TSVs for each unit size and split into train/holdout.

1‑nt example (using the split nuclear CDS FASTA):
```bash
python scripts/03_sequence2unit_nt.py \
  --fasta out_dir/02_split_fasta/human_nuclear_cds.fa \
  --mode  1nt \
  --train_tsv   out_dir/03_unit/train/human_nuclear_1nt_train.tsv \
  --holdout_tsv out_dir/03_unit/holdout/human_nuclear_1nt_holdout.tsv \
  --train_frac  0.7
```

Repeat for mitochondrial and for `2nt` and `3nt` as needed, e.g.:
```bash
# Mito, 2‑nt
python scripts/03_sequence2unit_nt.py \
  --fasta out_dir/02_split_fasta/human_mito_cds.fa \
  --mode  2nt \
  --train_tsv   out_dir/03_unit/train/human_mito_2nt_train.tsv \
  --holdout_tsv out_dir/03_unit/holdout/human_mito_2nt_holdout.tsv

# Nuclear, 3‑nt (codon stride)
python scripts/03_sequence2unit_nt.py \
  --fasta out_dir/02_split_fasta/human_nuclear_cds.fa \
  --mode  3nt \
  --train_tsv   out_dir/03_unit/train/human_nuclear_3nt_train.tsv \
  --holdout_tsv out_dir/03_unit/holdout/human_nuclear_3nt_holdout.tsv
```

## 4) Train 2‑state Multinomial HMMs

Train a model per unit size; choose EM/Viterbi/Hybrid regimes and iteration counts.

Common arguments:
- `--ngram {1,2,3}`
- `--train_method {em,viterbi,hybrid}`
- `--n_em_iter`, `--n_viterbi_iter`
- `--n_workers` for parallel EM/Viterbi steps

Example: 1‑nt, hybrid 10 EM + 10 Viterbi
```bash
mkdir -p out_dir/04_model/1nt/hybrid_em10_vit10
python scripts/04_train_hmm_nt.py \
  --nuclear_nt_tsv out_dir/03_unit/train/human_nuclear_1nt_train.tsv \
  --mito_nt_tsv    out_dir/03_unit/train/human_mito_1nt_train.tsv \
  --ngram 1 \
  --train_method hybrid \
  --n_em_iter 10 \
  --n_viterbi_iter 10 \
  --n_workers 2 \
  --out_model_json  out_dir/04_model/1nt/hybrid_em10_vit10/model.json \
  --out_vocab_json  out_dir/04_model/1nt/hybrid_em10_vit10/vocab.json \
  --out_states_json out_dir/04_model/1nt/hybrid_em10_vit10/states.json
```

Repeat for 2‑nt and 3‑nt by changing `--ngram` and output directories.

## 5) Decode sequences (Viterbi) and optionally plot

The unified decoder works for 1‑nt, 2‑nt, or 3‑nt via `--ngram`. Explicit paths to model, vocab, and states are required.

Inputs can be a FASTA file, repeated `--seq` (with optional `--seq_id`), or `--stdin`.

Examples:

Decode a FASTA with Viterbi paths and plotting:
```bash
python scripts/05_decode_path_nt.py \
  --method viterbi \
  --ngram 1 \
  --model_json  out_dir/04_model/1nt/hybrid_em10_vit10/model.json \
  --vocab_json  out_dir/04_model/1nt/hybrid_em10_vit10/vocab.json \
  --states_json out_dir/04_model/1nt/hybrid_em10_vit10/states.json \
  --fasta out_dir/02_split_fasta/human_mito_cds.fa \
  --emit_path \
  --plotting \
  --out_tsv decode_output.tsv
```

Decode inline sequences (two sequences):
```bash
python scripts/05_decode_path_nt.py \
  --method viterbi \
  --ngram 2 \
  --model_json  out_dir/04_model/2nt/hybrid_em10_vit10/model.json \
  --vocab_json  out_dir/04_model/2nt/hybrid_em10_vit10/vocab.json \
  --states_json out_dir/04_model/2nt/hybrid_em10_vit10/states.json \
  --seq "ACGTACGTAC" --seq_id s1 \
  --seq "GGGAAATTTCC" --seq_id s2 \
  --min_len 0 \
  --out_tsv decode_output.tsv
```

Notes:
- `--plotting` generates a timestamped `Plot_res_*` directory with PNGs (stderr logs will list their paths). No images are automatically downloaded; you can view them in the web UI or open them directly.
- `--emit_path` adds per‑sequence Viterbi state paths to the TSV (as `#<id>_PATH\t...`).

## 6) Benchmark across unit sizes and regimes

This pipeline trains five regimes for each unit size (1/2/3‑nt), decodes the holdout set, and computes metrics. Make sure tokenized train/holdout TSVs exist in `out_dir/03_unit/{train,holdout}`.

Recommended invocation (explicitly point to the existing decoder script):
```bash
python scripts/06_benchmark_pipeline.py \
  --project_root "$(pwd)" \
  --train_script scripts/04_train_hmm_nt.py \
  --decode_script scripts/05_decode_path_nt.py \
  --n_workers 2
```

Outputs (under `out_dir/04_model/`):
- `1nt/`, `2nt/`, `3nt/` model directories containing `model.json`, `vocab.json`, `states.json`
- `predictions/` with decoded TSVs
- `benchmark_results.csv` aggregating timing, memory, and accuracy metrics

Tip: A convenience wrapper exists at `scripts/run_benchmark.sh`. If it references `05_decode_path_nt_v2.py` in your copy, replace with `scripts/05_decode_path_nt.py` or pass `--decode_script` explicitly as above.

## Minimal “Start to Finish” Example (Human, CDS)

```bash
# 1) IDs
python scripts/01_from_gtf_extract_id.py \
  --gtf data/Homo_sapiens.GRCh38.115.chr.gtf \
  --outdir out_dir/01_ids \
  --prefix human_protein_coding_

# 2) Split FASTA
python scripts/02_fasta_split_by_id.py \
  --fasta data/Homo_sapiens.GRCh38.cds.all.fa \
  --mito_ids out_dir/01_ids/human_protein_coding_ids_mito.txt \
  --nuc_ids  out_dir/01_ids/human_protein_coding_ids_nuclear.txt \
  --outdir   out_dir/02_split_fasta \
  --mode     cds \
  --prefix   human_

# 3) Tokenize (1‑nt shown; repeat for mito/nuclear and 2‑nt/3‑nt as needed)
python scripts/03_sequence2unit_nt.py \
  --fasta out_dir/02_split_fasta/human_nuclear_cds.fa \
  --mode  1nt \
  --train_tsv   out_dir/03_unit/train/human_nuclear_1nt_train.tsv \
  --holdout_tsv out_dir/03_unit/holdout/human_nuclear_1nt_holdout.tsv
python scripts/03_sequence2unit_nt.py \
  --fasta out_dir/02_split_fasta/human_mito_cds.fa \
  --mode  1nt \
  --train_tsv   out_dir/03_unit/train/human_mito_1nt_train.tsv \
  --holdout_tsv out_dir/03_unit/holdout/human_mito_1nt_holdout.tsv

# 4) Train (hybrid 10+10 as example)
mkdir -p out_dir/04_model/1nt/hybrid_em10_vit10
python scripts/04_train_hmm_nt.py \
  --nuclear_nt_tsv out_dir/03_unit/train/human_nuclear_1nt_train.tsv \
  --mito_nt_tsv    out_dir/03_unit/train/human_mito_1nt_train.tsv \
  --ngram 1 \
  --train_method hybrid \
  --n_em_iter 10 \
  --n_viterbi_iter 10 \
  --n_workers 2 \
  --out_model_json  out_dir/04_model/1nt/hybrid_em10_vit10/model.json \
  --out_vocab_json  out_dir/04_model/1nt/hybrid_em10_vit10/vocab.json \
  --out_states_json out_dir/04_model/1nt/hybrid_em10_vit10/states.json

# 5) Decode with plots and paths
python scripts/05_decode_path_nt.py \
  --method viterbi \
  --ngram 1 \
  --model_json  out_dir/04_model/1nt/hybrid_em10_vit10/model.json \
  --vocab_json  out_dir/04_model/1nt/hybrid_em10_vit10/vocab.json \
  --states_json out_dir/04_model/1nt/hybrid_em10_vit10/states.json \
  --fasta out_dir/02_split_fasta/human_mito_cds.fa \
  --emit_path \
  --plotting \
  --out_tsv decode_output.tsv
```

## Troubleshooting

- Empty outputs or many skipped sequences:
  - Check `--min_len` vs your input lengths (decoder default is 150 nt).
  - Ensure tokenization mode matches the model (`--ngram`).
- Plot files not visible:
  - Use `--plotting`; look for `Plot_res_*` messages in stderr. In the web UI, enable plotting to display thumbnails.
- Benchmark script path:
  - If your copy references `05_decode_path_nt_v2.py`, pass `--decode_script scripts/05_decode_path_nt.py` explicitly.

