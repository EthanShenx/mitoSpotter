# mitoSpotter

An annotation‑independent classifier that distinguishes mitochondrial from nuclear transcripts using Hidden Markov Models (HMMs). mitoSpotter learns simple 2‑state HMMs over nucleotide n‑grams (1‑nt, 2‑nt, or 3‑nt) and decodes new sequences to call “mitochondrial” vs “nuclear”, along with interpretable per‑state fractions and optional state paths.

• Scripts for end‑to‑end training, decoding, and benchmarking.
• Optional Flask‑based web UI for interactive decoding and exports.


## Features

- 2‑state Multinomial HMM trained on nucleotide tokens (1‑nt/2‑nt/3‑nt).
- Three training regimes: EM, Viterbi, and Hybrid (EM→Viterbi).
- Fast Viterbi decoding with summary metrics per sequence.
- Simple, explicit JSON artifacts: `model.json`, `vocab.json`, `states.json`.
- Reproducible benchmark pipeline across unit sizes and regimes.
- Web UI wrapper that shells out to the decoder script.


## Install

Prerequisites:
- Python 3.9+ recommended

Core dependencies:
- numpy
- biopython
- tqdm (optional: for progress bars during training)
- flask (optional: only for the web UI)

Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install numpy biopython tqdm flask
```


## Quickstart: Decode With A Trained Model

Pick a trained model triple (model/vocab/states) matching your desired n‑gram size. The benchmark pipeline writes models under `out_dir/04_model/<1nt|2nt|3nt>/<regime>/` as:

- `model.json` – start, transition, emission probabilities
- `vocab.json` – token order/index used during training
- `states.json` – `{ "0": "nuclear", "1": "mitochondrial" }`

Decode a FASTA with the unified nucleotide decoder `scripts/05_decode_path_nt.py:1`:

```bash
python scripts/05_decode_path_nt.py \
  --ngram 1 \
  --model_json out_dir/04_model/1nt/hybrid_em10_vit10/model.json \
  --vocab_json out_dir/04_model/1nt/hybrid_em10_vit10/vocab.json \
  --states_json out_dir/04_model/1nt/hybrid_em10_vit10/states.json \
  --fasta path/to/input.fa \
  --min_len 150 \
  --out_tsv results.tsv
```

Output TSV columns:

- `id` – sequence ID
- `loglik` – forward log‑likelihood of the observation sequence
- `nuclear_frac` – fraction of decoded path in nuclear state
- `mito_frac` – fraction of decoded path in mitochondrial state
- `call` – winning label by fraction (ties favor nuclear)
- `seq_len_nt` – nucleotide length (after cleaning A/C/G/T)
- `n_tokens_{k}mer` – number of tokens given n‑gram size

If `--emit_path` is provided, additional lines are emitted as `#<id>_PATH\t<state_ids...>`.


## Train From Scratch (End‑to‑End)

1) Extract transcript ID sets from a GTF using `scripts/01_from_gtf_extract_id.py:1`:

```bash
python scripts/01_from_gtf_extract_id.py \
  --gtf data/Homo_sapiens.GRCh38.115.chr.gtf \
  --outdir out_dir/01_ids \
  --prefix human_protein_coding_  # optional
```

This writes `out_dir/01_ids/human_protein_coding_ids_mito.txt` and `..._ids_nuclear.txt`.

2) Split cDNA/CDS FASTA by mito/nuclear IDs with `scripts/02_fasta_split_by_id.py:1`:

```bash
python scripts/02_fasta_split_by_id.py \
  --fasta data/Homo_sapiens.GRCh38.cds.all.fa \
  --mito_ids out_dir/01_ids/human_protein_coding_ids_mito.txt \
  --nuc_ids  out_dir/01_ids/human_protein_coding_ids_nuclear.txt \
  --mode cds \
  --outdir out_dir/02_split_fasta \
  --prefix human_
```

3) Tokenize sequences into 1‑nt/2‑nt/3‑nt units and split train/holdout using `scripts/03_sequence2unit_nt.py:1`:

```bash
# Nuclear
python scripts/03_sequence2unit_nt.py \
  --fasta out_dir/02_split_fasta/human_nuclear_cds.fa \
  --mode 1nt \
  --train_tsv  out_dir/03_unit/human_nuclear_1nt_train.tsv \
  --holdout_tsv out_dir/03_unit/human_nuclear_1nt_holdout.tsv

# Mitochondrial
python scripts/03_sequence2unit_nt.py \
  --fasta out_dir/02_split_fasta/human_mito_cds.fa \
  --mode 1nt \
  --train_tsv  out_dir/03_unit/human_mito_1nt_train.tsv \
  --holdout_tsv out_dir/03_unit/human_mito_1nt_holdout.tsv
```

4) Train a 2‑state HMM with `scripts/04_train_hmm_nt.py:1` (choose EM/Viterbi/Hybrid):

```bash
python scripts/04_train_hmm_nt.py \
  --nuclear_nt_tsv out_dir/03_unit/human_nuclear_1nt_train.tsv \
  --mito_nt_tsv    out_dir/03_unit/human_mito_1nt_train.tsv \
  --ngram 1 \
  --train_method hybrid \
  --n_em_iter 10 --n_viterbi_iter 10 \
  --n_workers 4 \
  --out_model_json  out_dir/04_model/1nt/hybrid_em10_vit10/model.json \
  --out_vocab_json  out_dir/04_model/1nt/hybrid_em10_vit10/vocab.json \
  --out_states_json out_dir/04_model/1nt/hybrid_em10_vit10/states.json
```

5) Decode and evaluate on holdout with `scripts/05_decode_path_nt.py:1`:

```bash
python scripts/05_decode_path_nt.py \
  --ngram 1 \
  --model_json out_dir/04_model/1nt/hybrid_em10_vit10/model.json \
  --vocab_json out_dir/04_model/1nt/hybrid_em10_vit10/vocab.json \
  --states_json out_dir/04_model/1nt/hybrid_em10_vit10/states.json \
  --fasta out_dir/02_split_fasta/human_mito_cds.fa \
  --out_tsv out_dir/04_model/pred_mito.tsv
```


## Benchmark Pipeline

The pipeline `scripts/06_benchmark_pipeline.py:1` trains 1‑nt/2‑nt/3‑nt models across five regimes and evaluates on holdout sets, saving results and a summary table.

Inputs expected under `out_dir/03_unit/` (from step 3). Models and results are written under `out_dir/04_model/`.

Example:

```bash
python scripts/06_benchmark_pipeline.py \
  --project_root $(pwd) \
  --train_script scripts/04_train_hmm_nt.py \
  --decode_script scripts/05_decode_path_nt.py \
  --n_workers 4
```

Notes:
- The default epilog mentions `05_decode_path_nt_v2.py`; use `scripts/05_decode_path_nt.py:1` as shown above.
- Results are saved to `out_dir/04_model/benchmark_results.csv` and printed in a compact table.


## Use Cases For Each Script (Human example)

This section mirrors your requested, script‑by‑script workflow for human (GRCh38) data.

1) First download / clone and prepare a fresh environment

```bash
# Clone (or place sources locally)
git clone <your-repo-url> mitoSpotter
cd mitoSpotter

# Conda environment (or use venv)
conda create -n mitoSpotter python=3.10 -y
conda activate mitoSpotter

# Dependencies
pip install numpy biopython tqdm flask
```

2) At the mitoSpotter root directory, create output folders

```bash
mkdir -p out_dir/01_ids \
         out_dir/02_split_fasta \
         out_dir/03_unit/train out_dir/03_unit/holdout \
         out_dir/04_model
```

3) 01_from_gtf_extract_id.py

a) Basic: protein-coding only (default)

```bash
python scripts/01_from_gtf_extract_id.py \
  --gtf data/Homo_sapiens.GRCh38.115.chr.gtf \
  --outdir out_dir/01_ids \
  --protein_coding_only \
  --prefix human_protein_coding_
```

b) Include all transcript types (not just protein coding)

```bash
python scripts/01_from_gtf_extract_id.py \
  --gtf data/Homo_sapiens.GRCh38.115.chr.gtf \
  --outdir out_dir/01_ids \
  --prefix human_all_  \
  --no-protein_coding_only
```

4) 02_fasta_split_by_id.py — split a CDS FASTA

```bash
python scripts/02_fasta_split_by_id.py \
  --fasta data/Homo_sapiens.GRCh38.cds.all.fa \
  --mito_ids out_dir/01_ids/human_protein_coding_ids_mito.txt \
  --nuc_ids out_dir/01_ids/human_protein_coding_ids_nuclear.txt \
  --outdir out_dir/02_split_fasta \
  --prefix human_ \
  --mode cds
```

5) 03_sequence2unit_nt.py — tokenize into 1nt/2nt/3nt and split train/holdout

```bash
for loc in nuclear mito; do
  for kind in cds; do
    for mode in 3nt 2nt 1nt; do
      python scripts/03_sequence2unit_nt.py \
        --fasta out_dir/02_split_fasta/human_${loc}_${kind}.fa \
        --mode ${mode} \
        --train_tsv out_dir/03_unit/train/human_${loc}_${mode}_train.tsv \
        --holdout_tsv out_dir/03_unit/holdout/human_${loc}_${mode}_holdout.tsv \
        --train_frac 0.7
    done
  done
done
```

Optional: 03_sequence2unit_orf.py (if present in your environment)

Note: This repository snapshot does not include `scripts/03_sequence2unit_orf.py`. If you have that script available, the following loop shows how to generate amino‑acid (`aa`) and reading‑frame (`3nt_orf`) tokens using a genetic code per location:

```bash
for loc in nuclear mito; do

  if [[ "$loc" == "nuclear" ]]; then
    genetic_code="nuclear"
  elif [[ "$loc" == "mito" ]]; then
    genetic_code="mito"
  fi

  for kind in cds; do
    for mode in aa 3nt_orf; do

      python scripts/03_sequence2unit_orf.py \
        --fasta out_dir/02_split_fasta/human_${loc}_${kind}.fa \
        --mode "${mode}" \
        --genetic_code "${genetic_code}" \
        --train_tsv out_dir/03_unit/train/human_${loc}_${mode}_train.tsv \
        --holdout_tsv out_dir/03_unit/holdout/human_${loc}_${mode}_holdout.tsv \
        --train_frac 0.7

    done
  done
done
```

6) 04_train_hmm_nt.py — EM runs

```bash
for N in 1 2 3; do
  python scripts/04_train_hmm_nt.py \
    --nuclear_nt_tsv out_dir/03_unit/train/human_nuclear_${N}nt_train.tsv \
    --mito_nt_tsv    out_dir/03_unit/train/human_mito_${N}nt_train.tsv \
    --ngram ${N} \
    --train_method em \
    --learn et \
    --n_workers 20 \
    --n_em_iter 20 \
    --out_model_json  out_dir/04_model/human_${N}nt_model_em.json \
    --out_vocab_json  out_dir/04_model/human_${N}nt_vocab_em.json \
    --out_states_json out_dir/04_model/human_${N}nt_states_em.json \
    --track_memory
done
```

7) 04_train_hmm_nt.py — Viterbi runs

```bash
for N in 1 2 3; do
  python scripts/04_train_hmm_nt.py \
    --nuclear_nt_tsv out_dir/03_unit/train/human_nuclear_${N}nt_train.tsv \
    --mito_nt_tsv    out_dir/03_unit/train/human_mito_${N}nt_train.tsv \
    --ngram ${N} \
    --train_method viterbi \
    --n_workers 20 \
    --n_viterbi_iter 20 \
    --out_model_json  out_dir/04_model/human_${N}nt_model_viterbi.json \
    --out_vocab_json  out_dir/04_model/human_${N}nt_vocab_viterbi.json \
    --out_states_json out_dir/04_model/human_${N}nt_states_viterbi.json \
    --track_memory
done
```

8) 04_train_hmm_nt.py — Hybrid runs

```bash
configs=(
  "e10_v10 10 10"
  "e5_v15 5 15"
  "e15_v5 15 5"
)

for N in 1 2 3; do
  for cfg in "${configs[@]}"; do
    set -- $cfg
    SUFFIX=$1
    N_EM=$2
    N_VIT=$3

    python scripts/04_train_hmm_nt.py \
      --nuclear_nt_tsv out_dir/03_unit/train/human_nuclear_${N}nt_train.tsv \
      --mito_nt_tsv out_dir/03_unit/train/human_mito_${N}nt_train.tsv \
      --ngram ${N} \
      --train_method hybrid \
      --n_workers 20 \
      --learn et \
      --n_em_iter ${N_EM} \
      --n_viterbi_iter ${N_VIT} \
      --out_model_json  out_dir/04_model/human_${N}nt_model_hybrid_${SUFFIX}.json \
      --out_vocab_json  out_dir/04_model/human_${N}nt_vocab_hybrid_${SUFFIX}.json \
      --out_states_json out_dir/04_model/human_${N}nt_states_hybrid_${SUFFIX}.json \
      --track_memory
  done
done
```

## Web UI (Optional)

The demo UI in `webui/` provides an interactive front‑end that shells out to the decoder. It auto‑discovers trained assets from `out/` (or your configured directory in `webui/pipeline_runner.py:1`).

Run locally:

```bash
pip install flask biopython numpy
python -m webui.app  # serves http://localhost:8000
```

Then open http://localhost:8000 and upload a FASTA or paste sequences. You can export results as TSV/CSV from the UI.


## Data Preparation Tips

- Use Ensembl GTF and FASTA for consistent IDs (see `data/`).
- The ID extractor `scripts/01_from_gtf_extract_id.py:1` drops Ensembl transcript version suffixes, e.g., `ENST… .2 → ENST…`.
- For 3‑nt tokenization, sequences are truncated to multiples of 3 to preserve codons.
- The decoder applies a `--min_len` filter on nucleotide length before tokenization (default 150 nt).


## Repository Structure

- `scripts/01_from_gtf_extract_id.py:1` – build mito/nuclear transcript ID lists from a GTF
- `scripts/02_fasta_split_by_id.py:1` – split cDNA/CDS FASTA into mito/nuclear subsets
- `scripts/03_sequence2unit_nt.py:1` – convert sequences into 1‑nt/2‑nt/3‑nt token TSVs
- `scripts/04_train_hmm_nt.py:1` – train 2‑state HMM (EM/Viterbi/Hybrid), save JSON artifacts
- `scripts/05_decode_path_nt.py:1` – unified nucleotide decoder (Viterbi + forward LL)
- `scripts/06_benchmark_pipeline.py:1` – orchestrate training/decoding and compute metrics
- `webui/` – Flask UI; `webui/app.py:1` server, `webui/static/` assets, `webui/pipeline_runner.py:1` glue
- `data/` – reference GTF and FASTA (large)
- `out_dir/` – IDs, splits, token TSVs, models, predictions


## Troubleshooting

- If you see file‑not‑found errors for models/vocabs/states, verify the paths and that the chosen `--ngram` matches the artifact set.
- Large GTF/FASTA files can be several GB; ensure adequate disk and RAM. Training supports `--n_workers` for parallelism.
- Enable memory reporting with `--track_memory` on training/decoding scripts (uses `tracemalloc`).
- Decoder input: provide one of `--fasta`, repeated `--seq`/`--seq_id` pairs, or `--stdin`.


## License and Citation

No license file is included in this repository. By default, all rights are reserved. For licensing or citation guidance, please contact the author/maintainer.
