# mitoSpotter Web UI (Demo)

A minimal but extendable web console that wraps the existing decoder scripts without modifying them. It targets the unified NT decoder (`scripts/05_decode_path_nt.py`). The backend reuses your Python scripts, while the frontend offers an interactive form for job submission, plotting, and result visualization.

## Features

- Upload FASTA or paste raw/FASTA‑formatted nucleotide sequences (mutually exclusive demo helpers; see below).
- Choose decoder type (n‑gram): `nt1`, `nt2`, or `nt3`.
- Optional features:
  - Include Viterbi state path (per‑sequence path visualization).
  - Generate and show plots (GC/AT content, state proportions, classification counts, log‑likelihood distribution).
- Results view:
  - Summary sentence (predicted class of the first record).
  - Plots grid (click to zoom and download; no auto‑download).
  - Results table (ID, Call, Nuclear%, Mito%, Length).
  - Paths section (compact N/M runs) when path emission is enabled.
- “See instructions” button shows an in‑app help modal.
- Elegant UI palette (`#4c2c2e`, `#e8b287`, `#ae9890`, `#a888a0`).
- Backend and UI are decoupled; the web layer shells out to the decoder.

## Local Quickstart

1. **Install dependencies** (in your preferred environment):

   ```bash
   pip install flask biopython numpy matplotlib
   ```

   > `biopython` and `numpy` are required by the pipeline; `matplotlib` is required to generate plots.

2. **Run the demo server** from the project root:

   ```bash
   python -m webui.app
   ```

   The server listens on `http://0.0.0.0:8000` by default.

3. **Open the UI** at [http://localhost:8000](http://localhost:8000) and submit a job.

Windows note: if plots are not displayed when “Generate and show plots” is ticked, confirm `matplotlib` is installed in the same environment running Flask. The decoder warns on stderr if plotting is disabled.

## Assets & Deployment Notes

- Decoder used by the web layer:
  - `scripts/05_decode_path_nt.py`
- For 1-nt mode:
  - `out/<sp>_mitoSpotter_hmm_1nt.json`
  - `out/<sp>_nt_vocab.json`
  - `out/<sp>_state_names.json`
- For 2-nt mode:
  - `out/<sp>_mitoSpotter_hmm_2nt.json`
  - `out/<sp>_nt_2_vocab.json` or `out/<sp>_nt2_vocab.json` (fallback `out/<sp>_nt_vocab.json`)
  - `out/<sp>_state_names.json`
- For 3-nt mode:
  - `out/<sp>_mitoSpotter_hmm_3nt.json`
  - `out/<sp>_nt3_vocab.json`
  - `out/<sp>_state_names.json`
- Amino‑acid and codon decoders are not used by the current web UI.
- All resources are resolved relative to the project root (`DecodeConfig.with_project_defaults`). If you reorganise files or add more trained models, restart the server; the UI rescans assets on each request.
- For production, consider running under `gunicorn` or another WSGI server:

  ```bash
  gunicorn -w 2 -b 0.0.0.0:8000 'webui.app:app'
  ```

  Static assets are served by Flask; reverse proxies (nginx, Caddy, etc.) can handle caching/compression as needed.

## Using the UI

1) Choose decoder type (1‑nt, 2‑nt, 3‑nt).
2) Provide input via one of:
   - Paste sequences in the textarea (FASTA or plain lines), or
   - Upload a single FASTA file.
   - Demo helpers: “Load demo sequence” or “Load demo FASTA”. To prevent conflicts, loading the demo sequence clears any loaded FASTA; loading the demo FASTA is blocked until the demo sequence is cleared.
3) Options:
   - Include Viterbi state path: show compact N/M runs.
   - Generate and show plots: produce GC/AT and state proportion charts, classification counts, and LL distribution.
4) Click “Run Decode”.
5) Review results:
   - Summary sentence
   - Plots grid (click to zoom; Download button to save)
   - Results table and optional Paths
6) Export TSV/CSV via the Results card’s “Download Results” menu.

## Extending the Demo

- Add authentication or per-user work queues by building additional endpoints around `DecodeRunner`.
- Swap `DecodeRunner` for a job dispatcher (Celery, message queue) without touching the UI; only the `/api/run` handler needs to change.
- Surface more runtime parameters by whitelisting them in the form and passing through to `DecodeRunner`.
- The frontend uses vanilla JS; you can migrate to a component framework later without changing the backend contract (`POST /api/run` with form data, JSON response). The unified decoder expects `ngram` (1/2/3), explicit model/vocab/states paths, and optional `--emit_path`/`--plotting` flags.

## Testing

- Basic invocation can be verified with a small FASTA snippet or existing files from `out_dir/`.
- The backend `DecodeRunner` is isolated in `webui/pipeline_runner.py`, making it straightforward to unit test by mocking `subprocess.run`.
- Plot serving endpoint: `/api/plots/<relative_path>` securely serves images only from `Plot_res_*` under the project root.
