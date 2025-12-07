# mitoSpotter Web UI (Demo)

A minimal but extendable web console that wraps the existing decoder scripts without modifying them. It now targets the unified NT decoder (`scripts/05_decode_path_nt.py`). The backend reuses your Python scripts, while the frontend offers an interactive form for job submission and result visualization.

## Features

- Upload FASTA files or paste raw/FASTA-formatted nucleotide sequences.
- Configure decode options (mode `nt1/nt2/nt3`, method `viterbi/posterior`, `min_len`, optional path emission).
- Elegant UI with the requested color palette (`#4c2c2e`, `#e8b287`, `#ae9890`, `#a888a0`).
- Displays decoded summaries in a table, plus optional state paths.
- Backend and pipeline are decoupled; the web layer simply shells out to the existing script.
- Modes: single-nucleotide (1-nt), di-nucleotide (2-nt), tri-nucleotide (3-nt).

## Local Quickstart

1. **Install dependencies** (inside your preferred virtual environment):

   ```bash
   pip install flask biopython numpy matplotlib
   ```

   > `biopython` and `numpy` are already required by the pipeline itself. Add other packages here if your environment does not have them yet.

2. **Run the demo server** from the project root:

   ```bash
   python -m webui.app
   ```

   The server listens on `http://0.0.0.0:8000` by default.

3. **Open the UI** at [http://localhost:8000](http://localhost:8000) and submit a job.

## Deployment Notes

- The web layer uses:
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
- Amino-acid and codon decoders are not used by the current web layer.
- All resources are resolved relative to the project root (`DecodeConfig.with_project_defaults`). If you reorganise files or add more trained models, update `DecodeConfig` or extend `/api/config` to expose the new presets.
- For production, consider running under `gunicorn` or another WSGI server:

  ```bash
  gunicorn -w 2 -b 0.0.0.0:8000 'webui.app:app'
  ```

  Static assets are served by Flask; reverse proxies (nginx, Caddy, etc.) can handle caching/compression as needed.

## Extending the Demo

- Add authentication or per-user work queues by building additional endpoints around `DecodeRunner`.
- Swap `DecodeRunner` for a job dispatcher (Celery, message queue) without touching the UI; only the `/api/run` handler needs to change.
- Surface more runtime parameters by whitelisting them in the form and passing through to `DecodeRunner`.
- The frontend uses vanilla JS; you can migrate to a component framework later without changing the backend contract (`POST /api/run` with form data, JSON response). The unified decoder expects: `mode` in {`nt1`,`nt2`,`nt3`}, `method` in {`viterbi`,`posterior`}, and `min_len` (nt threshold).

## Testing

- Basic invocation can be verified with a small FASTA snippet or existing files from `out/`.
- The backend `DecodeRunner` is isolated in `webui/pipeline_runner.py`, making it straightforward to unit test by mocking `subprocess.run`.
