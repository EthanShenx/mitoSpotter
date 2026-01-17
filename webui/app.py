from __future__ import annotations
# Flask web entrypoint for mitoSpotter demo UI and API

import json
import queue
import tempfile
import threading
import time
# Standard library imports and typing helpers
from pathlib import Path
from typing import List, Tuple, Generator

try:  # Allow running via `python -m webui.app` or `python webui/app.py`
    from .pipeline_runner import DecodeConfig, DecodeRunner
    from .training_runner import TrainingConfig, TrainingRunner
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from webui.pipeline_runner import DecodeConfig, DecodeRunner  # type: ignore
    from webui.training_runner import TrainingConfig, TrainingRunner  # type: ignore

from flask import Flask, jsonify, request, send_from_directory, Response
# Flask app configured to serve static assets and JSON APIs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Create app and initialize a DecodeRunner using discovered assets
app = Flask(__name__, static_folder="static", static_url_path="/static")
decode_config = DecodeConfig.with_project_defaults(PROJECT_ROOT)
runner = DecodeRunner(decode_config)

# Initialize training runner
training_runner = TrainingRunner(PROJECT_ROOT)

# Store for SSE progress queues (job_id -> queue)
training_progress_queues: dict = {}


def refresh_runner() -> None:
    """Rescan assets so newly added species bundles appear without restart."""
    # Rebuild config/runner so UI sees new models added to out/ or out_dir/
    global decode_config, runner
    decode_config = DecodeConfig.with_project_defaults(PROJECT_ROOT)
    runner = DecodeRunner(decode_config)


def _parse_sequence_text(raw: str) -> List[Tuple[str, str]]:
    """Interpret pasted text as either FASTA or line-separated sequences."""
    # Supports both FASTA headers (>) and plain newline-delimited sequences
    raw = raw.strip()
    if not raw:
        return []

    lines = raw.splitlines()
    sequences: List[Tuple[str, str]] = []

    if lines[0].startswith(">"):
        header = None
        seq_chunks: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(seq_chunks)))
                    seq_chunks = []
                header = line[1:].strip() or f"seq{len(sequences)+1}"
            else:
                seq_chunks.append("".join(line.split()).upper())
        if header is not None:
            sequences.append((header, "".join(seq_chunks)))
    else:
        for idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            sequences.append((f"seq{idx}", "".join(line.split()).upper()))

    return sequences


@app.get("/api/config")
# Endpoint: returns available modes/species and default asset paths for the UI
def get_config():
    refresh_runner()
    cfg = runner.config
    # Defaults: prefer nt1, then nt2, then nt3 (unified NT decoder)
    default_mode = "nt1" if cfg.species_map_by_mode.get("nt1") else ("nt2" if cfg.species_map_by_mode.get("nt2") else "nt3")
    default_assets = cfg.get_species(default_mode)

    def list_species(mode: str):
        opts = []
        for key in sorted(cfg.species_map_by_mode.get(mode, {})):
            assets = cfg.species_map_by_mode[mode][key]
            opts.append(
                {
                    "value": assets.species_id,
                    "label": assets.label,
                    "model_json": str(assets.model_json.relative_to(PROJECT_ROOT)),
                    "vocab_json": str(assets.vocab_json.relative_to(PROJECT_ROOT)),
                    "states_json": str(assets.states_json.relative_to(PROJECT_ROOT)),
                }
            )
        return opts

    return jsonify(
        {
            "defaults": {
                "mode": default_mode,
                "method": "viterbi",
                "species_id": default_assets.species_id,
                "model_json": str(default_assets.model_json.relative_to(PROJECT_ROOT)),
                "vocab_json": str(default_assets.vocab_json.relative_to(PROJECT_ROOT)),
                "states_json": str(default_assets.states_json.relative_to(PROJECT_ROOT)),
            },
            "modes": {
                "nt1": {"species_options": list_species("nt1")},
                "nt2": {"species_options": list_species("nt2")},
                "nt3": {"species_options": list_species("nt3")},
            },
        }
    )


@app.post("/api/run")
# Endpoint: accepts sequences or FASTA upload and runs the decoder via DecodeRunner
def run_decode():
    refresh_runner()
    form = request.form
    mode = form.get("mode", "nt1")
    method = "viterbi"  # UI fixed to Viterbi; script supports only Viterbi per requirements
    emit_path = form.get("emit_path", "false").lower() in ("true", "1", "yes", "on")
    plotting = form.get("plotting", "false").lower() in ("true", "1", "yes", "on")
    species_id = form.get("species")
    # Regime selector for fallback assets in out_dir/04_model: "pure_em" or "pure_viterbi"
    regime = form.get("regime")
    sequence_text = form.get("sequence_text", "")

    sequences = _parse_sequence_text(sequence_text) if sequence_text else []

    uploaded = request.files.get("fasta_file")
    temp_file_path = None

    if uploaded and uploaded.filename:
        suffix = Path(uploaded.filename).suffix or ".fa"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            uploaded.save(tmp.name)
            temp_file_path = Path(tmp.name)

    if not sequences and temp_file_path is None:
        return jsonify({"error": "Provide a FASTA file or enter nucleotide sequences."}), 400

    try:
        result = runner.run(
            sequences=sequences,
            fasta_path=temp_file_path,
            mode=mode,
            method=method,
            emit_path=emit_path,
            species_id=regime or species_id,
            plotting=plotting,
        )
    except Exception as exc:  # noqa: BLE001 keep surface succinct
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500

    if temp_file_path and temp_file_path.exists():
        temp_file_path.unlink(missing_ok=True)

    return jsonify(result)


@app.get("/")
# Serve the SPA index
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/healthz")
# Lightweight health check
def healthcheck():
    return jsonify({"status": "ok"})


@app.get("/favicon.ico")
def favicon():
    # Serve the site logo as the favicon so it appears in the browser tab.
    return send_from_directory(app.static_folder, "logo.png")


@app.get("/api/plots/<path:relpath>")
# Endpoint: securely serve generated plot images from Plot_res_* folders
def get_plot(relpath: str):
    """Serve plot images generated during a decode run.

    Security: only allow paths within the project root and under a Plot_res_* directory,
    and restrict to common image extensions.
    """
    # Basic extension check
    allowed_ext = {".png", ".jpg", ".jpeg"}
    if Path(relpath).suffix.lower() not in allowed_ext:
        return jsonify({"error": "Unsupported file type"}), 400

    # Only allow Plot_res_* top-level directories
    parts = Path(relpath).parts
    if not parts or not parts[0].startswith("Plot_res_"):
        return jsonify({"error": "Invalid plot path"}), 400

    abs_path = (PROJECT_ROOT / relpath).resolve()
    try:
        # Ensure the resolved path is within the project root
        abs_path.relative_to(PROJECT_ROOT)
    except Exception:
        return jsonify({"error": "Path traversal denied"}), 400

    if not abs_path.exists() or not abs_path.is_file():
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(abs_path.parent, abs_path.name)


# -------------------- Training API Endpoints -------------------- #

@app.post("/api/train")
def start_training():
    """Start a new training job. Returns job_id for progress tracking."""
    form = request.form

    # Get uploaded files
    nuclear_file = request.files.get("nuclear_tsv")
    mito_file = request.files.get("mito_tsv")

    if not nuclear_file or not nuclear_file.filename:
        return jsonify({"error": "Nuclear TSV file is required"}), 400
    if not mito_file or not mito_file.filename:
        return jsonify({"error": "Mito TSV file is required"}), 400

    # Save uploaded files to temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="mitospotter_train_"))
    nuclear_path = temp_dir / "nuclear.tsv"
    mito_path = temp_dir / "mito.tsv"

    nuclear_file.save(str(nuclear_path))
    mito_file.save(str(mito_path))

    # Parse training parameters
    ngram = int(form.get("ngram", "1"))
    train_method = form.get("train_method", "em")
    learn = form.get("learn", "et")
    n_em_iter = int(form.get("n_em_iter", "20"))
    n_viterbi_iter = int(form.get("n_viterbi_iter", "20"))
    self_loop = float(form.get("self_loop", "0.995"))
    emis_smooth = float(form.get("emis_smooth", "1.0"))
    trans_smooth = float(form.get("trans_smooth", "1.0"))
    sample = float(form.get("sample", "1.0"))

    # Determine output directory
    output_dir = PROJECT_ROOT / "out" / f"{ngram}nt"

    # Create training config
    config = TrainingConfig(
        nuclear_tsv=nuclear_path,
        mito_tsv=mito_path,
        output_dir=output_dir,
        ngram=ngram,
        train_method=train_method,
        learn=learn,
        n_em_iter=n_em_iter,
        n_viterbi_iter=n_viterbi_iter,
        self_loop=self_loop,
        emis_smooth=emis_smooth,
        trans_smooth=trans_smooth,
        sample=sample,
    )

    # Create job
    job = training_runner.create_job(config)

    # Create progress queue for this job
    progress_queue: queue.Queue = queue.Queue()
    training_progress_queues[job.job_id] = progress_queue

    # Start training in background thread
    def run_training():
        def progress_callback(info: dict):
            try:
                progress_queue.put_nowait(info)
            except queue.Full:
                pass

        try:
            training_runner.run_training(job, progress_callback)
        finally:
            # Signal completion
            progress_queue.put({"type": "done"})
            # Clean up temp files after a delay
            def cleanup():
                time.sleep(5)
                try:
                    nuclear_path.unlink(missing_ok=True)
                    mito_path.unlink(missing_ok=True)
                    temp_dir.rmdir()
                except Exception:
                    pass
            threading.Thread(target=cleanup, daemon=True).start()

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({
        "job_id": job.job_id,
        "status": "started",
        "message": "Training job started. Use /api/train/progress/<job_id> for SSE updates."
    })


@app.get("/api/train/progress/<job_id>")
def training_progress(job_id: str):
    """SSE endpoint for real-time training progress."""
    progress_queue = training_progress_queues.get(job_id)

    if not progress_queue:
        return jsonify({"error": "Job not found"}), 404

    def generate() -> Generator[str, None, None]:
        """Generate SSE events from the progress queue."""
        while True:
            try:
                # Wait for progress update with timeout
                info = progress_queue.get(timeout=30)

                # Check for completion signal
                if info.get("type") == "done":
                    job = training_runner.get_job(job_id)
                    if job:
                        yield f"data: {json.dumps(job.to_dict())}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break

                yield f"data: {json.dumps(info)}\n\n"

            except queue.Empty:
                # Send keepalive
                yield f": keepalive\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/train/status/<job_id>")
def training_status(job_id: str):
    """Get current status of a training job."""
    job = training_runner.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job.to_dict())


if __name__ == "__main__":
    # Dev server entrypoint (use gunicorn for production)
    app.run(host="0.0.0.0", port=8000, debug=True)
