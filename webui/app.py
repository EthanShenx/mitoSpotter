from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

try:  # Allow running via `python -m webui.app` or `python webui/app.py`
    from .pipeline_runner import DecodeConfig, DecodeRunner
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from webui.pipeline_runner import DecodeConfig, DecodeRunner  # type: ignore

from flask import Flask, jsonify, request, send_from_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__, static_folder="static", static_url_path="/static")
decode_config = DecodeConfig.with_project_defaults(PROJECT_ROOT)
runner = DecodeRunner(decode_config)


def refresh_runner() -> None:
    """Rescan assets so newly added species bundles appear without restart."""
    global decode_config, runner
    decode_config = DecodeConfig.with_project_defaults(PROJECT_ROOT)
    runner = DecodeRunner(decode_config)


def _parse_sequence_text(raw: str) -> List[Tuple[str, str]]:
    """Interpret pasted text as either FASTA or line-separated sequences."""
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
def get_config():
    refresh_runner()
    cfg = runner.config
    # Defaults: prefer codon; then aa; then nt1; then nt2; then nt3
    default_mode = (
        "codon"
        if cfg.species_map_by_mode.get("codon")
        else (
            "aa"
            if cfg.species_map_by_mode.get("aa")
            else ("nt1" if cfg.species_map_by_mode.get("nt1") else ("nt2" if cfg.species_map_by_mode.get("nt2") else "nt3"))
        )
    )
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
                "species_id": default_assets.species_id,
                "model_json": str(default_assets.model_json.relative_to(PROJECT_ROOT)),
                "vocab_json": str(default_assets.vocab_json.relative_to(PROJECT_ROOT)),
                "states_json": str(default_assets.states_json.relative_to(PROJECT_ROOT)),
                "code": "auto",
                "min_orf_nt": 150,
            },
            "modes": {
                "codon": {"species_options": list_species("codon")},
                "aa": {"species_options": list_species("aa")},
                "nt1": {"species_options": list_species("nt1")},
                "nt2": {"species_options": list_species("nt2")},
                "nt3": {"species_options": list_species("nt3")},
            },
            "code_options": [
                {"value": "auto", "label": "Auto detect"},
                {"value": "standard", "label": "Standard"},
                {"value": "vertebrate_mito", "label": "Vertebrate mitochondrial"},
            ],
        }
    )


@app.post("/api/run")
def run_decode():
    refresh_runner()
    form = request.form
    mode = form.get("mode", "codon")
    code = form.get("code", "auto")
    min_orf_nt = int(form.get("min_orf_nt", 150) or 150)
    emit_path = form.get("emit_path", "false").lower() in ("true", "1", "yes", "on")
    species_id = form.get("species")
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
            code=code,
            min_orf_nt=min_orf_nt,
            emit_path=emit_path,
            species_id=species_id,
        )
    except Exception as exc:  # noqa: BLE001 keep surface succinct
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500

    if temp_file_path and temp_file_path.exists():
        temp_file_path.unlink(missing_ok=True)

    return jsonify(result)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/healthz")
def healthcheck():
    return jsonify({"status": "ok"})


@app.get("/favicon.ico")
def favicon():
    # Serve the site logo as the favicon so it appears in the browser tab.
    return send_from_directory(app.static_folder, "logo.png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
