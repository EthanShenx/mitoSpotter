import csv
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SPECIES_LABELS = {
    "hs": "Human (hs)",
    "mm": "Mouse (mm)",
    "rn": "Rat (rn)",
}


@dataclass
class SpeciesAssets:
    """Describes the trained model asset bundle for a species code."""

    species_id: str
    label: str
    model_json: Path
    vocab_json: Path
    states_json: Path
    units: str  # "codons" or "nt"


@dataclass
class DecodeConfig:
    """Holds script paths, assets dir, and available species bundles per mode.

    This revision aligns to the unified NT decoder (scripts/05_decode_path_nt.py).
    Only nt1/nt2/nt3 modes are exposed.
    """

    script_paths: Dict[str, Path]  # {"nt1": path, "nt2": path, "nt3": path}
    assets_dir: Path
    species_map_by_mode: Dict[str, Dict[str, SpeciesAssets]]  # mode -> {species_id: assets}
    default_species_id_by_mode: Dict[str, str]

    @classmethod
    def with_project_defaults(cls, project_root: Path) -> "DecodeConfig":
        """Create a config by scanning the default project layout."""

        # Unified decoder supports ngram=1/2/3 via a single script
        unified = project_root / "scripts" / "05_decode_path_nt.py"
        script_paths = {"nt1": unified, "nt2": unified, "nt3": unified}
        assets_dir = project_root / "out"

        species_map_by_mode: Dict[str, Dict[str, SpeciesAssets]] = {"nt1": {}, "nt2": {}, "nt3": {}}

        def add_species(mode: str, species_id: str, model_path: Path, vocab_path: Path, states_path: Path) -> None:
            if not (model_path.exists() and vocab_path.exists() and states_path.exists()):
                return
            label = SPECIES_LABELS.get(species_id, species_id.upper())
            if mode == "nt1":
                units = "nt"
            elif mode == "nt2":
                units = "dinucs"
            elif mode == "nt3":
                units = "trinucs"
            else:
                units = "nt"

            species_map_by_mode[mode][species_id] = SpeciesAssets(
                species_id=species_id,
                label=label,
                model_json=model_path,
                vocab_json=vocab_path,
                states_json=states_path,
                units=units,
            )

        # Scan 1-nt mode assets (states may be either *_nt_state_names.json or *_state_names.json)
        nt_suffix = "_mitoSpotter_hmm_1nt.json"
        for hmm_path in sorted(assets_dir.glob(f"*{nt_suffix}")):
            prefix = hmm_path.name.split(nt_suffix, 1)[0]
            if not prefix:
                continue
            vocab_path = assets_dir / f"{prefix}_nt_vocab.json"
            states_pref = assets_dir / f"{prefix}_nt_state_names.json"
            states_alt = assets_dir / f"{prefix}_state_names.json"
            states_path = states_pref if states_pref.exists() else states_alt
            add_species("nt1", prefix, model_path=hmm_path, vocab_path=vocab_path, states_path=states_path)

        # Scan 2-nt mode assets (prefer *_nt_2_vocab.json or *_nt2_vocab.json, fallback to *_nt_vocab.json)
        nt2_suffix = "_mitoSpotter_hmm_2nt.json"
        for hmm_path in sorted(assets_dir.glob(f"*{nt2_suffix}")):
            prefix = hmm_path.name.split(nt2_suffix, 1)[0]
            if not prefix:
                continue
            vocab_pref_a = assets_dir / f"{prefix}_nt_2_vocab.json"
            vocab_pref_b = assets_dir / f"{prefix}_nt2_vocab.json"
            vocab_fallback = assets_dir / f"{prefix}_nt_vocab.json"
            if vocab_pref_a.exists():
                vocab_path = vocab_pref_a
            elif vocab_pref_b.exists():
                vocab_path = vocab_pref_b
            else:
                vocab_path = vocab_fallback
            states_pref = assets_dir / f"{prefix}_nt_state_names.json"
            states_alt = assets_dir / f"{prefix}_state_names.json"
            states_path = states_pref if states_pref.exists() else states_alt
            add_species("nt2", prefix, model_path=hmm_path, vocab_path=vocab_path, states_path=states_path)

        # Scan 3-nt mode assets
        nt3_suffix = "_mitoSpotter_hmm_3nt.json"
        for hmm_path in sorted(assets_dir.glob(f"*{nt3_suffix}")):
            prefix = hmm_path.name.split(nt3_suffix, 1)[0]
            if not prefix:
                continue
            vocab_path = assets_dir / f"{prefix}_nt3_vocab.json"
            states_path = assets_dir / f"{prefix}_state_names.json"
            add_species("nt3", prefix, model_path=hmm_path, vocab_path=vocab_path, states_path=states_path)

        # Amino-acid mode assets are not used by the unified NT decoder/UI

        # If nothing found under project_root/out, fall back to scanning benchmark outputs
        if not species_map_by_mode["nt1"] and not species_map_by_mode["nt2"] and not species_map_by_mode["nt3"]:
            alt_root = project_root / "out_dir" / "04_model"
            if alt_root.exists():
                # Scan directories like: out_dir/04_model/1nt/<regime>/{model,vocab,states}.json
                for mode, unit_dir in ("nt1", "1nt"), ("nt2", "2nt"), ("nt3", "3nt"):
                    mode_dir = alt_root / unit_dir
                    if not mode_dir.exists():
                        continue
                    for regime_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
                        model_path = regime_dir / "model.json"
                        vocab_path = regime_dir / "vocab.json"
                        states_path = regime_dir / "states.json"
                        species_id = f"{unit_dir}-{regime_dir.name}"
                        # Label with unit and regime for clarity in the UI
                        if model_path.exists() and vocab_path.exists() and states_path.exists():
                            # Temporarily override label resolution inside add_species by passing species_id
                            add_species(mode, species_id, model_path=model_path, vocab_path=vocab_path, states_path=states_path)
                if species_map_by_mode["nt1"] or species_map_by_mode["nt2"] or species_map_by_mode["nt3"]:
                    assets_dir = alt_root  # reflect the actual assets root for any fallback usage

        if not species_map_by_mode["nt1"] and not species_map_by_mode["nt2"] and not species_map_by_mode["nt3"]:
            raise FileNotFoundError(
                f"No species asset bundles found in {project_root / 'out'} or {project_root / 'out_dir' / '04_model'}"
            )

        default_species_id_by_mode: Dict[str, str] = {}
        for mode, smap in species_map_by_mode.items():
            if smap:
                default_species_id_by_mode[mode] = sorted(smap.keys())[0]

        return cls(
            script_paths=script_paths,
            assets_dir=assets_dir,
            species_map_by_mode=species_map_by_mode,
            default_species_id_by_mode=default_species_id_by_mode,
        )

    def get_species(self, mode: str, species_id: Optional[str] = None) -> SpeciesAssets:
        if mode not in self.species_map_by_mode:
            raise KeyError(f"Unknown mode '{mode}'.")
        smap = self.species_map_by_mode[mode]
        if not smap:
            raise KeyError(f"No species available for mode '{mode}'.")
        target_id = species_id or self.default_species_id_by_mode.get(mode) or sorted(smap.keys())[0]
        if target_id not in smap:
            raise KeyError(f"Unknown species '{target_id}' for mode '{mode}'. Available: {sorted(smap)}")
        return smap[target_id]


class DecodeRunner:
    """Thin wrapper that invokes the existing mitoSpotter decode script."""

    def __init__(self, config: DecodeConfig) -> None:
        self.config = config
        for mode, path in self.config.script_paths.items():
            if not path.exists():
                # Allow missing script for a mode as long as assets also absent; skip strict check
                self.config.script_paths[mode] = path

        # basic validation of resources
        for mode_map in self.config.species_map_by_mode.values():
            for resources in mode_map.values():
                for path in (resources.model_json, resources.vocab_json, resources.states_json):
                    if not path.exists():
                        raise FileNotFoundError(f"Required resource missing: {path}")

    def run(
        self,
        *,
        sequences: Optional[Iterable[Tuple[str, str]]] = None,
        fasta_path: Optional[Path] = None,
        mode: str = "nt1",
        method: str = "viterbi",
        min_len: Optional[int] = None,
        emit_path: bool = False,
        workdir: Optional[Path] = None,
        species_id: Optional[str] = None,
        plotting: bool = False,
    ) -> Dict[str, List[Dict[str, object]]]:
        """Run the decode script and parse its TSV output (unified NT decoder)."""
        if not sequences and not fasta_path:
            raise ValueError("At least one of sequences or fasta_path must be provided.")

        mode = mode or "nt1"
        if mode not in ("nt1", "nt2", "nt3"):
            raise ValueError("mode must be 'nt1', 'nt2', or 'nt3'")

        method = method or "viterbi"
        if method not in ("viterbi", "posterior"):
            raise ValueError("method must be 'viterbi' or 'posterior'")

        assets = self.config.get_species(mode, species_id)
        script_path = self.config.script_paths[mode]
        if not script_path.exists():
            raise FileNotFoundError(f"Decode script missing for mode '{mode}': {script_path}")

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            effective_fasta = fasta_path

            if sequences:
                seq_file = tmpdir / "input_sequences.fa"
                with seq_file.open("w", encoding="ascii") as fh:
                    for idx, (seq_id, seq) in enumerate(sequences, start=1):
                        header = seq_id or f"seq{idx}"
                        fh.write(f">{header}\n")
                        fh.write(f"{seq.strip()}\n")
                effective_fasta = seq_file

            output_tsv = tmpdir / "decode_output.tsv"

            # Build candidate command lists in order of preference
            base_ngram = [
                "python",
                str(script_path),
                "--ngram",
                "1" if mode == "nt1" else ("2" if mode == "nt2" else "3"),
            ]

            explicit_base = base_ngram + [
                "--model_json", str(assets.model_json),
                "--vocab_json", str(assets.vocab_json),
                "--states_json", str(assets.states_json),
                "--out_tsv", str(output_tsv),
            ]
            explicit_with_method = base_ngram + ["--method", method] + explicit_base[len(base_ngram):]

            fallback_base = base_ngram + [
                "--species", assets.species_id,
                "--assets_dir", str(self.config.assets_dir),
                "--out_tsv", str(output_tsv),
            ]
            fallback_with_method = base_ngram + ["--method", method] + fallback_base[len(base_ngram):]

            def add_common(cmd_list: List[str]) -> List[str]:
                out = cmd_list[:]
                if emit_path:
                    out.append("--emit_path")
                if effective_fasta is not None:
                    out.extend(["--fasta", str(effective_fasta)])
                if plotting:
                    out.append("--plotting")
                return out

            candidates = [
                add_common(explicit_with_method),
                add_common(explicit_base),
                add_common(fallback_with_method),
                add_common(fallback_base),
            ]

            def _run_or_raise(args_list):
                return subprocess.run(
                    args_list,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=str(workdir or script_path.parent.parent),
                )

            last_err = None
            completed = None
            for c in candidates:
                try:
                    completed = _run_or_raise(c)
                    break
                except subprocess.CalledProcessError as e:
                    last_err = e
            else:
                msg = (last_err.stderr if last_err else "") or "decoder failed"
                raise RuntimeError(f"Decoder failed. Last error: {msg.strip()}")

            records: List[Dict[str, object]] = []
            paths: List[Dict[str, object]] = []

            with output_tsv.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    if line.startswith("#"):
                        # Path lines example: #{id}_PATH\tstate_id state_id ...
                        if "\t" in line:
                            key, rest = line[1:].split("\t", 1)
                            if key.endswith("_PATH"):
                                seq_id = key[:-5]
                                states = rest.strip().split()
                                paths.append({"id": seq_id, "states": states})
                        continue
                    row = line.split("\t")
                    # Support alternate PATH row format: id\tPATH\t...states...
                    if len(row) >= 2 and row[1] == "PATH":
                        seq_id = row[0]
                        states = []
                        if len(row) >= 3:
                            states = (row[-1] if len(row) == 3 else " ".join(row[2:])).split()
                        paths.append({"id": seq_id, "states": states})
                        continue
                    # Expect results row; be tolerant of column counts
                    if len(row) < 5:
                        continue
                    rid = row[0]
                    # Column 1 may be loglik/logprob
                    try:
                        loglik = float(row[1])
                    except ValueError:
                        # if header row or non-numeric, skip
                        continue
                    # nuclear/mito fractions may be at 2/3, with call then lengths
                    def _to_float(s: str, default: float = 0.0) -> float:
                        try:
                            return float(s)
                        except Exception:
                            return default
                    nuclear_frac = _to_float(row[2], 0.0) if len(row) > 2 else 0.0
                    mito_frac = _to_float(row[3], 0.0) if len(row) > 3 else 0.0
                    call = row[4] if len(row) > 4 and row[4] else ("mitochondrial" if mito_frac >= nuclear_frac else "nuclear")
                    # Attempt to read sequence and token lengths if present
                    seq_len_nt = None
                    token_count = None
                    if len(row) > 5:
                        try:
                            seq_len_nt = int(row[5])
                        except Exception:
                            seq_len_nt = None
                    if len(row) > 6:
                        try:
                            token_count = int(row[6])
                        except Exception:
                            token_count = None

                    rec = {
                        "id": rid,
                        "loglik": loglik,
                        "logprob": loglik,  # backwards-compat field name
                        "winner": call,
                        "nuclear_frac": nuclear_frac,
                        "mito_frac": mito_frac,
                        "length": seq_len_nt if seq_len_nt is not None else (token_count or 0),
                        "len_nt": seq_len_nt if seq_len_nt is not None else 0,
                        "tokens": token_count if token_count is not None else 0,
                        "units": assets.units,
                    }
                    # Optional convenience for UI header
                    if assets.units == "dinucs":
                        rec["len_dinuc"] = token_count
                    elif assets.units == "trinucs":
                        rec["len_trinuc"] = token_count
                    records.append(rec)

            # Attempt to parse generated plot files from decoder stderr when plotting is enabled
            plot_files: List[str] = []
            plot_dir: Optional[str] = None
            if plotting and completed is not None:
                stderr_text = completed.stderr or ""
                for line in stderr_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Detect the plot directory line
                    if line.startswith("[INFO] Generating plots in:"):
                        # e.g., "[INFO] Generating plots in: Plot_res_20231231_235959"
                        try:
                            plot_dir = line.split(":", 1)[1].strip()
                        except Exception:
                            plot_dir = None
                    # Detect individual generated file lines: "- Plot_res_.../file.png"
                    if line.startswith("-"):
                        path = line.lstrip("- ")
                        if path.lower().endswith((".png", ".jpg", ".jpeg")):
                            plot_files.append(path)

                # Fallback: if we have a directory but no files parsed, scan it
                if plot_dir and not plot_files:
                    # Decoder ran with cwd set to project root by default
                    project_root = script_path.parent.parent
                    dir_path = (workdir or project_root) / plot_dir
                    try:
                        for p in sorted(dir_path.glob("*.png")):
                            try:
                                rel = str(p.relative_to(project_root))
                            except Exception:
                                rel = str(p.name)
                            plot_files.append(rel)
                    except Exception:
                        pass

            return {
                "records": records,
                "paths": paths,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "species_id": assets.species_id,
                "mode": mode,
                "method": method,
                "plots": plot_files,
                "plot_dir": plot_dir or "",
            }
