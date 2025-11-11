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
    """Holds script paths, assets dir, and available species bundles per mode."""

    script_paths: Dict[str, Path]  # {"codon": path, "nt1": path, "nt2": path, "nt3": path, "aa": path}
    assets_dir: Path
    species_map_by_mode: Dict[str, Dict[str, SpeciesAssets]]  # mode -> {species_id: assets}
    default_species_id_by_mode: Dict[str, str]

    @classmethod
    def with_project_defaults(cls, project_root: Path) -> "DecodeConfig":
        """Create a config by scanning the default project layout."""

        script_paths = {
            "codon": project_root / "scripts" / "05_viterbi_decode_scratch.py",
            "nt1": project_root / "scripts" / "05_viterbi_decode_scratch_1_nt.py",
            "nt2": project_root / "scripts" / "05_viterbi_decode_scratch_2_nt.py",
            "nt3": project_root / "scripts" / "05_viterbi_decode_scratch_3_nt.py",
            "aa": project_root / "scripts" / "05_viterbi_decode_scratch_aa.py",
        }
        assets_dir = project_root / "out"

        species_map_by_mode: Dict[str, Dict[str, SpeciesAssets]] = {"codon": {}, "nt1": {}, "nt2": {}, "nt3": {}, "aa": {}}

        def add_species(mode: str, species_id: str, model_path: Path, vocab_path: Path, states_path: Path) -> None:
            if not (model_path.exists() and vocab_path.exists() and states_path.exists()):
                return
            label = SPECIES_LABELS.get(species_id, species_id.upper())
            units = "codons"
            if mode == "nt1":
                units = "nt"
            elif mode == "nt2":
                units = "dinucs"
            elif mode == "nt3":
                units = "trinucs"
            elif mode == "aa":
                units = "aa"

            species_map_by_mode[mode][species_id] = SpeciesAssets(
                species_id=species_id,
                label=label,
                model_json=model_path,
                vocab_json=vocab_path,
                states_json=states_path,
                units=units,
            )

        # Scan codon-mode assets
        hmm_suffix = "_mitoSpotter_hmm_codon.json"
        legacy_suffix = "_mitoSpotter_hmm.json"
        for hmm_path in sorted(assets_dir.glob(f"*{hmm_suffix}")):
            prefix = hmm_path.name.split(hmm_suffix, 1)[0]
            if prefix:
                add_species(
                    "codon",
                    prefix,
                    model_path=hmm_path,
                    vocab_path=assets_dir / f"{prefix}_codon_vocab.json",
                    states_path=assets_dir / f"{prefix}_state_names.json",
                )
        # Legacy support
        for hmm_path in sorted(assets_dir.glob(f"*{legacy_suffix}")):
            prefix = hmm_path.name.split(legacy_suffix, 1)[0]
            if prefix and prefix not in species_map_by_mode["codon"]:
                add_species(
                    "codon",
                    prefix,
                    model_path=hmm_path,
                    vocab_path=assets_dir / f"{prefix}_codon_vocab.json",
                    states_path=assets_dir / f"{prefix}_state_names.json",
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

        # Scan amino-acid mode assets
        aa_suffix = "_mitoSpotter_hmm_aa.json"
        for hmm_path in sorted(assets_dir.glob(f"*{aa_suffix}")):
            prefix = hmm_path.name.split(aa_suffix, 1)[0]
            if not prefix:
                continue
            vocab_path = assets_dir / f"{prefix}_aa_vocab.json"
            states_path = assets_dir / f"{prefix}_state_names.json"
            add_species("aa", prefix, model_path=hmm_path, vocab_path=vocab_path, states_path=states_path)

        if not species_map_by_mode["codon"] and not species_map_by_mode["nt1"] and not species_map_by_mode["nt2"]:
            raise FileNotFoundError(f"No species asset bundles found in {assets_dir}")

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
        mode: str = "codon",
        code: str = "auto",
        min_orf_nt: int = 150,
        emit_path: bool = False,
        workdir: Optional[Path] = None,
        species_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, object]]]:
        """Run the decode script and parse its TSV output."""
        if not sequences and not fasta_path:
            raise ValueError("At least one of sequences or fasta_path must be provided.")

        mode = mode or "codon"
        if mode not in ("codon", "nt1", "nt2", "nt3", "aa"):
            raise ValueError("mode must be 'codon', 'nt1', 'nt2', 'nt3', or 'aa'")

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

            cmd = ["python", str(script_path), "--species", assets.species_id, "--assets_dir", str(self.config.assets_dir), "--out_tsv", str(output_tsv)]

            # Common options
            cmd.extend(["--code", code])

            if mode == "codon":
                cmd.extend(["--min_orf_nt", str(min_orf_nt)])
            elif mode == "nt1":
                # No ORF limits for nt1: analyze full cleaned sequence
                cmd.extend(["--segment", "cds", "--min_len", "0"])
            elif mode == "nt2":
                # Mirror nt1 behavior: analyze full cleaned sequence, no ORF constraints
                cmd.extend(["--segment", "cds", "--min_len", "0"])
            elif mode == "nt3":
                # Mirror nt1/nt2 behavior: analyze full cleaned sequence, no ORF constraints
                cmd.extend(["--segment", "cds", "--min_len", "0"])
            else:  # aa
                # AA mode uses nt threshold; default to ORF segmentation like codon
                cmd.extend(["--segment", "orf", "--min_nt", str(min_orf_nt)])

            if emit_path:
                cmd.append("--emit_path")

            if effective_fasta is not None:
                cmd.extend(["--fasta", str(effective_fasta)])

            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(workdir or script_path.parent.parent),
            )

            records: List[Dict[str, object]] = []
            paths: List[Dict[str, object]] = []

            with output_tsv.open("r", encoding="utf-8") as fh:
                reader = csv.reader(fh, delimiter="\t")
                for row in reader:
                    if not row or row[0].startswith("#"):
                        continue
                    if len(row) >= 2 and row[1] == "PATH":
                        paths.append(
                            {
                                "id": row[0],
                                "context": row[2] if len(row) > 2 else "",
                                "states": row[3].split() if len(row) > 3 else [],
                            }
                        )
                        continue
                    if len(row) < 6:
                        continue
                    length_val = int(row[5])
                    rec = {
                        "id": row[0],
                        "logprob": float(row[1]),
                        "winner": row[2],
                        "nuclear_frac": float(row[3]),
                        "mito_frac": float(row[4]),
                        "length": length_val,
                        "units": assets.units,
                    }
                    # Backwards-compatible fields for existing UI
                    if assets.units == "codons":
                        rec["len_codons"] = length_val
                    elif assets.units == "nt":
                        rec["len_nt"] = length_val
                    elif assets.units == "aa":
                        rec["len_aa"] = length_val
                    elif assets.units == "dinucs":
                        rec["len_dinuc"] = length_val
                    elif assets.units == "trinucs":
                        rec["len_trinuc"] = length_val
                    else:
                        pass
                    records.append(rec)

            return {
                "records": records,
                "paths": paths,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "species_id": assets.species_id,
                "mode": mode,
            }
