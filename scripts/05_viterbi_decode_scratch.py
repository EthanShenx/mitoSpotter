#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Bio import SeqIO
import argparse, os.path as op, json, re, sys, numpy as np

# --- Genetic code differences ---
STD_STOPS = {"TAA","TAG","TGA"}
VERTEBRATE_MITO_STOPS = {"TAA","TAG","AGA","AGG"}

# --- ORF & utils ---

def longest_orf(seq, stops, min_nt=150):
    # Scan 3 forward frames; keep the longest ORF sans stop
    s = str(seq).upper().replace("U","T")
    best = ""
    for frame in (0,1,2):
        start = frame; i = frame
        while i+3 <= len(s):
            c = s[i:i+3]
            if not set(c) <= set("ACGT"):
                if i-start > len(best): best = s[start:i]
                start = i + 3; i += 3; continue
            if c in stops:
                if i-start > len(best): best = s[start:i]
                start = i + 3
            i += 3
        tail = s[start: len(s) - ((len(s)-start)%3)]
        if len(tail) > len(best): best = tail
    return best if len(best) >= min_nt else ""

def nt_to_codons(nt):
    return [nt[i:i+3] for i in range(0, len(nt)-2, 3)]

def load_model(model_json):
    M = json.load(open(model_json))
    pi = np.array(M["startprob"], dtype=np.float64)
    A  = np.array(M["transmat"], dtype=np.float64)
    B  = np.array(M["emissionprob"], dtype=np.float64)
    codons = M["codon_order"]
    idx = {c:i for i,c in enumerate(codons)}
    return pi, A, B, idx

def load_state_names(states_json):
    # Expect {"0":"nuclear","1":"mitochondrial"}
    S = json.load(open(states_json))
    as_int = {int(k): v for k, v in S.items()}
    inv = {v: k for k, v in as_int.items()}
    nuc_id  = inv.get("nuclear", 0)
    mito_id = inv.get("mitochondrial", 1)
    return as_int, nuc_id, mito_id

# --- HMM ops (mirror training) ---

def viterbi(pi, A, B, obs):
    # Log-space Viterbi
    T = len(obs); N = A.shape[0]
    logA = np.log(A + 1e-300)
    logB = np.log(B + 1e-300)
    logpi = np.log(pi + 1e-300)
    d = np.zeros((T,N), dtype=np.float64)
    psi = np.zeros((T,N), dtype=np.int64)
    d[0] = logpi + logB[:, obs[0]]; psi[0] = -1
    for t in range(1, T):
        for j in range(N):
            s = d[t-1] + logA[:, j]
            psi[t, j] = int(np.argmax(s))
            d[t, j] = s[psi[t, j]] + logB[j, obs[t]]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(d[-1]))
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return float(d[-1, path[-1]]), path

def forward_loglik(pi, A, B, obs):
    # Scaled forward -> log-likelihood
    T = len(obs)
    alpha = np.zeros((T, A.shape[0]), dtype=np.float64)
    c = np.zeros(T, dtype=np.float64)
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum() or 1e-300
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum() or 1e-300
        alpha[t] /= c[t]
    return -float(np.sum(np.log(c + 1e-300)))

def summarize_path(path, nuc_id=0, mito_id=1, state_names=None):
    # Report fractions using IDs; map winner to name if given
    total = len(path)
    if total == 0:
        return {"nuclear_frac":0.0, "mito_frac":0.0, "winner":"NA"}
    nuc = np.mean(path==nuc_id); mito = np.mean(path==mito_id)
    winner_id = nuc_id if nuc >= mito else mito_id
    if state_names and winner_id in state_names:
        winner = state_names[winner_id]
    else:
        winner = "nuclear" if winner_id==nuc_id else "mitochondrial"
    return {"nuclear_frac":round(float(nuc),4),
            "mito_frac":round(float(mito),4),
            "winner":winner}

# --- Input multiplexing ---

def iter_inputs_from_cli(args):
    """Yield (id, seq_string) from --seq/--stdin/--fasta."""
    if args.seq:
        for i, s in enumerate(args.seq, start=1):
            sid = (args.seq_id[i-1] if args.seq_id and i<=len(args.seq_id) else f"seq{i}")
            yield (sid, s)
    if args.stdin:
        raw = sys.stdin.read()
        if raw:
            s = "".join(raw.split())
            sid = args.stdin_id or "stdin_seq"
            yield (sid, s)
    if args.fasta:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            yield (rec.id, str(rec.seq))

def must_have_inputs(args):
    return bool(args.fasta or args.seq or args.stdin)

# --- Species asset resolver ---

def resolve_assets(species, assets_dir):
    """Resolve model/vocab/state files for a species code."""
    model_candidates = [
        op.join(assets_dir, f"{species}_mitoSpotter_hmm_codon.json"),
        op.join(assets_dir, f"{species}_mitoSpotter_hmm.json"),
    ]
    model_json = next((p for p in model_candidates if op.exists(p)), None)
    if model_json is None:
        raise SystemExit(
            f"Asset missing for species '{species}': expected one of {[op.basename(p) for p in model_candidates]}"
        )

    vocab_json  = op.join(assets_dir, f"{species}_codon_vocab.json")
    states_json = op.join(assets_dir, f"{species}_state_names.json")
    for p in (model_json, vocab_json, states_json):
        if not op.exists(p):
            raise SystemExit(f"Asset missing for species '{species}': {p}")
    return model_json, vocab_json, states_json

# --- CLI ---

def main():
    ap = argparse.ArgumentParser(
        description="Viterbi decode with mitoSpotter HMM (from scratch). "
                    "Inputs: --fasta or repeated --seq or --stdin. "
                    "Species is REQUIRED to pick assets."
    )
    # species is mandatory
    ap.add_argument("--species", choices=["hs","mm","rn"], required=True,
                    help="Species code: hs=human, mm=mouse, rn=rat (REQUIRED).")
    ap.add_argument(
        "--assets_dir",
        default="out",
        help=(
            "Dir holding <sp>_mitoSpotter_hmm_codon.json (or legacy _mitoSpotter_hmm.json), "
            "*_codon_vocab.json, *_state_names.json."
        ),
    )
    # inputs
    ap.add_argument("--fasta", help="FASTA file of sequences (optional)")
    ap.add_argument("--seq", action="append",
                    help="Inline nucleotide sequence (A/C/G/T/U); can repeat.")
    ap.add_argument("--seq_id", action="append",
                    help="ID(s) for --seq; align by order. Optional.")
    ap.add_argument("--stdin", action="store_true",
                    help="Read a single sequence from STDIN (whitespace stripped).")
    ap.add_argument("--stdin_id", default="stdin_seq",
                    help="ID for the STDIN sequence (default: stdin_seq).")
    # decoding opts
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--code", choices=["standard","vertebrate_mito","auto"], default="auto")
    ap.add_argument("--min_orf_nt", type=int, default=150)
    ap.add_argument("--emit_path", action="store_true")
    args = ap.parse_args()

    if not must_have_inputs(args):
        raise SystemExit("Please provide at least one of --fasta, --seq, or --stdin.")

    model_p, vocab_p, states_p = resolve_assets(args.species, args.assets_dir)
    pi, A, B, idx = load_model(model_p)
    state_names, nuc_id, mito_id = load_state_names(states_p)

    with open(args.out_tsv, "w") as fo:
        fo.write("#id\tlogprob\twinner\tnuclear_frac\tmito_frac\tlen_codons\n")
        for rec_id, raw_seq in iter_inputs_from_cli(args):
            candidates = []
            for code in (["standard","vertebrate_mito"] if args.code=="auto" else [args.code]):
                stops = STD_STOPS if code=="standard" else VERTEBRATE_MITO_STOPS
                orf = longest_orf(raw_seq, stops, min_nt=args.min_orf_nt)
                if not orf: 
                    continue
                cods = [c for c in nt_to_codons(orf) if re.fullmatch(r"[ACGT]{3}", c)]
                obs = np.array([idx[c] for c in cods if c in idx], dtype=np.int64)
                if obs.size == 0: 
                    continue
                ll = forward_loglik(pi, A, B, obs)
                vll, path = viterbi(pi, A, B, obs)
                candidates.append((ll, vll, path, len(obs), code))

            if not candidates:
                continue
            # Choose by forward log-likelihood; report Viterbi path of the winner
            best = max(candidates, key=lambda t: t[0])
            ll, vll, path, L, code = best
            summ = summarize_path(path, nuc_id=nuc_id, mito_id=mito_id, state_names=state_names)
            fo.write(f"{rec_id}\t{ll:.3f}\t{summ['winner']}\t{summ['nuclear_frac']}\t{summ['mito_frac']}\t{L}\n")
            if args.emit_path:
                fo.write(f"{rec_id}\tPATH\t{code}\t{' '.join(map(str, path.tolist()))}\n")
    print(f"[OK] decode -> {args.out_tsv}")

if __name__ == "__main__":
    main()
