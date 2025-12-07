#!/usr/bin/env python3
"""
HMM Viterbi decoder for nuclear vs mitochondrial classification.

Accepts explicit --model_json, --vocab_json, --states_json paths
for benchmarking custom-trained models.
"""

import argparse
import json
import os.path as op
import re
import sys
import time
from datetime import datetime
import numpy as np
from Bio import SeqIO

# Optional memory tracking - only used when explicitly requested
try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False

# Nucleotide vocabulary
BASES = ["A", "C", "G", "T"]


# -------------------- Timing and Memory Utilities -------------------- #

def format_timestamp(ts):
    """Format a timestamp as a human-readable string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds):
    """Format duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} min {secs:.2f} sec"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)} hr {int(minutes)} min {secs:.2f} sec"


def format_memory(bytes_value):
    """Format memory size in bytes to a human-readable string."""
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.2f} KB"
    elif bytes_value < 1024 ** 3:
        return f"{bytes_value / (1024 ** 2):.2f} MB"
    else:
        return f"{bytes_value / (1024 ** 3):.2f} GB"


def report_timing(start_time, end_time):
    """Print timing report to stderr."""
    duration = end_time - start_time
    print("\n" + "=" * 60, file=sys.stderr)
    print("TIMING REPORT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Start time : {format_timestamp(start_time)}", file=sys.stderr)
    print(f"  End time   : {format_timestamp(end_time)}", file=sys.stderr)
    print(f"  Total time : {format_duration(duration)}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def report_memory(peak_memory):
    """Print memory report to stderr."""
    print("\n" + "-" * 60, file=sys.stderr)
    print("MEMORY REPORT", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print(f"  Peak memory usage : {format_memory(peak_memory)}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)


# -------------------- Sequence Processing -------------------- #

def clean_nt(s):
    """Clean nucleotide string: uppercase, replace U->T, keep only A/C/G/T."""
    return re.sub(r"[^ACGT]", "", str(s).upper().replace("U", "T"))


def load_model(model_json, vocab_json):
    """Load HMM parameters pi, A, B from JSON."""
    M = json.load(open(model_json))
    pi = np.array(M["startprob"], dtype=np.float64)
    A = np.array(M["transmat"], dtype=np.float64)
    B = np.array(M["emissionprob"], dtype=np.float64)
    
    # Load vocabulary mapping
    vocab_data = json.load(open(vocab_json))
    if "bases" in vocab_data:  # 1-nt model
        idx = {b: i for i, b in enumerate(vocab_data["bases"])}
    elif "dinucs" in vocab_data:  # 2-nt model
        idx = {d: i for i, d in enumerate(vocab_data["dinucs"])}
    elif "trinucs" in vocab_data:  # 3-nt model
        idx = {t: i for i, t in enumerate(vocab_data["trinucs"])}
    else:  # Fallback
        idx = vocab_data.get("index", {})

    return pi, A, B, idx


def load_states(p):
    """Load state name mapping, and resolve nuclear/mitochondrial IDs."""
    S = json.load(open(p))
    S = {int(k): v for k, v in S.items()}
    inv = {v: k for k, v in S.items()}
    nuc_id = inv.get("nuclear", 0)
    mito_id = inv.get("mitochondrial", 1)
    return S, nuc_id, mito_id


# -------------------- HMM Algorithms -------------------- #

def viterbi(pi, A, B, obs):
    """
    Viterbi decoding: find most likely state path.
    
    Returns:
        viterbi_ll: log-likelihood of best path
        path: state sequence
    """
    T = len(obs)
    N = A.shape[0]
    logA = np.log(A + 1e-300)
    logB = np.log(B + 1e-300)
    logpi = np.log(pi + 1e-300)
    
    d = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=np.int64)
    
    d[0] = logpi + logB[:, obs[0]]
    psi[0] = -1
    
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


def forward_ll(pi, A, B, obs):
    """
    Forward algorithm with scaling to compute log-likelihood.
    
    Returns:
        ll: log-likelihood of the observation sequence
    """
    T = len(obs)
    N = A.shape[0]
    
    alpha = np.zeros((T, N), dtype=np.float64)
    c = np.zeros(T, dtype=np.float64)
    
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum() or 1e-300
    alpha[0] /= c[0]
    
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum() or 1e-300
        alpha[t] /= c[t]
    
    return float(np.sum(np.log(c + 1e-300)))


# -------------------- Summary Functions -------------------- #

def summarize_viterbi(path, nuc_id=0, mito_id=1, state_names=None):
    """
    Summarize Viterbi path with winner call.
    
    Returns:
        dict with nuclear_frac, mito_frac, call (winner)
    """
    if len(path) == 0:
        return {"nuclear_frac": 0.0, "mito_frac": 0.0, "call": "NA"}
    
    nuc = float(np.mean(path == nuc_id))
    mito = float(np.mean(path == mito_id))
    winner_id = nuc_id if nuc >= mito else mito_id
    
    name = state_names.get(
        winner_id,
        "nuclear" if winner_id == nuc_id else "mitochondrial"
    ) if state_names else (
        "nuclear" if winner_id == nuc_id else "mitochondrial"
    )
    
    return {
        "nuclear_frac": round(nuc, 4),
        "mito_frac": round(mito, 4),
        "call": name
    }


# -------------------- Input Iteration -------------------- #

def iter_inputs(args):
    """Iterate over all input sequences based on CLI options."""
    if args.seq:
        for i, s in enumerate(args.seq, 1):
            sid = (
                args.seq_id[i-1]
                if args.seq_id and i <= len(args.seq_id)
                else f"seq{i}"
            )
            yield sid, s
    
    if args.stdin:
        raw = sys.stdin.read()
        if raw:
            yield args.stdin_id, raw
    
    if args.fasta:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            yield rec.id, str(rec.seq)


def sequence_to_observations(sequence, ngram, vocab_idx):
    """
    Convert sequence to observation indices based on ngram size.
    
    Returns:
        obs: numpy array of observation indices
        token_count: number of tokens generated
        nt_length: original nucleotide length (after cleaning)
    """
    s = clean_nt(sequence)
    nt_length = len(s)
    
    if ngram == 1:
        tokens = list(s)
    elif ngram == 2:
        tokens = [s[i:i+2] for i in range(len(s)-1)]
    elif ngram == 3:
        L = (len(s)//3) * 3
        tokens = [s[i:i+3] for i in range(0, L, 3)]
    else:
        tokens = list(s)

    obs = [vocab_idx[t] for t in tokens if t in vocab_idx]
    return np.array(obs, dtype=np.int64), len(tokens), nt_length


# -------------------- Main -------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=(
            "HMM Viterbi decoder for nuclear vs mitochondrial classification. "
            "Requires explicit model path options."
        )
    )
    
    # Core parameters
    ap.add_argument("--ngram", type=int, choices=[1,2,3], default=1)
    
    # Model loading - explicit paths required
    ap.add_argument("--model_json", required=True,
                    help="Path to model JSON file")
    ap.add_argument("--vocab_json", required=True,
                    help="Path to vocabulary JSON file")
    ap.add_argument("--states_json", required=True,
                    help="Path to states JSON file")
    
    # Input options
    ap.add_argument("--fasta")
    ap.add_argument("--seq", action="append")
    ap.add_argument("--seq_id", action="append")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--stdin_id", default="stdin_seq")
    
    # Sequence processing
    ap.add_argument(
        "--min_len", 
        type=int, 
        default=150, 
        help="Minimum sequence length in nucleotides (applied before tokenization)"
    )
    
    # Output
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument(
        "--emit_path",
        action="store_true",
        help="Emit decoded path per sequence."
    )
    
    # Memory tracking option - only enabled when explicitly set
    ap.add_argument(
        "--track_memory",
        action="store_true",
        help="Enable memory tracking and report peak memory usage after decoding. "
             "Requires tracemalloc module (included in Python standard library)."
    )
    
    args = ap.parse_args()
    
    if not (args.fasta or args.seq or args.stdin):
        raise SystemExit("Provide input via --fasta or --seq (repeatable) or --stdin.")
    
    # Verify model files exist
    for p in (args.model_json, args.vocab_json, args.states_json):
        if not op.exists(p):
            raise SystemExit(f"File not found: {p}")
    
    # -------------------- Start timing -------------------- #
    start_time = time.time()
    
    # -------------------- Start memory tracking if requested -------------------- #
    if args.track_memory:
        if TRACEMALLOC_AVAILABLE:
            tracemalloc.start()
            print("[INFO] Memory tracking enabled.", file=sys.stderr)
        else:
            print("[WARNING] tracemalloc not available. Memory tracking disabled.", file=sys.stderr)
            args.track_memory = False
    
    # Load model and assets
    pi, A, B, vocab_idx = load_model(args.model_json, args.vocab_json)
    state_names, nuc_id, mito_id = load_states(args.states_json)
    
    # Track processing statistics
    n_processed = 0
    n_skipped = 0
    
    with open(args.out_tsv, "w") as fo:
        
        fo.write(f"#id\tloglik\tnuclear_frac\tmito_frac\tcall\tseq_len_nt\tn_tokens_{args.ngram}mer\n")
        
        for rid, raw in iter_inputs(args):
            obs, token_count, nt_length = sequence_to_observations(raw, args.ngram, vocab_idx)
            
            # Apply min_len filter to nucleotide length
            if nt_length < args.min_len:
                n_skipped += 1
                continue
        
            if obs.size == 0:
                n_skipped += 1
                continue
            
            n_processed += 1
            
            ll = forward_ll(pi, A, B, obs)
            _, path = viterbi(pi, A, B, obs)
            
            summ = summarize_viterbi(path, nuc_id, mito_id, state_names)
            
            fo.write(
                f"{rid}\t{ll:.3f}\t"
                f"{summ['nuclear_frac']}\t{summ['mito_frac']}\t"
                f"{summ['call']}\t{nt_length}\t{token_count}\n"
            )
            
            if args.emit_path:
                path_str = " ".join(map(str, path.tolist()))
                fo.write(f"#{rid}_PATH\t{path_str}\n")
    
    # -------------------- End timing -------------------- #
    end_time = time.time()
    
    # Print summary
    print(f"[OK] Viterbi decode -> {args.out_tsv}")
    print(f"[OK] Processed: {n_processed} sequences, Skipped: {n_skipped} sequences")
    
    # -------------------- Report memory usage if tracking was enabled -------------------- #
    if args.track_memory:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report_memory(peak)
    
    # -------------------- Always report timing -------------------- #
    report_timing(start_time, end_time)


if __name__ == "__main__":
    main()