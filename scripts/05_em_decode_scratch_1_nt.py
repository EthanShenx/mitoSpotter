#!/usr/bin/env python3
# Posterior decoding with 1-nt HMM (nuclear vs mitochondrial).
# Uses forward-backward instead of pure Viterbi fractions and
# optionally produces a PDF visualization of per-position posteriors.

import argparse
import json
import os.path as op
import re
import sys
import numpy as np
from Bio import SeqIO

import matplotlib
# Use non-interactive backend for CLI environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set global font to Arial (will fall back if Arial is not available)
matplotlib.rcParams["font.family"] = "Arial"

# Nucleotide vocabulary
BASES = ["A", "C", "G", "T"]
IDX = {b: i for i, b in enumerate(BASES)}


def clean_nt(s):
    """Clean nucleotide string: uppercase, replace U->T, keep only A/C/G/T."""
    return re.sub(r"[^ACGT]", "", str(s).upper().replace("U", "T"))


def resolve_assets(species, assets_dir):
    """Resolve model/vocab/state files for a given species."""
    model = op.join(assets_dir, f"{species}_mitoSpotter_hmm_1nt.json")
    vocab = op.join(assets_dir, f"{species}_nt_vocab.json")
    states = op.join(assets_dir, f"{species}_state_names.json")
    for p in (model, vocab, states):
        if not op.exists(p):
            raise SystemExit(f"Missing asset: {p}")
    return model, vocab, states


def load_model(model_json):
    """Load HMM parameters pi, A, B from JSON."""
    M = json.load(open(model_json))
    pi = np.array(M["startprob"], dtype=np.float64)
    A = np.array(M["transmat"], dtype=np.float64)
    B = np.array(M["emissionprob"], dtype=np.float64)
    return pi, A, B


def load_states(p):
    """Load state name mapping, and resolve nuclear/mitochondrial IDs."""
    S = json.load(open(p))
    S = {int(k): v for k, v in S.items()}
    inv = {v: k for k, v in S.items()}
    nuc_id = inv.get("nuclear", 0)
    mito_id = inv.get("mitochondrial", 1)
    return S, nuc_id, mito_id


def forward_backward(pi, A, B, obs):
    """
    Forward-backward with scaling.

    Returns:
        loglik: log-likelihood of the sequence
        gamma: posterior probabilities, shape (T, N)
    """
    T = len(obs)
    N = A.shape[0]

    if T == 0:
        raise ValueError("Empty observation sequence is not allowed.")

    # Forward pass with scaling
    alpha = np.zeros((T, N), dtype=np.float64)
    c = np.zeros(T, dtype=np.float64)

    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum()
    if c[0] <= 0.0:
        c[0] = 1e-300
    alpha[0] /= c[0]

    for t in range(1, T):
        # alpha[t-1] @ A -> shape (N,)
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum()
        if c[t] <= 0.0:
            c[t] = 1e-300
        alpha[t] /= c[t]

    loglik = -float(np.sum(np.log(c)))

    # Backward pass with same scaling
    beta = np.zeros((T, N), dtype=np.float64)
    beta[-1] = 1.0 / c[-1]

    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
        beta[t] /= c[t]

    # Posterior probabilities
    gamma = alpha * beta
    gamma_sum = gamma.sum(axis=1, keepdims=True)
    gamma_sum[gamma_sum == 0.0] = 1e-300
    gamma /= gamma_sum

    return loglik, gamma


def posterior_decode(gamma):
    """Posterior decoding: argmax over states for each position."""
    if gamma.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.argmax(gamma, axis=1).astype(np.int64)


def summarize_posterior(
    gamma,
    nuc_id=0,
    mito_id=1,
    state_names=None,
    hi_thresh=0.8,
    margin=0.2,
):
    """
    Summarize posterior over nuclear/mitochondrial with an ambiguous zone.

    Returns:
        dict with nuclear_frac, mito_frac, call
        call in {"nuclear", "mitochondrial", "ambiguous"}
    """
    if gamma.size == 0:
        return {"nuclear_frac": 0.0, "mito_frac": 0.0, "call": "ambiguous"}

    nuc_frac = float(gamma[:, nuc_id].mean())
    mito_frac = float(gamma[:, mito_id].mean())

    # Default call is ambiguous
    call_label = "ambiguous"
    call_id = None

    # High-confidence mitochondrial
    if (mito_frac >= hi_thresh) and (mito_frac - nuc_frac >= margin):
        call_id = mito_id
        call_label = "mitochondrial"

    # High-confidence nuclear
    elif (nuc_frac >= hi_thresh) and (nuc_frac - mito_frac >= margin):
        call_id = nuc_id
        call_label = "nuclear"

    # Optional: if you want to map call_id to custom state_names, you can do it here.
    if call_id is not None and state_names:
        # For now we keep generic labels "nuclear"/"mitochondrial".
        pass

    return {
        "nuclear_frac": round(nuc_frac, 4),
        "mito_frac": round(mito_frac, 4),
        "call": call_label,
    }


def iter_inputs(args):
    """Iterate over all input sequences based on CLI options."""
    # From --seq
    if args.seq:
        for i, s in enumerate(args.seq, 1):
            sid = (
                args.seq_id[i - 1]
                if args.seq_id and i <= len(args.seq_id)
                else f"seq{i}"
            )
            yield sid, s

    # From --stdin
    if args.stdin:
        raw = sys.stdin.read()
        if raw:
            yield args.stdin_id, raw

    # From --fasta
    if args.fasta:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            yield rec.id, str(rec.seq)


def make_posterior_figure(gamma, nuc_id, mito_id, seq_id):
    """
    Create a stacked bar chart figure for posterior N/M probabilities.

    Each position is a bar:
        - N (nuclear) on bottom, color #da618f
        - M (mitochondrial) on top, color #26abdf

    The total figure size is fixed; bar width shrinks as length increases.
    """
    T = gamma.shape[0]
    x = np.arange(T)

    nuc_probs = gamma[:, nuc_id]
    mito_probs = gamma[:, mito_id]

    # Colors: N = #da618f, M = #26abdf
    color_n = "#da618f"
    color_m = "#26abdf"

    # Fixed figure size (inches): width and height do not depend on T
    fig, ax = plt.subplots(figsize=(10, 3))

    # Bar width in data coordinates; with xlim set to [-0.5, T-0.5],
    # the visual width will shrink when T grows, but figure size is constant.
    bar_width = 1.0

    ax.bar(
        x,
        nuc_probs,
        width=bar_width,
        color=color_n,
        label="N (nuclear)",
        linewidth=0,
    )
    ax.bar(
        x,
        mito_probs,
        bottom=nuc_probs,
        width=bar_width,
        color=color_m,
        label="M (mitochondrial)",
        linewidth=0,
    )

    # X/Y limits
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(0.0, 1.0)

    # X ticks: dense for short sequences, sparse for long ones
    if T <= 50:
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x], fontsize=6, rotation=90)
    else:
        step = max(1, T // 20)
        tick_idx = np.arange(0, T, step)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(i + 1) for i in tick_idx], fontsize=6, rotation=90)

    # Y ticks
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Posterior probability", fontsize=9)
    ax.set_xlabel("Position (1-based)", fontsize=9)

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # Title
    ax.set_title(f"{seq_id} posterior N/M probabilities", fontsize=10)

    # Make room on the right for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Legend on the right
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )

    return fig


def main():
    ap = argparse.ArgumentParser(
        description=(
            "1-nt posterior decoding (nuclear vs mitochondrial) using "
            "forward-backward on full cleaned CDS, with optional PDF visualization."
        )
    )
    ap.add_argument("--species", choices=["hs", "mm", "rn"], required=True)
    ap.add_argument("--assets_dir", default="out")

    # Input options
    ap.add_argument("--fasta")
    ap.add_argument("--seq", action="append")
    ap.add_argument("--seq_id", action="append")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--stdin_id", default="stdin_seq")

    # Posterior calling thresholds
    ap.add_argument(
        "--hi_thresh",
        type=float,
        default=0.8,
        help="High-confidence threshold for nuclear/mitochondrial fraction.",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="Minimal difference between fractions to avoid ambiguous call.",
    )

    # Output (TSV + optional PDF)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument(
        "--emit_path",
        action="store_true",
        help="If set, emit posterior-decoded path per sequence.",
    )
    ap.add_argument(
        "--out_pdf",
        help=(
            "If set, write a PDF with one page per sequence visualization "
            "of per-position N/M posterior probabilities."
        ),
    )

    args = ap.parse_args()

    if not (args.fasta or args.seq or args.stdin):
        raise SystemExit("Provide input via --fasta or --seq (repeatable) or --stdin.")

    model_p, _, states_p = resolve_assets(args.species, args.assets_dir)
    pi, A, B = load_model(model_p)
    state_names, nuc_id, mito_id = load_states(states_p)

    # Prepare optional PDF writer
    pdf = None
    if args.out_pdf:
        pdf = PdfPages(args.out_pdf)

    with open(args.out_tsv, "w") as fo:
        # Header: id, log-likelihood, nuclear_frac, mito_frac, call, length
        fo.write("#id\tlogprob\tnuclear_frac\tmito_frac\tcall\tlen_nt\n")

        for rid, raw in iter_inputs(args):
            # Clean full CDS; no length filter except empty sequences.
            s = clean_nt(raw)
            obs = np.array([IDX[c] for c in s if c in IDX], dtype=np.int64)

            if obs.size == 0:
                # Skip sequences that become empty after cleaning
                continue

            # Forward-backward and posterior summary
            ll, gamma = forward_backward(pi, A, B, obs)
            L = len(obs)

            summ = summarize_posterior(
                gamma,
                nuc_id=nuc_id,
                mito_id=mito_id,
                state_names=state_names,
                hi_thresh=args.hi_thresh,
                margin=args.margin,
            )

            fo.write(
                f"{rid}\t{ll:.3f}\t"
                f"{summ['nuclear_frac']}\t{summ['mito_frac']}\t"
                f"{summ['call']}\t{L}\n"
            )

            # Optional posterior-decoded path
            if args.emit_path:
                path = posterior_decode(gamma)
                path_str = " ".join(map(str, path.tolist()))
                fo.write(f"{rid}\tPATH\tcds\t{path_str}\n")

            # Optional visualization: one page per sequence
            if pdf is not None:
                fig = make_posterior_figure(gamma, nuc_id, mito_id, rid)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    if pdf is not None:
        pdf.close()

    print(f"[OK] posterior decode -> {args.out_tsv}")
    if args.out_pdf:
        print(f"[OK] PDF visualization -> {args.out_pdf}")


if __name__ == "__main__":
    main()
