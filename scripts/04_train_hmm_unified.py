#!/usr/bin/env python3
# Unified training script for a 2-state Multinomial HMM on 1-nt sequences (A/C/G/T).
# Supports three training methods:
#   - EM       : Baum-Welch (forward-backward, soft EM)
#   - Viterbi  : Viterbi training (hard EM)
#   - hybrid   : EM first, then Viterbi fine-tuning

import argparse
import json
import os
import sys
import numpy as np

# Graceful fallback if tqdm is missing
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x


NUC, MITO = 0, 1
VOCAB = ["A", "C", "G", "T"]
IDX = {b: i for i, b in enumerate(VOCAB)}


# -------------------- Data loading & helpers -------------------- #

def read_nt_tsv(p):
    """
    Read sequences from a TSV file.
    Each line format: id \t A C G T ...
    Returns a list of np.ndarray(int64) with values in [0, M-1].
    """
    seqs = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            xs = [IDX[c] for c in parts[1].split() if c in IDX]
            if xs:
                seqs.append(np.asarray(xs, dtype=np.int64))
    return seqs


def freq_from_seqs(seqs, alpha=1.0):
    """
    Laplace-smoothed mono-nucleotide frequencies over VOCAB.
    """
    cnt = np.full(len(VOCAB), alpha, dtype=np.float64)
    for s in seqs:
        for x in s:
            cnt[x] += 1.0
    return cnt / cnt.sum()


# -------------------- Forward-backward (soft EM) -------------------- #

def forward_backward_scaled(pi, A, B, obs):
    """
    Scaled forward-backward for numerical stability.

    pi:  (N,)
    A:   (N, N)
    B:   (N, M)
    obs: (T,) int64

    Returns:
      gamma: (T, N)
      xi:    (T-1, N, N)
      ll:    float, log-likelihood log p(obs)
    """
    T = len(obs)
    N = A.shape[0]
    alpha = np.zeros((T, N))
    beta = np.zeros((T, N))
    c = np.zeros(T)

    # forward
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum() or 1e-300
    alpha[0] /= c[0]

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum() or 1e-300
        alpha[t] /= c[t]

    # backward
    beta[-1] = 1.0 / c[-1]
    for t in range(T - 2, -1, -1):
        beta[t] = (A * B[:, obs[t + 1]]).dot(beta[t + 1])
        beta[t] /= c[t]

    # gamma
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

    # xi
    xi = np.zeros((T - 1, N, N))
    for t in range(T - 1):
        denom = (alpha[t] @ A * B[:, obs[t + 1]]).dot(beta[t + 1]) or 1e-300
        xi[t] = (alpha[t][:, None] * A) * (
            B[:, obs[t + 1]] * beta[t + 1]
        )[None, :] / denom

    ll = -np.sum(np.log(c + 1e-300))
    return gamma, xi, ll


def baum_welch(
    seqs,
    pi,
    A,
    B,
    n_iter=20,
    learn="t",
    emis_alpha=1.0,
    show_progress=True,
    tol=1e-4,
):
    """
    Standard Baum-Welch EM (soft EM).

    seqs: list of np.ndarray(int64)
    pi, A, B: initial parameters
    n_iter: maximum number of EM iterations
    learn:
      "t"   -> learn transitions only
      "et"  -> learn transitions + emissions
      "none" (not used here) -> skip EM
    emis_alpha: Laplace smoothing for emission probabilities
    tol: relative tolerance for early stopping on log-likelihood
    """
    N, M = B.shape
    outer = tqdm(
        range(n_iter),
        desc="EM",
        unit="iter",
        disable=not sys.stderr.isatty() or not show_progress,
    )

    prev_ll = None
    for it in outer:
        pi_num = np.zeros(N)
        A_num = np.zeros((N, N))
        A_den = np.zeros(N)
        B_num = np.full((N, M), emis_alpha)
        B_den = np.full(N, emis_alpha * M)
        tot_ll = 0.0

        inner = tqdm(
            seqs,
            desc=f"iter {it+1}/{n_iter}",
            unit="seq",
            leave=False,
            disable=not sys.stderr.isatty() or not show_progress,
        )

        for obs in inner:
            gamma, xi, ll = forward_backward_scaled(pi, A, B, obs)
            tot_ll += ll

            pi_num += gamma[0]
            A_num += xi.sum(0)
            A_den += gamma[:-1].sum(0)

            for t, o in enumerate(obs):
                B_num[:, o] += gamma[t]
                B_den += gamma[t]

        # normalize
        pi = pi_num / (pi_num.sum() or 1e-300)
        if learn in ("t", "et"):
            A = A_num / (A_den[:, None] + 1e-300)
            A = (A.T / (A.sum(1) + 1e-300)).T
        if learn == "et":
            B = B_num / (B_den[:, None] + 1e-300)
            B = (B.T / (B.sum(1) + 1e-300)).T

        if hasattr(outer, "set_postfix"):
            outer.set_postfix({"loglik": f"{tot_ll:.1f}"})
        else:
            print(f"[EM] iter={it+1} loglik={tot_ll:.3f}")

        # early stopping
        if prev_ll is not None and abs(tot_ll - prev_ll) < tol * (1 + abs(prev_ll)):
            break
        prev_ll = tot_ll

    return pi, A, B


# -------------------- Viterbi decode & hard EM -------------------- #

def viterbi_decode(pi, A, B, obs):
    """
    Viterbi decoding: find the most likely hidden state path.

    pi:  (N,)
    A:   (N, N)
    B:   (N, M)
    obs: (T,) int64

    Returns:
      path:   (T,) int64
      loglik: float, log p(x, z*)
    """
    T = len(obs)
    N = A.shape[0]

    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_B = np.log(B + 1e-300)

    dp = np.full((T, N), -np.inf)
    ptr = np.zeros((T, N), dtype=np.int64)

    # t = 0
    o0 = obs[0]
    dp[0] = log_pi + log_B[:, o0]

    # t = 1..T-1
    for t in range(1, T):
        o = obs[t]
        for j in range(N):
            scores = dp[t - 1] + log_A[:, j]
            best_prev = np.argmax(scores)
            dp[t, j] = scores[best_prev] + log_B[j, o]
            ptr[t, j] = best_prev

    path = np.zeros(T, dtype=np.int64)
    last_state = np.argmax(dp[T - 1])
    path[T - 1] = last_state
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    loglik = dp[T - 1, last_state]
    return path, loglik


def viterbi_train(
    seqs,
    pi,
    A,
    B,
    n_iter=20,
    emis_alpha=1.0,
    trans_alpha=1.0,
    show_progress=True,
    tol=1e-4,
):
    """
    Viterbi training (hard EM).

    seqs: list of np.ndarray(int64)
    pi, A, B: initial parameters
    emis_alpha: Laplace smoothing for emission counts
    trans_alpha: Laplace smoothing for transition counts
    tol: relative tolerance for early stopping on joint log-likelihood
    """
    N, M = B.shape
    outer = tqdm(
        range(n_iter),
        desc="Viterbi-EM",
        unit="iter",
        disable=not sys.stderr.isatty() or not show_progress,
    )

    prev_ll = None
    for it in outer:
        pi_cnt = np.full(N, 1e-12)  # tiny prior for initial distribution
        A_cnt = np.full((N, N), trans_alpha)
        B_cnt = np.full((N, M), emis_alpha)

        tot_ll = 0.0

        inner = tqdm(
            seqs,
            desc=f"iter {it+1}/{n_iter}",
            unit="seq",
            leave=False,
            disable=not sys.stderr.isatty() or not show_progress,
        )

        for obs in inner:
            path, ll = viterbi_decode(pi, A, B, obs)
            tot_ll += ll

            # initial state
            pi_cnt[path[0]] += 1.0

            # transitions
            for t in range(len(path) - 1):
                A_cnt[path[t], path[t + 1]] += 1.0

            # emissions
            for t, o in enumerate(obs):
                B_cnt[path[t], o] += 1.0

        # normalize
        pi = pi_cnt / (pi_cnt.sum() + 1e-300)
        A = A_cnt / (A_cnt.sum(axis=1, keepdims=True) + 1e-300)
        B = B_cnt / (B_cnt.sum(axis=1, keepdims=True) + 1e-300)

        if hasattr(outer, "set_postfix"):
            outer.set_postfix({"joint_ll": f"{tot_ll:.1f}"})
        else:
            print(f"[Viterbi] iter={it+1} joint_loglik={tot_ll:.3f}")

        # early stopping on joint likelihood
        if prev_ll is not None and abs(tot_ll - prev_ll) < tol * (1 + abs(prev_ll)):
            break
        prev_ll = tot_ll

    return pi, A, B


# -------------------- Main: unified training interface -------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Train a 2-state 1-nt HMM (A/C/G/T) from nuclear/mito CDS using EM, Viterbi, or hybrid."
    )
    parser.add_argument("--nuclear_nt_tsv", required=True,
                        help="Input TSV with nuclear sequences")
    parser.add_argument("--mito_nt_tsv", required=True,
                        help="Input TSV with mitochondrial sequences")
    parser.add_argument("--out_model_json", required=True,
                        help="Output JSON file for trained model")
    parser.add_argument("--out_vocab_json", required=True,
                        help="Output JSON file for vocabulary")
    parser.add_argument("--out_states_json", required=True,
                        help="Output JSON file for state names")

    # Training method
    parser.add_argument(
        "--train_method",
        choices=["em", "viterbi", "hybrid"],
        default="em",
        help="Training method: 'em' (soft EM), 'viterbi' (hard EM), or 'hybrid' (EM then Viterbi).",
    )

    # EM-related options
    parser.add_argument(
        "--learn",
        choices=["t", "et", "none"],
        default="t",
        help="What EM learns: 't' (transitions), 'et' (transitions + emissions), 'none' (skip EM). Used for EM or hybrid.",
    )
    parser.add_argument(
        "--n_em_iter",
        type=int,
        default=20,
        help="Number of EM (soft) iterations (used when train_method is 'em' or 'hybrid').",
    )

    # Viterbi-related options
    parser.add_argument(
        "--n_viterbi_iter",
        type=int,
        default=0,
        help="Number of Viterbi (hard) iterations (used when train_method is 'viterbi' or 'hybrid').",
    )
    parser.add_argument(
        "--trans_smooth",
        type=float,
        default=1.0,
        help="Laplace smoothing for transition counts in Viterbi training.",
    )

    # General HMM hyper-parameters
    parser.add_argument(
        "--self_loop",
        type=float,
        default=0.995,
        help="Initial self-loop probability for each state (transition matrix initialization).",
    )
    parser.add_argument(
        "--emis_smooth",
        type=float,
        default=1.0,
        help="Laplace smoothing for emissions (also used as prior in EM/Viterbi).",
    )

    args = parser.parse_args()

    # Load data
    nuc = read_nt_tsv(args.nuclear_nt_tsv)
    mit = read_nt_tsv(args.mito_nt_tsv)
    allseq = nuc + mit
    if not allseq:
        raise SystemExit("No sequences to train.")

    # Initialize emissions from class frequencies
    nuc_emit = freq_from_seqs(nuc, args.emis_smooth)
    mit_emit = freq_from_seqs(mit, args.emis_smooth)
    B = np.vstack([nuc_emit, mit_emit]).astype(np.float64)

    # Initialize transitions and start distribution
    s = args.self_loop
    A = np.array([[s, 1.0 - s],
                  [1.0 - s, s]], dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)

    # -------------------- Training selection -------------------- #

    train_method = args.train_method

    if train_method == "em":
        # Pure EM (soft)
        if args.learn != "none" and args.n_em_iter > 0:
            pi, A, B = baum_welch(
                allseq,
                pi,
                A,
                B,
                n_iter=args.n_em_iter,
                learn=args.learn,
                emis_alpha=args.emis_smooth,
                show_progress=True,
                tol=1e-4,
            )
        else:
            print("[INFO] EM training skipped (learn='none' or n_em_iter <= 0).")

    elif train_method == "viterbi":
        # Pure Viterbi (hard)
        if args.n_viterbi_iter > 0:
            pi, A, B = viterbi_train(
                allseq,
                pi,
                A,
                B,
                n_iter=args.n_viterbi_iter,
                emis_alpha=args.emis_smooth,
                trans_alpha=args.trans_smooth,
                show_progress=True,
                tol=1e-4,
            )
        else:
            print("[INFO] Viterbi training skipped (n_viterbi_iter <= 0).")

    elif train_method == "hybrid":
        # 1) Optional EM, then 2) Viterbi fine-tuning
        if args.learn != "none" and args.n_em_iter > 0:
            pi, A, B = baum_welch(
                allseq,
                pi,
                A,
                B,
                n_iter=args.n_em_iter,
                learn=args.learn,
                emis_alpha=args.emis_smooth,
                show_progress=True,
                tol=1e-4,
            )
        else:
            print("[INFO] EM phase skipped in hybrid (learn='none' or n_em_iter <= 0).")

        if args.n_viterbi_iter > 0:
            pi, A, B = viterbi_train(
                allseq,
                pi,
                A,
                B,
                n_iter=args.n_viterbi_iter,
                emis_alpha=args.emis_smooth,
                trans_alpha=args.trans_smooth,
                show_progress=True,
                tol=1e-4,
            )
        else:
            print("[INFO] Viterbi phase skipped in hybrid (n_viterbi_iter <= 0).")

    else:
        raise SystemExit(f"Unknown train_method: {train_method}")

    # -------------------- Save artifacts -------------------- #

    os.makedirs(os.path.dirname(args.out_model_json) or ".", exist_ok=True)
    model = {
        "n_states": 2,
        "n_observations": len(VOCAB),
        "startprob": pi.tolist(),
        "transmat": A.tolist(),
        "emissionprob": B.tolist(),
        "base_order": VOCAB,
        "training": {
            "train_method": args.train_method,
            "learn": args.learn,
            "n_em_iter": args.n_em_iter,
            "n_viterbi_iter": args.n_viterbi_iter,
            "self_loop": args.self_loop,
            "emis_smooth": args.emis_smooth,
            "trans_smooth": args.trans_smooth,
        },
    }
    with open(args.out_model_json, "w") as f:
        json.dump(model, f, indent=2)
    with open(args.out_vocab_json, "w") as f:
        json.dump({"bases": VOCAB, "index": IDX}, f, indent=2)
    with open(args.out_states_json, "w") as f:
        json.dump({"0": "nuclear", "1": "mitochondrial"}, f, indent=2)

    print(f"[OK] model  -> {args.out_model_json}")
    print(f"[OK] vocab  -> {args.out_vocab_json}")
    print(f"[OK] states -> {args.out_states_json}")


if __name__ == "__main__":
    main()

