#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, numpy as np
from tqdm import tqdm, trange

NUC_STATE, MITO_STATE = 0, 1

# ---------- Data IO ----------

def build_vocab():
    # 64 codons in lexicographic order
    bases = ["A","C","G","T"]
    codons = [a+b+c for a in bases for b in bases for c in bases]
    idx = {c:i for i,c in enumerate(codons)}
    return codons, idx

def read_tsv_codons(p):
    # Load codon sequences per transcript
    seqs = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) < 2: continue
            seqs.append(parts[1].split())
    return seqs

def seqs_to_int_lists(seqs, idx):
    # Return list of np.int64 arrays (each is one sequence of symbols)
    out = []
    for cods in seqs:
        xs = [idx[c] for c in cods if c in idx]
        if xs: out.append(np.asarray(xs, dtype=np.int64))
    return out

def counts_from_seqs(seqs, vocab_size, smoothing=1.0):
    # Laplace-smoothed multinomial
    cnt = np.full(vocab_size, smoothing, dtype=np.float64)
    for s in seqs:
        for c in s:
            cnt[c] += 1.0
    prob = cnt / cnt.sum()
    return prob

# ---------- HMM core (Multinomial) ----------

def forward_backward_scaled(pi, A, B, obs):
    """
    Scaled forward-backward to avoid underflow.
    obs: np.array shape (T,), int symbols
    Returns: gamma (T,N), xi (T-1,N,N), loglik
    """
    T = len(obs); N = A.shape[0]
    alpha = np.zeros((T, N), dtype=np.float64)
    beta  = np.zeros((T, N), dtype=np.float64)
    c = np.zeros(T, dtype=np.float64)  # scaling factors

    # Init
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum()
    if c[0] <= 0: c[0] = 1e-300
    alpha[0] /= c[0]

    # Forward
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum()
        if c[t] <= 0: c[t] = 1e-300
        alpha[t] /= c[t]

    # Backward
    beta[-1] = 1.0 / c[-1]
    for t in range(T-2, -1, -1):
        beta[t] = (A * B[:, obs[t+1]]).dot(beta[t+1])
        beta[t] /= c[t]

    # Posteriors
    gamma = alpha * beta
    gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)

    xi = np.zeros((T-1, N, N), dtype=np.float64)
    for t in range(T-1):
        denom = (alpha[t] @ A * B[:, obs[t+1]]).dot(beta[t+1])
        if denom <= 0: denom = 1e-300
        # Outer-like update per i->j
        xi[t] = (alpha[t][:,None] * A) * (B[:, obs[t+1]] * beta[t+1])[None,:] / denom

    loglik = -np.sum(np.log(c + 1e-300))
    return gamma, xi, loglik

def baum_welch(seqs, pi, A, B, n_iter=20, learn="t", emis_smooth=1.0):
    """
    EM for Multinomial-HMM.
    learn: "none" | "t" | "et"  (transitions only, or emissions+transitions)
    """
    N, M = B.shape
    use_tqdm = trange is not None  # fallback to prints if tqdm missing
    outer = trange(n_iter, desc="EM", unit="iter") if use_tqdm else range(n_iter)

    for it in outer:
        # Accumulators
        pi_num = np.zeros(N, dtype=np.float64)
        A_num  = np.zeros((N, N), dtype=np.float64)
        A_den  = np.zeros(N, dtype=np.float64)
        B_num  = np.full((N, M), emis_smooth, dtype=np.float64)  # Laplace prior
        B_den  = np.full(N, emis_smooth * M, dtype=np.float64)
        total_ll = 0.0

        seq_iter = tqdm(seqs, desc=f"E-step {it+1}/{n_iter}", leave=False, unit="seq") if use_tqdm else seqs
        for obs in seq_iter:
            gamma, xi, ll = _per_seq_stats(obs, pi, A, B)
            total_ll += ll

            pi_num += gamma[0]
            A_num  += xi.sum(axis=0)            # sum over t
            A_den  += gamma[:-1].sum(axis=0)    # exclude last time for transitions

            # emission counts
            for t, o in enumerate(obs):
                B_num[:, o] += gamma[t]
                B_den       += gamma[t]

        # M-step
        pi = _normalize(pi_num)

        if learn in ("t", "et"):
            A = A_num / (A_den[:,None] + 1e-300)
            A = np.where(A <= 1e-300, 1e-300, A)
            A = (A.T / A.sum(axis=1)).T

        if learn == "et":
            B = B_num / (B_den[:,None] + 1e-300)
            B = np.where(B <= 1e-300, 1e-300, B)
            B = (B.T / B.sum(axis=1)).T

        # Progress display
        if use_tqdm:
            outer.set_postfix_str(f"ll={total_ll:.3f}")
        else:
            print(f"[EM] iter={it+1}/{n_iter} loglik={total_ll:.3f}")

    return pi, A, B

def _per_seq_stats(obs, pi, A, B):
    gamma, xi, ll = forward_backward_scaled(pi, A, B, obs)
    return gamma, xi, ll

def viterbi(pi, A, B, obs):
    """Classic Viterbi in log-space. Returns (logprob, path)"""
    T = len(obs); N = A.shape[0]
    logA = np.log(A + 1e-300)
    logB = np.log(B + 1e-300)
    logpi = np.log(pi + 1e-300)

    delta = np.zeros((T, N), dtype=np.float64)
    psi   = np.zeros((T, N), dtype=np.int64)

    delta[0] = logpi + logB[:, obs[0]]
    psi[0] = -1

    for t in range(1, T):
        for j in range(N):
            scores = delta[t-1] + logA[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] + logB[j, obs[t]]

    path = np.zeros(T, dtype=np.int64)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    logprob = delta[-1, path[-1]]
    return logprob, path

def log_likelihood(pi, A, B, obs):
    """Return sequence log-likelihood via scaling factors."""
    _, _, ll = forward_backward_scaled(pi, A, B, obs)
    return ll

def _normalize(v):
    s = v.sum()
    if s <= 0: return np.full_like(v, 1.0/len(v))
    return v / s

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Train a 2-state Multinomial-HMM (from scratch) on codon usage.")
    ap.add_argument("--nuclear_tsv", required=True)
    ap.add_argument("--mito_tsv", required=True)
    ap.add_argument("--out_model_json", required=True)
    ap.add_argument("--out_vocab_json", required=True)
    ap.add_argument("--out_states_json", required=True)
    ap.add_argument("--learn", choices=["t","et","none"], default="t",
                    help="Which params to learn by EM. t=transitions only, et=emit+trans, none=no EM.")
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--self_loop", type=float, default=0.99)
    ap.add_argument("--emis_smooth", type=float, default=1.0, help="Laplace alpha for emissions.")
    args = ap.parse_args()

    codons, idx = build_vocab()
    nuc_cod = read_tsv_codons(args.nuclear_tsv)
    mit_cod = read_tsv_codons(args.mito_tsv)

    nuc_int = seqs_to_int_lists(nuc_cod, idx)
    mit_int = seqs_to_int_lists(mit_cod, idx)
    all_int = nuc_int + mit_int
    if not all_int:
        raise SystemExit("No sequences found for training.")

    M = len(codons); N = 2

    # Init emissions from per-class frequencies (codon usage preference)
    nuc_emis = counts_from_seqs(nuc_int, M, smoothing=args.emis_smooth)
    mit_emis = counts_from_seqs(mit_int, M, smoothing=args.emis_smooth)
    B = np.vstack([nuc_emis, mit_emis])

    # Init transitions as strong self-loops
    s = args.self_loop
    A = np.array([[s, 1.0-s],
                  [1.0-s, s]], dtype=np.float64)

    # Init start as uniform
    pi = np.array([0.5, 0.5], dtype=np.float64)

    # Optional EM refinement
    if args.learn != "none":
        pi, A, B = baum_welch(all_int, pi, A, B, n_iter=args.n_iter, learn=args.learn, emis_smooth=args.emis_smooth)

    # Save artifacts
    os.makedirs(os.path.dirname(args.out_model_json), exist_ok=True)
    model = {
        "n_states": N,
        "n_observations": M,
        "startprob": pi.tolist(),
        "transmat": A.tolist(),
        "emissionprob": B.tolist(),
        "codon_order": codons
    }
    with open(args.out_model_json, "w") as f: json.dump(model, f, indent=2)
    with open(args.out_vocab_json, "w") as f: json.dump({"codons": codons, "index": {c:i for i,c in enumerate(codons)}}, f, indent=2)
    with open(args.out_states_json, "w") as f: json.dump({str(NUC_STATE):"nuclear", str(MITO_STATE):"mitochondrial"}, f, indent=2)

    print(f"[OK] model -> {args.out_model_json}")
    print(f"[OK] vocab -> {args.out_vocab_json}")
    print(f"[OK] states -> {args.out_states_json}")

if __name__ == "__main__":
    main()

