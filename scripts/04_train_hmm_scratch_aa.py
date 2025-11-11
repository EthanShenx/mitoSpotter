#!/usr/bin/env python3
# Train 2-state Multinomial-HMM on amino-acid tokens (20 AA). Pure numpy, tqdm for progress.
import argparse, json, os, numpy as np
from tqdm import tqdm

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
IDX = {a:i for i,a in enumerate(AA_VOCAB)}

def read_aa_tsv(p):
    seqs = []
    with open(p) as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            parts=line.split("\t")
            if len(parts) < 2: continue
            toks = parts[1].split()
            xs = [IDX[t] for t in toks if t in IDX]
            if xs: seqs.append(np.asarray(xs, dtype=np.int64))
    return seqs

def counts_from_seqs(seqs, M, smoothing=1.0):
    cnt = np.full(M, smoothing, dtype=np.float64)
    for s in seqs:
        for x in s: cnt[x] += 1.0
    prob = cnt / cnt.sum()
    return prob

def forward_backward_scaled(pi,A,B,obs):
    T=len(obs); N=A.shape[0]
    alpha=np.zeros((T,N)); beta=np.zeros((T,N)); c=np.zeros(T)
    alpha[0]=pi*B[:,obs[0]]; c[0]=alpha[0].sum() or 1e-300; alpha[0]/=c[0]
    for t in range(1,T):
        alpha[t]=(alpha[t-1]@A)*B[:,obs[t]]
        c[t]=alpha[t].sum() or 1e-300; alpha[t]/=c[t]
    beta[-1]=1.0/c[-1]
    for t in range(T-2,-1,-1):
        beta[t]=(A*B[:,obs[t+1]]).dot(beta[t+1]); beta[t]/=c[t]
    gamma=alpha*beta; gamma/=gamma.sum(axis=1,keepdims=True)+1e-300
    xi=np.zeros((T-1,N,N))
    for t in range(T-1):
        denom=(alpha[t]@A*B[:,obs[t+1]]).dot(beta[t+1]) or 1e-300
        xi[t]=(alpha[t][:,None]*A)*(B[:,obs[t+1]]*beta[t+1])[None,:]/denom
    ll=-np.sum(np.log(c+1e-300))
    return gamma,xi,ll

def baum_welch(seqs,pi,A,B,n_iter=20,learn="t",emis_smooth=1.0):
    N,M=B.shape
    for it in range(n_iter):
        pi_num = np.zeros(N); A_num=np.zeros((N,N)); A_den=np.zeros(N)
        B_num = np.full((N,M), emis_smooth); B_den = np.full(N, emis_smooth*M)
        tot = 0.0
        # E-step over sequences with progress bar
        for obs in tqdm(seqs, desc=f"E-step iter {it+1}", leave=False):
            gamma,xi,ll = forward_backward_scaled(pi,A,B,obs); tot += ll
            pi_num += gamma[0]; A_num += xi.sum(0); A_den += gamma[:-1].sum(0)
            for t,o in enumerate(obs):
                B_num[:,o] += gamma[t]; B_den += gamma[t]
        # M-step
        pi = pi_num / (pi_num.sum() + 1e-300)
        if learn in ("t","et"):
            A = A_num / (A_den[:,None] + 1e-300); A = (A.T / A.sum(axis=1)).T
        if learn == "et":
            B = B_num / (B_den[:,None] + 1e-300); B = (B.T / B.sum(axis=1)).T
        tqdm.write(f"[EM] iter={it+1} loglik={tot:.3f}")
    return pi,A,B

def main():
    ap = argparse.ArgumentParser(description="Train AA HMM (20 AA) from nuclear/mito AA TSV. Uses tqdm.")
    ap.add_argument("--nuclear_aa_tsv", required=True)
    ap.add_argument("--mito_aa_tsv", required=True)
    ap.add_argument("--out_model_json", required=True)
    ap.add_argument("--out_vocab_json", required=True)
    ap.add_argument("--out_states_json", required=True)
    ap.add_argument("--learn", choices=["t","et","none"], default="t")
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--self_loop", type=float, default=0.995)
    ap.add_argument("--emis_smooth", type=float, default=1.0)
    args = ap.parse_args()

    nuc = read_aa_tsv(args.nuclear_aa_tsv)
    mit = read_aa_tsv(args.mito_aa_tsv)
    allseq = nuc + mit
    if not allseq:
        raise SystemExit("No AA sequences found for training.")

    # Emission init from empirical AA frequency
    nuc_em = counts_from_seqs(nuc, len(AA_VOCAB), smoothing=args.emis_smooth)
    mit_em = counts_from_seqs(mit, len(AA_VOCAB), smoothing=args.emis_smooth)
    B = np.vstack([nuc_em, mit_em])
    s = args.self_loop
    A = np.array([[s,1-s],[1-s,s]], dtype=np.float64)
    pi = np.array([0.5,0.5], dtype=np.float64)

    if args.learn != "none":
        pi, A, B = baum_welch(allseq, pi, A, B, n_iter=args.n_iter, learn=args.learn, emis_smooth=args.emis_smooth)

    os.makedirs(os.path.dirname(args.out_model_json) or ".", exist_ok=True)
    model = {
        "n_states": 2,
        "n_observations": len(AA_VOCAB),
        "startprob": pi.tolist(),
        "transmat": A.tolist(),
        "emissionprob": B.tolist(),
        "aa_order": AA_VOCAB
    }
    with open(args.out_model_json, "w") as f:
        json.dump(model, f, indent=2)
    with open(args.out_vocab_json, "w") as f:
        json.dump({"aa": AA_VOCAB, "index": IDX}, f, indent=2)
    with open(args.out_states_json, "w") as f:
        json.dump({"0":"nuclear","1":"mitochondrial"}, f, indent=2)

    print(f"[OK] model -> {args.out_model_json}")
    print(f"[OK] vocab -> {args.out_vocab_json}")
    print(f"[OK] states -> {args.out_states_json}")

if __name__ == "__main__":
    main()

