#!/usr/bin/env python3
# Train a 2-state Multinomial HMM on 1-nt sequences (A/C/G/T), from scratch.
import argparse, json, os, sys, numpy as np

# Graceful fallback if tqdm is missing
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k): return x

NUC, MITO = 0, 1
VOCAB = ["A","C","G","T"]
IDX = {b:i for i,b in enumerate(VOCAB)}

def read_nt_tsv(p):
    # Each line: id \t A C G T ...
    seqs = []
    with open(p) as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            parts=line.split("\t")
            if len(parts)<2: continue
            xs=[IDX[c] for c in parts[1].split() if c in IDX]
            if xs: seqs.append(np.asarray(xs, dtype=np.int64))
    return seqs

def freq_from_seqs(seqs, alpha=1.0):
    # Laplace-smoothed mono-nucleotide frequencies
    cnt = np.full(len(VOCAB), alpha, dtype=np.float64)
    for s in seqs:
        for x in s: cnt[x] += 1.0
    return cnt / cnt.sum()

def forward_backward_scaled(pi,A,B,obs):
    # Scaled forward-backward for numerical stability
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

def baum_welch(seqs,pi,A,B,n_iter=20,learn="t",emis_alpha=1.0, show_progress=True):
    # EM with tqdm progress bars (outer: iterations; inner: sequences)
    N,M=B.shape
    outer = tqdm(range(n_iter), desc="EM", unit="iter",
                 disable=not sys.stderr.isatty() or not show_progress)
    for it in outer:
        pi_num=np.zeros(N); A_num=np.zeros((N,N)); A_den=np.zeros(N)
        B_num=np.full((N,M), emis_alpha); B_den=np.full(N, emis_alpha*M)
        tot=0.0

        inner = tqdm(seqs, desc=f"iter {it+1}/{n_iter}", unit="seq",
                     leave=False, disable=not sys.stderr.isatty() or not show_progress)
        for obs in inner:
            gamma,xi,ll=forward_backward_scaled(pi,A,B,obs); tot+=ll
            pi_num+=gamma[0]; A_num+=xi.sum(0); A_den+=gamma[:-1].sum(0)
            for t,o in enumerate(obs):
                B_num[:,o]+=gamma[t]; B_den+=gamma[t]
        pi = pi_num/pi_num.sum() if pi_num.sum()>0 else np.full(N,1.0/N)
        if learn in ("t","et"):
            A = A_num/(A_den[:,None]+1e-300); A=(A.T/A.sum(1)).T
        if learn=="et":
            B = B_num/(B_den[:,None]+1e-300); B=(B.T/B.sum(1)).T

        # Show current log-likelihood in the outer bar
        if hasattr(outer, "set_postfix"):
            outer.set_postfix({"loglik": f"{tot:.1f}"})
        else:
            print(f"[EM] iter={it+1} loglik={tot:.3f}")
    return pi,A,B

def main():
    ap=argparse.ArgumentParser(description="Train 1-nt HMM (A/C/G/T) from nuclear/mito CDS.")
    ap.add_argument("--nuclear_nt_tsv", required=True)
    ap.add_argument("--mito_nt_tsv", required=True)
    ap.add_argument("--out_model_json", required=True)
    ap.add_argument("--out_vocab_json", required=True)
    ap.add_argument("--out_states_json", required=True)
    ap.add_argument("--learn", choices=["t","et","none"], default="t")
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--self_loop", type=float, default=0.995)
    ap.add_argument("--emis_smooth", type=float, default=1.0)
    args=ap.parse_args()

    # Load data
    nuc=read_nt_tsv(args.nuclear_nt_tsv)
    mit=read_nt_tsv(args.mito_nt_tsv)
    allseq=nuc+mit
    if not allseq: raise SystemExit("No sequences to train.")

    # Init emissions from class frequencies
    nuc_emit=freq_from_seqs(nuc, args.emis_smooth)
    mit_emit=freq_from_seqs(mit, args.emis_smooth)
    B=np.vstack([nuc_emit, mit_emit])

    # Init transitions and start
    s=args.self_loop; A=np.array([[s,1-s],[1-s,s]],dtype=np.float64)
    pi=np.array([0.5,0.5],dtype=np.float64)

    # EM (optional)
    if args.learn!="none":
        pi,A,B=baum_welch(allseq,pi,A,B,
                          n_iter=args.n_iter,
                          learn=args.learn,
                          emis_alpha=args.emis_smooth,
                          show_progress=True)

    # Save artifacts
    os.makedirs(os.path.dirname(args.out_model_json) or ".", exist_ok=True)
    model={"n_states":2,"n_observations":len(VOCAB),
           "startprob":pi.tolist(),"transmat":A.tolist(),
           "emissionprob":B.tolist(),"base_order":VOCAB}
    with open(args.out_model_json,"w") as f: json.dump(model,f,indent=2)
    with open(args.out_vocab_json,"w") as f: json.dump({"bases":VOCAB,"index":IDX},f,indent=2)
    with open(args.out_states_json,"w") as f: json.dump({"0":"nuclear","1":"mitochondrial"}, f, indent=2)

    print(f"[OK] model -> {args.out_model_json}")
    print(f"[OK] vocab -> {args.out_vocab_json}")
    print(f"[OK] states -> {args.out_states_json}")

if __name__=="__main__":
    main()