#!/usr/bin/env python3
# 2-nt Viterbi decoding. Species REQUIRED; accepts FASTA/--seq/--stdin.
from Bio import SeqIO
import argparse, os.path as op, json, re, sys, numpy as np

STD_STOPS={"TAA","TAG","TGA"}
VERTEBRATE_MITO_STOPS={"TAA","TAG","AGA","AGG"}
BASES=set("ACGT")

def clean_nt(s):
    s=str(s).upper().replace("U","T")
    return "".join([c for c in s if c in BASES])

def to_dinucs(s):
    return [s[i:i+2] for i in range(len(s)-1)]

def longest_orf(seq, stops, min_nt=150):
    s=clean_nt(seq); best=""
    for frame in (0,1,2):
        start=frame; i=frame
        while i+3<=len(s):
            c=s[i:i+3]
            if c in stops:
                if i-start>len(best): best=s[start:i]
                start=i+3
            i+=3
        tail=s[start: len(s)-((len(s)-start)%3)]
        if len(tail)>len(best): best=tail
    return best if len(best)>=min_nt else ""

def resolve_assets(species, assets_dir):
    model = op.join(assets_dir, f"{species}_mitoSpotter_hmm_2nt.json")
    # Support multiple vocabulary naming variants
    vocab_candidates = [
        op.join(assets_dir, f"{species}_nt_2_vocab.json"),
        op.join(assets_dir, f"{species}_nt2_vocab.json"),
        op.join(assets_dir, f"{species}_nt_vocab.json"),
    ]
    vocab = next((p for p in vocab_candidates if op.exists(p)), None)

    # Prefer *_nt_state_names.json but fall back to *_state_names.json
    states_pref = op.join(assets_dir, f"{species}_nt_state_names.json")
    states_alt  = op.join(assets_dir, f"{species}_state_names.json")
    states = states_pref if op.exists(states_pref) else states_alt

    for p in (model, vocab, states):
        if not p or not op.exists(p):
            raise SystemExit(f"Missing asset: {p}")
    return model, vocab, states

def load_model(model_json):
    M=json.load(open(model_json))
    pi=np.array(M["startprob"],dtype=np.float64)
    A =np.array(M["transmat"], dtype=np.float64)
    B =np.array(M["emissionprob"],dtype=np.float64)
    order=M.get("dinuc_order")  # not strictly needed at decode
    idx={d:i for i,d in enumerate(order)} if order else {d:i for i,d in enumerate(
        [a+b for a in "ACGT" for b in "ACGT"])}
    return pi,A,B,idx

def load_states(p):
    S=json.load(open(p)); S={int(k):v for k,v in S.items()}
    inv={v:k for k,v in S.items()}
    return S, inv.get("nuclear",0), inv.get("mitochondrial",1)

def viterbi(pi,A,B,obs):
    T=len(obs); N=A.shape[0]
    logA=np.log(A+1e-300); logB=np.log(B+1e-300); logpi=np.log(pi+1e-300)
    d=np.zeros((T,N)); psi=np.zeros((T,N),dtype=np.int64)
    d[0]=logpi+logB[:,obs[0]]; psi[0]=-1
    for t in range(1,T):
        for j in range(N):
            s=d[t-1]+logA[:,j]; psi[t,j]=int(np.argmax(s)); d[t,j]=s[psi[t,j]]+logB[j,obs[t]]
    path=np.zeros(T,dtype=np.int64); path[-1]=int(np.argmax(d[-1]))
    for t in range(T-2,-1,-1): path[t]=psi[t+1, path[t+1]]
    return float(d[-1, path[-1]]), path

def forward_ll(pi,A,B,obs):
    T=len(obs); N=A.shape[0]
    alpha=np.zeros((T,N)); c=np.zeros(T)
    alpha[0]=pi*B[:,obs[0]]; c[0]=alpha[0].sum() or 1e-300; alpha[0]/=c[0]
    for t in range(1,T):
        alpha[t]=(alpha[t-1]@A)*B[:,obs[t]]
        c[t]=alpha[t].sum() or 1e-300; alpha[t]/=c[t]
    return -float(np.sum(np.log(c+1e-300)))

def summarize(path, nuc_id=0, mito_id=1, state_names=None):
    if len(path)==0: return {"nuclear_frac":0.0,"mito_frac":0.0,"winner":"NA"}
    nuc=float(np.mean(path==nuc_id)); mito=float(np.mean(path==mito_id))
    winner_id=nuc_id if nuc>=mito else mito_id
    name=state_names.get(winner_id, "nuclear" if winner_id==nuc_id else "mitochondrial") if state_names else ("nuclear" if winner_id==nuc_id else "mitochondrial")
    return {"nuclear_frac":round(nuc,4),"mito_frac":round(mito,4),"winner":name}

def iter_inputs(args):
    if args.seq:
        for i,s in enumerate(args.seq,1):
            sid=(args.seq_id[i-1] if args.seq_id and i<=len(args.seq_id) else f"seq{i}")
            yield sid, s
    if args.stdin:
        raw=sys.stdin.read()
        if raw: yield args.stdin_id, raw
    if args.fasta:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            yield rec.id, str(rec.seq)

def main():
    ap=argparse.ArgumentParser(description="2-nt Viterbi decoding. Species REQUIRED.")
    ap.add_argument("--species", choices=["hs","mm","rn"], required=True)
    ap.add_argument("--assets_dir", default="out")
    ap.add_argument("--fasta"); ap.add_argument("--seq", action="append"); ap.add_argument("--seq_id", action="append")
    ap.add_argument("--stdin", action="store_true"); ap.add_argument("--stdin_id", default="stdin_seq")
    ap.add_argument("--segment", choices=["cds","orf"], default="cds", help="cds=use full cleaned input; orf=find longest ORF by stops.")
    ap.add_argument("--code", choices=["standard","vertebrate_mito","auto"], default="auto", help="Used only if --segment orf.")
    ap.add_argument("--min_len", type=int, default=150)
    ap.add_argument("--out_tsv", required=True); ap.add_argument("--emit_path", action="store_true")
    args=ap.parse_args()

    if not (args.fasta or args.seq or args.stdin):
        raise SystemExit("Provide input via --fasta or --seq (repeatable) or --stdin.")

    model_p, vocab_p, states_p = resolve_assets(args.species, args.assets_dir)
    pi,A,B,idx = load_model(model_p)
    state_names, nuc_id, mito_id = load_states(states_p)

    with open(args.out_tsv, "w") as fo:
        fo.write("#id\tlogprob\twinner\tnuclear_frac\tmito_frac\tlen_tokens\n")
        for rid, raw in iter_inputs(args):
            candidates=[]
            if args.segment=="cds":
                s=clean_nt(raw)
                if len(s)<max(args.min_len,2): continue
                tokens=to_dinucs(s)
                obs=np.array([idx[t] for t in tokens if t in idx], dtype=np.int64)
                if obs.size==0: continue
                ll=forward_ll(pi,A,B,obs); vll,path=viterbi(pi,A,B,obs)
                candidates.append((ll,vll,path,obs.size,"cds"))
            else:
                for code in (["standard","vertebrate_mito"] if args.code=="auto" else [args.code]):
                    stops=STD_STOPS if code=="standard" else VERTEBRATE_MITO_STOPS
                    orf=longest_orf(raw, stops, min_nt=args.min_len)
                    if not orf: continue
                    tokens=to_dinucs(orf)
                    obs=np.array([idx[t] for t in tokens if t in idx], dtype=np.int64)
                    if obs.size==0: continue
                    ll=forward_ll(pi,A,B,obs); vll,path=viterbi(pi,A,B,obs)
                    candidates.append((ll,vll,path,obs.size,code))
            if not candidates: continue
            best=max(candidates, key=lambda t:t[0])
            ll,_,path,L,tag=best
            summ=summarize(path, nuc_id, mito_id, state_names)
            fo.write(f"{rid}\t{ll:.3f}\t{summ['winner']}\t{summ['nuclear_frac']}\t{summ['mito_frac']}\t{L}\n")
            if args.emit_path:
                fo.write(f"{rid}\tPATH\t{tag}\t{' '.join(map(str, path.tolist()))}\n")
    print(f"[OK] decode -> {args.out_tsv}")

if __name__=="__main__":
    main()
