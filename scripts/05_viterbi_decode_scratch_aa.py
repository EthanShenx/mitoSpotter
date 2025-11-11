#!/usr/bin/env python3
# Viterbi decode using AA HMM. Species REQUIRED. Accepts FASTA/--seq/--stdin.
from Bio import SeqIO
import argparse, os.path as op, json, re, sys, numpy as np

# codon tables copied from previous script to ensure consistency
STD_CODON_TO_AA = {
 "TTT":"F","TTC":"F","TTA":"L","TTG":"L","TCT":"S","TCC":"S","TCA":"S","TCG":"S",
 "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","TGT":"C","TGC":"C","TGA":"*","TGG":"W",
 "CTT":"L","CTC":"L","CTA":"L","CTG":"L","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
 "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
 "ATT":"I","ATC":"I","ATA":"I","ATG":"M","ACT":"T","ACC":"T","ACA":"T","ACG":"T",
 "AAT":"N","AAC":"N","AAA":"K","AAG":"K","AGT":"S","AGC":"S","AGA":"R","AGG":"R",
 "GTT":"V","GTC":"V","GTA":"V","GTG":"V","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
 "GAT":"D","GAC":"D","GAA":"E","GAG":"E","GGT":"G","GGC":"G","GGA":"G","GGG":"G"
}
MITO_CODON_TO_AA = STD_CODON_TO_AA.copy()
MITO_CODON_TO_AA.update({"TGA":"W","AGA":"*","AGG":"*","ATA":"M"})

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
IDX = {a:i for i,a in enumerate(AA_VOCAB)}

def clean_nt(s):
    return re.sub(r"[^ACGT]", "", str(s).upper().replace("U","T"))

def find_longest_orf_nt(seq, stop_codons, min_nt=150):
    s = clean_nt(seq)
    best = ""
    for frame in (0,1,2):
        start=frame; i=frame
        while i+3<=len(s):
            codon = s[i:i+3]
            if codon in stop_codons:
                if i-start > len(best): best = s[start:i]
                start = i+3
            i += 3
        tail = s[start: len(s) - ((len(s)-start)%3)]
        if len(tail) > len(best): best = tail
    return best if len(best) >= min_nt else ""

def nt_to_codons(nt):
    return [nt[i:i+3] for i in range(0, len(nt)-2, 3)]

def codons_to_aa_array(codons, table):
    aas=[]
    for c in codons:
        aa = table.get(c, "X")
        if aa == "X": return None
        if aa == "*": continue
        if aa not in AA_VOCAB: return None
        aas.append(aa)
    return np.array([IDX[a] for a in aas], dtype=np.int64)

# --- model loading and HMM ops ---
def resolve_assets(species, assets_dir):
    model = op.join(assets_dir, f"{species}_mitoSpotter_hmm_aa.json")
    vocab = op.join(assets_dir, f"{species}_aa_vocab.json")
    states = op.join(assets_dir, f"{species}_state_names.json")
    for p in (model,vocab,states):
        if not op.exists(p): raise SystemExit(f"Missing asset: {p}")
    return model, vocab, states

def load_model(model_json):
    M = json.load(open(model_json))
    pi = np.array(M["startprob"], dtype=np.float64)
    A  = np.array(M["transmat"], dtype=np.float64)
    B  = np.array(M["emissionprob"], dtype=np.float64)
    order = M.get("aa_order")
    idx = {a:i for i,a in enumerate(order)} if order else IDX
    return pi,A,B,idx

def load_states(states_json):
    S = json.load(open(states_json)); S = {int(k):v for k,v in S.items()}
    inv = {v:k for k,v in S.items()}
    return S, inv.get("nuclear",0), inv.get("mitochondrial",1)

def viterbi(pi,A,B,obs):
    T=len(obs); N=A.shape[0]
    logA=np.log(A+1e-300); logB=np.log(B+1e-300); logpi=np.log(pi+1e-300)
    d=np.zeros((T,N)); psi=np.zeros((T,N),dtype=np.int64)
    d[0]=logpi+logB[:,obs[0]]; psi[0]=-1
    for t in range(1,T):
        for j in range(N):
            s = d[t-1] + logA[:,j]; psi[t,j]=int(np.argmax(s)); d[t,j]=s[psi[t,j]] + logB[j,obs[t]]
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
    total = len(path)
    if total == 0: return {"nuclear_frac":0.0, "mito_frac":0.0, "winner":"NA"}
    nuc = float(np.mean(path==nuc_id)); mito = float(np.mean(path==mito_id))
    winner_id = nuc_id if nuc >= mito else mito_id
    name = state_names.get(winner_id, "nuclear" if winner_id==nuc_id else "mitochondrial") if state_names else ("nuclear" if winner_id==nuc_id else "mitochondrial")
    return {"nuclear_frac":round(nuc,4),"mito_frac":round(mito,4),"winner":name}

def iter_inputs(args):
    if args.seq:
        for i,s in enumerate(args.seq,1):
            sid = (args.seq_id[i-1] if args.seq_id and i<=len(args.seq_id) else f"seq{i}")
            yield sid, s
    if args.stdin:
        raw = sys.stdin.read()
        if raw: yield args.stdin_id, raw
    if args.fasta:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            yield rec.id, str(rec.seq)

# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="Viterbi decode with AA HMM. Species REQUIRED.")
    ap.add_argument("--species", choices=["hs","mm","rn"], required=True)
    ap.add_argument("--assets_dir", default="out")
    ap.add_argument("--fasta"); ap.add_argument("--seq", action="append"); ap.add_argument("--seq_id", action="append")
    ap.add_argument("--stdin", action="store_true"); ap.add_argument("--stdin_id", default="stdin_seq")
    ap.add_argument("--segment", choices=["cds","orf"], default="orf")
    ap.add_argument("--code", choices=["standard","vertebrate_mito","auto"], default="auto",
                    help="Only meaningful if --segment orf.")
    ap.add_argument("--min_nt", type=int, default=150)
    ap.add_argument("--out_tsv", required=True); ap.add_argument("--emit_path", action="store_true")
    args = ap.parse_args()

    if not (args.fasta or args.seq or args.stdin):
        raise SystemExit("Provide input via --fasta or --seq or --stdin.")

    model_p, vocab_p, states_p = resolve_assets(args.species, args.assets_dir)
    pi,A,B,idx = load_model(model_p)
    state_names, nuc_id, mito_id = load_states(states_p)

    with open(args.out_tsv, "w") as fo:
        fo.write("#id\tlogprob\twinner\tnuclear_frac\tmito_frac\tlen_aa\n")
        for rid, raw in iter_inputs(args):
            candidates = []
            if args.segment == "cds":
                s = clean_nt(raw)
                if len(s) < args.min_nt: continue
                cods = nt_to_codons(s)
                obs = codons_to_aa_array(cods, STD_CODON_TO_AA)  # default standard; user can pre-translate if desired
                if obs is None or obs.size==0: continue
                ll = forward_ll(pi,A,B,obs); vll,path = viterbi(pi,A,B,obs)
                candidates.append((ll,vll,path,obs.size,"cds"))
            else:
                for code in (["standard","vertebrate_mito"] if args.code=="auto" else [args.code]):
                    stops = {"TAA","TAG","TGA"} if code=="standard" else {"TAA","TAG","AGA","AGG"}
                    orf = find_longest_orf_nt(raw, stops, min_nt=args.min_nt)
                    if not orf: continue
                    cods = nt_to_codons(orf)
                    table = STD_CODON_TO_AA if code=="standard" else MITO_CODON_TO_AA
                    obs = codons_to_aa_array(cods, table)
                    if obs is None or obs.size==0: continue
                    ll = forward_ll(pi,A,B,obs); vll,path = viterbi(pi,A,B,obs)
                    candidates.append((ll,vll,path,obs.size,code))
            if not candidates: continue
            best = max(candidates, key=lambda t: t[0])
            ll,_,path,L,tag = best
            summ = summarize(path, nuc_id, mito_id, state_names)
            fo.write(f"{rid}\t{ll:.3f}\t{summ['winner']}\t{summ['nuclear_frac']}\t{summ['mito_frac']}\t{L}\n")
            if args.emit_path:
                fo.write(f"{rid}\tPATH\t{tag}\t{' '.join(map(str, path.tolist()))}\n")
    print(f"[OK] decode -> {args.out_tsv}")

if __name__ == "__main__":
    main()

