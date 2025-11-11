#!/usr/bin/env python3
# Convert CDS FASTA to per-transcript base sequences (A/C/G/T) for 1-nt HMM.
from Bio import SeqIO
import argparse, re, os

def nts_from_cds(seq, min_len=150):
    s = str(seq).upper().replace("U", "T")
    s = re.sub(r"[^ACGT]", "", s)
    return s if len(s) >= min_len else ""

def main():
    ap = argparse.ArgumentParser(description="Emit per-transcript base sequences from CDS FASTA.")
    ap.add_argument("--cds_fasta", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--min_len", type=int, default=150, help="Minimum CDS length (nt).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    kept = 0
    with open(args.out_tsv, "w") as fo:
        for rec in SeqIO.parse(args.cds_fasta, "fasta"):
            tid = rec.id.split(".")[0]
            nts = nts_from_cds(rec.seq, min_len=args.min_len)
            if not nts: 
                continue
            fo.write(tid + "\t" + " ".join(list(nts)) + "\n")
            kept += 1
    print(f"[OK] wrote {kept} rows -> {args.out_tsv}")

if __name__ == "__main__":
    main()
