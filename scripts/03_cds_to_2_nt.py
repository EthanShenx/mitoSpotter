#!/usr/bin/env python3
# Convert CDS FASTA to per-transcript dinucleotide (2-nt) token sequences.
from Bio import SeqIO
import argparse, re, os

VALID = set("ACGT")

def clean_nt(s: str) -> str:
    # Uppercase, U->T, drop non-ACGT
    s = str(s).upper().replace("U", "T")
    return "".join([c for c in s if c in VALID])

def to_dinucs(s: str):
    # Overlapping dinucleotides with stride 1
    return [s[i:i+2] for i in range(len(s)-1)]

def main():
    ap = argparse.ArgumentParser(description="Emit dinucleotide (2-nt) sequences from CDS FASTA.")
    ap.add_argument("--cds_fasta", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--min_len", type=int, default=150, help="Minimum CDS length in nt.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    kept = 0
    with open(args.out_tsv, "w") as fo:
        for rec in SeqIO.parse(args.cds_fasta, "fasta"):
            tid = rec.id.split(".")[0]
            s = clean_nt(rec.seq)
            if len(s) < max(args.min_len, 2):
                continue
            di = to_dinucs(s)
            if not di:
                continue
            fo.write(tid + "\t" + " ".join(di) + "\n")
            kept += 1
    print(f"[OK] wrote {kept} rows -> {args.out_tsv}")

if __name__ == "__main__":
    main()

