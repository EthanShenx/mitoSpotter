#!/usr/bin/env python3
# Split CDS FASTA into mito/nuclear by transcript IDs (version-stripped).
from Bio import SeqIO
import argparse, os

def load_ids(p):
    return set(x.strip() for x in open(p) if x.strip())

def main():
    ap = argparse.ArgumentParser(description="Split CDS FASTA by mito/nuclear transcript IDs.")
    ap.add_argument("--cds_fasta", required=True)
    ap.add_argument("--mito_ids", required=True)
    ap.add_argument("--nuc_ids", required=True)
    ap.add_argument("--out_mito_fa", required=True)
    ap.add_argument("--out_nuc_fa", required=True)
    args = ap.parse_args()

    mito_ids = load_ids(args.mito_ids)
    nuc_ids  = load_ids(args.nuc_ids)
    os.makedirs(os.path.dirname(args.out_mito_fa) or ".", exist_ok=True)

    km = kn = 0
    with open(args.out_mito_fa, "w") as fm, open(args.out_nuc_fa, "w") as fn:
        for rec in SeqIO.parse(args.cds_fasta, "fasta"):
            tid = rec.id.split(".")[0]
            if tid in mito_ids:
                SeqIO.write(rec, fm, "fasta"); km += 1
            elif tid in nuc_ids:
                SeqIO.write(rec, fn, "fasta"); kn += 1
    print(f"[OK] mito CDS: {km} -> {args.out_mito_fa}")
    print(f"[OK] nuclear CDS: {kn} -> {args.out_nuc_fa}")

if __name__ == "__main__":
    main()
