#!/usr/bin/env python3
"""
Unified script for splitting FASTA (cDNA or CDS) by transcript IDs.
Modes:
  --mode cds   → process CDS FASTA
  --mode cdna  → process cDNA FASTA
"""

from Bio import SeqIO
import argparse, os, sys

def load_ids(p):
    return set(x.strip() for x in open(p) if x.strip())

def main():
    ap = argparse.ArgumentParser(description="Split FASTA (cDNA/CDS) by mito/nuclear transcript IDs.")
    ap.add_argument("--fasta", required=True, help="Input FASTA (cDNA or CDS).")
    ap.add_argument("--mito_ids", required=True, help="File containing mitochondrial transcript IDs.")
    ap.add_argument("--nuc_ids", required=True, help="File containing nuclear transcript IDs.")
    ap.add_argument("--outdir", required=True, help="Directory to write output files.")
    ap.add_argument("--mode", required=True, choices=["cdna","cds"],
                    help="Specify whether input FASTA is cDNA or CDS.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    mito_ids = load_ids(args.mito_ids)
    nuc_ids  = load_ids(args.nuc_ids)

    # Output filenames depend on mode
    if args.mode == "cdna":
        mito_out = os.path.join(args.outdir, "mito_cdna.fa")
        nuc_out  = os.path.join(args.outdir, "nuclear_cdna.fa")
    else:  # cds
        mito_out = os.path.join(args.outdir, "mito_cds.fa")
        nuc_out  = os.path.join(args.outdir, "nuclear_cds.fa")

    fm = open(mito_out, "w")
    fn = open(nuc_out, "w")

    km = kn = 0
    for rec in SeqIO.parse(args.fasta, "fasta"):
        tid = rec.id.split(".")[0]  # remove version
        if tid in mito_ids:
            SeqIO.write(rec, fm, "fasta")
            km += 1
        elif tid in nuc_ids:
            SeqIO.write(rec, fn, "fasta")
            kn += 1

    fm.close()
    fn.close()

    print(f"[OK] mode = {args.mode}")
    print(f"[OK] mito sequences: {km} -> {mito_out}")
    print(f"[OK] nuclear sequences: {kn} -> {nuc_out}")

if __name__ == "__main__":
    main()
