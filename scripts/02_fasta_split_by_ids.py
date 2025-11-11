#!/usr/bin/env python3
from Bio import SeqIO
import argparse, os, sys

def load_ids(p):
    return set(x.strip() for x in open(p) if x.strip())

def main():
    ap = argparse.ArgumentParser(description="Split cDNA FASTA by given transcript ID lists.")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--mito_ids", required=True)
    ap.add_argument("--nuc_ids", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    mito_ids = load_ids(args.mito_ids)
    nuc_ids  = load_ids(args.nuc_ids)

    mito_out = os.path.join(args.outdir, "mito_cdna.fa")
    nuc_out  = os.path.join(args.outdir, "nuclear_cdna.fa")

    mito_handle = open(mito_out, "w")
    nuc_handle  = open(nuc_out, "w")

    # Iterate FASTA once for efficiency
    kept_mito = kept_nuc = 0
    for rec in SeqIO.parse(args.fasta, "fasta"):
        tid = rec.id.split(".")[0]  # drop version
        if tid in mito_ids:
            SeqIO.write(rec, mito_handle, "fasta")
            kept_mito += 1
        elif tid in nuc_ids:
            SeqIO.write(rec, nuc_handle, "fasta")
            kept_nuc += 1

    mito_handle.close()
    nuc_handle.close()
    print(f"[OK] mito sequences: {kept_mito} -> {mito_out}")
    print(f"[OK] nuclear sequences: {kept_nuc} -> {nuc_out}")

if __name__ == "__main__":
    main()

