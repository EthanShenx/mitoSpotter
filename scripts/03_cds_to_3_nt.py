#!/usr/bin/env python3
# Emit non-overlapping 3-nt tokens from CDS/ORF FASTA -> TSV (id \t AAA CCC ...).
from Bio import SeqIO
import argparse, re, os

VALID = set("ACGT")

def clean_nt(s):
    return "".join([c for c in str(s).upper().replace("U","T") if c in VALID])

def find_longest_orf_nt(seq, stops, min_nt=150):
    s = clean_nt(seq)
    best = ""
    for frame in (0,1,2):
        start = frame
        i = frame
        while i + 3 <= len(s):
            codon = s[i:i+3]
            if codon in stops:
                if i - start > len(best): best = s[start:i]
                start = i + 3
            i += 3
        tail = s[start: len(s) - ((len(s)-start) % 3)]
        if len(tail) > len(best): best = tail
    return best if len(best) >= min_nt else ""

def to_triplets_nonoverlap(s):
    L = (len(s)//3)*3
    return [s[i:i+3] for i in range(0, L, 3)]

def main():
    p = argparse.ArgumentParser(description="Emit non-overlapping 3-nt tokens from FASTA.")
    p.add_argument("--fasta", required=True)
    p.add_argument("--segment", choices=["cds","orf"], default="cds")
    p.add_argument("--code", choices=["standard","vertebrate_mito"], default="standard",
                   help="Relevant when --segment orf")
    p.add_argument("--min_nt", type=int, default=150)
    p.add_argument("--out_tsv", required=True)
    args = p.parse_args()

    stops_std = {"TAA","TAG","TGA"}
    stops_mito = {"TAA","TAG","AGA","AGG"}
    stops = stops_std if args.code == "standard" else stops_mito

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    kept = 0
    with open(args.out_tsv, "w") as fo:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            tid = rec.id.split(".")[0]
            if args.segment == "cds":
                s = clean_nt(rec.seq)
                if len(s) < args.min_nt: 
                    continue
            else:
                s = find_longest_orf_nt(rec.seq, stops, min_nt=args.min_nt)
                if not s: 
                    continue
            trip = to_triplets_nonoverlap(s)
            if not trip:
                continue
            fo.write(tid + "\t" + " ".join(trip) + "\n")
            kept += 1
    print(f"[OK] wrote {kept} rows -> {args.out_tsv}")

if __name__ == "__main__":
    main()

