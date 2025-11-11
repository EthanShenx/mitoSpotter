#!/usr/bin/env python3
from Bio import SeqIO
import argparse, os, sys, re

STD_STOPS = {"TAA","TAG","TGA"}
VERTEBRATE_MITO_STOPS = {"TAA","TAG","AGA","AGG"}

def find_longest_orf_nt(seq, stop_codons, min_nt=150):
    """Find the longest ORF (nt) across 3 forward frames; return nt string length>=min_nt."""
    s = str(seq).upper().replace("U","T")
    best = ("", -1, -1)  # (orf_nt, frame, start_index)
    for frame in [0,1,2]:
        orf_start = frame
        i = frame
        while i + 3 <= len(s):
            codon = s[i:i+3]
            if any(c not in "ACGT" for c in codon):
                # Reset when N's appear
                if i - orf_start >= 3 and i - orf_start > best[1]:
                    cand = s[orf_start:i]
                    if len(cand) > best[1]:
                        best = (cand, len(cand), orf_start)
                orf_start = i + 3
                i += 3
                continue
            if codon in stop_codons:
                # Close ORF before stop (stop not included)
                if i - orf_start >= 3 and i - orf_start > best[1]:
                    cand = s[orf_start:i]
                    if len(cand) > best[1]:
                        best = (cand, len(cand), orf_start)
                orf_start = i + 3
            i += 3
        # tail
        if len(s) - orf_start >= 3 and (len(s) - orf_start) > best[1]:
            cand = s[orf_start: len(s) - ((len(s)-orf_start)%3)]
            if len(cand) > best[1]:
                best = (cand, len(cand), orf_start)
    orf_nt = best[0]
    if len(orf_nt) >= min_nt:
        return orf_nt
    return ""

def nt_to_codons(nt):
    return [nt[i:i+3] for i in range(0, len(nt)-2, 3)]

def main():
    ap = argparse.ArgumentParser(description="Extract ORF and emit codon sequences.")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--genetic_code", choices=["standard","vertebrate_mito"], required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--min_orf_nt", type=int, default=150, help="Minimum ORF length in nt.")
    args = ap.parse_args()

    stops = STD_STOPS if args.genetic_code == "standard" else VERTEBRATE_MITO_STOPS

    kept = 0
    with open(args.out_tsv, "w") as fo:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            tid = rec.id.split(".")[0]
            orf_nt = find_longest_orf_nt(rec.seq, stops, min_nt=args.min_orf_nt)
            if not orf_nt:
                continue
            codons = nt_to_codons(orf_nt)
            # Filter any partials or invalid codons
            codons = [c for c in codons if len(c)==3 and re.fullmatch(r"[ACGT]{3}", c)]
            if not codons:
                continue
            fo.write(tid + "\t" + " ".join(codons) + "\n")
            kept += 1
    print(f"[OK] wrote {kept} codon rows -> {args.out_tsv}")

if __name__ == "__main__":
    main()

