#!/usr/bin/env python3
from Bio import SeqIO
import argparse, os, sys, re

STD_STOPS = {"TAA", "TAG", "TGA"}
VERTEBRATE_MITO_STOPS = {"TAA", "TAG", "AGA", "AGG"}

def find_longest_orf_nt(seq, stop_codons):
    """Find the longest ORF (nt) across 3 forward frames; return nt string (no min length filter)."""
    s = str(seq).upper().replace("U", "T")
    best = ("", -1, -1)  # (orf_nt, length, start_index)
    for frame in [0, 1, 2]:
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
            cand = s[orf_start: len(s) - ((len(s) - orf_start) % 3)]
            if len(cand) > best[1]:
                best = (cand, len(cand), orf_start)
    orf_nt = best[0]
    return orf_nt  # no min_nt threshold

def nt_to_codons(nt):
    return [nt[i:i+3] for i in range(0, len(nt) - 2, 3)]

def main():
    ap = argparse.ArgumentParser(description="Extract longest ORF and emit codon sequences, "
                                             "split into train (first 70%) and holdout (last 30%).")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--genetic_code", choices=["standard", "vertebrate_mito"], required=True)
    ap.add_argument("--train_tsv", required=True,
                    help="Output TSV for training (first 70% of ORFs).")
    ap.add_argument("--holdout_tsv", required=True,
                    help="Output TSV for holdout (last 30% of ORFs).")
    ap.add_argument("--train_frac", type=float, default=0.7,
                    help="Fraction of sequences to use for training (default: 0.7).")
    args = ap.parse_args()

    stops = STD_STOPS if args.genetic_code == "standard" else VERTEBRATE_MITO_STOPS

    lines = []
    for rec in SeqIO.parse(args.fasta, "fasta"):
        tid = rec.id.split(".")[0]
        orf_nt = find_longest_orf_nt(rec.seq, stops)
        if not orf_nt:
            # no valid ORF found (no ≥1 codon ORFs)
            continue
        codons = nt_to_codons(orf_nt)
        # Filter any partials or invalid codons
        codons = [c for c in codons if len(c) == 3 and re.fullmatch(r"[ACGT]{3}", c)]
        if not codons:
            continue
        line = tid + "\t" + " ".join(codons)
        lines.append(line)

    total = len(lines)
    if total == 0:
        print("[WARN] no ORFs passed; nothing to write.")
        return

    # 70% / 30% split (or general train_frac / rest)
    split_idx = int(total * args.train_frac)
    if split_idx <= 0:
        split_idx = 0
    elif split_idx >= total:
        split_idx = total

    train_lines = lines[:split_idx]
    holdout_lines = lines[split_idx:]

    os.makedirs(os.path.dirname(args.train_tsv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.holdout_tsv) or ".", exist_ok=True)

    with open(args.train_tsv, "w") as f_train:
        for line in train_lines:
            f_train.write(line + "\n")

    with open(args.holdout_tsv, "w") as f_hold:
        for line in holdout_lines:
            f_hold.write(line + "\n")

    print(f"[OK] total={total}, train={len(train_lines)}, holdout={len(holdout_lines)}")
    print(f"train -> {args.train_tsv}")
    print(f"holdout -> {args.holdout_tsv}")

if __name__ == "__main__":
    main()

# python3 extract_orf_codons_split.py \
#   --fasta transcripts.fa \
#   --genetic_code standard \
#   --train_tsv out/train_codons.tsv \
#   --holdout_tsv out/holdout_codons.tsv \
#   --train_frac 0.7
