#!/usr/bin/env python3
# Convert CDS FASTA to per-transcript dinucleotide (2-nt) token sequences,
# and split into train (first 70%) and holdout (last 30%).

from Bio import SeqIO
import argparse, os

VALID = set("ACGT")

def clean_nt(s: str) -> str:
    # Uppercase, U->T, drop non-ACGT
    s = str(s).upper().replace("U", "T")
    return "".join([c for c in s if c in VALID])

def to_dinucs(s: str):
    # Overlapping dinucleotides with stride 1
    return [s[i:i+2] for i in range(len(s) - 1)]

def main():
    ap = argparse.ArgumentParser(
        description="Emit dinucleotide (2-nt) sequences from CDS FASTA, "
                    "split into train (first 70%) and holdout (last 30%)."
    )
    ap.add_argument("--cds_fasta", required=True)
    ap.add_argument("--train_tsv", required=True,
                    help="Output TSV for training (first 70% of CDS).")
    ap.add_argument("--holdout_tsv", required=True,
                    help="Output TSV for holdout (last 30% of CDS).")
    ap.add_argument("--train_frac", type=float, default=0.7,
                    help="Fraction of sequences to use for training (default: 0.7).")
    args = ap.parse_args()

    # Collect all valid CDS dinucleotide lines
    lines = []
    for rec in SeqIO.parse(args.cds_fasta, "fasta"):
        tid = rec.id.split(".")[0]
        s = clean_nt(rec.seq)
        di = to_dinucs(s)
        if not di:
            # skip sequences that cannot form any dinucleotide
            continue
        line = tid + "\t" + " ".join(di)
        lines.append(line)

    total = len(lines)
    if total == 0:
        print("[WARN] no CDS produced any dinucleotides; nothing to write.")
        return

    # Split into train (first train_frac) and holdout (rest)
    split_idx = int(total * args.train_frac)
    if split_idx <= 0:
        split_idx = 0
    elif split_idx >= total:
        split_idx = total

    train_lines = lines[:split_idx]
    holdout_lines = lines[split_idx:]

    # Make sure output directories exist
    os.makedirs(os.path.dirname(args.train_tsv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.holdout_tsv) or ".", exist_ok=True)

    # Write train set
    with open(args.train_tsv, "w") as f_train:
        for line in train_lines:
            f_train.write(line + "\n")

    # Write holdout set
    with open(args.holdout_tsv, "w") as f_hold:
        for line in holdout_lines:
            f_hold.write(line + "\n")

    print(f"[OK] total={total}, train={len(train_lines)}, holdout={len(holdout_lines)}")
    print(f"train -> {args.train_tsv}")
    print(f"holdout -> {args.holdout_tsv}")

if __name__ == "__main__":
    main()

# python3 split_cds_2nt.py \
#   --cds_fasta cds.fa \
#   --train_tsv out/train_2nt.tsv \
#   --holdout_tsv out/holdout_2nt.tsv \
#   --train_frac 0.7