#!/usr/bin/env python3
# Emit non-overlapping 3-nt tokens from CDS FASTA -> TSV,
# and split into train (first 70%) and holdout (last 30%).

from Bio import SeqIO
import argparse, os

VALID = set("ACGT")

def clean_nt(s):
    return "".join([c for c in str(s).upper().replace("U", "T") if c in VALID])

def to_triplets_nonoverlap(s):
    L = (len(s) // 3) * 3
    return [s[i:i+3] for i in range(0, L, 3)]

def main():
    p = argparse.ArgumentParser(
        description="Emit non-overlapping 3-nt tokens from CDS FASTA, "
                    "split into train (first 70%) and holdout (last 30%)."
    )
    p.add_argument("--fasta", required=True,
                   help="Input CDS FASTA file.")
    p.add_argument("--train_tsv", required=True,
                   help="Output TSV for training (first 70% of sequences).")
    p.add_argument("--holdout_tsv", required=True,
                   help="Output TSV for holdout (last 30% of sequences).")
    p.add_argument("--train_frac", type=float, default=0.7,
                   help="Fraction of sequences to use for training (default: 0.7).")
    args = p.parse_args()

    lines = []
    for rec in SeqIO.parse(args.fasta, "fasta"):
        tid = rec.id.split(".")[0]

        # Treat all input as CDS
        s = clean_nt(rec.seq)

        trip = to_triplets_nonoverlap(s)
        if not trip:
            continue

        line = tid + "\t" + " ".join(trip)
        lines.append(line)

    total = len(lines)
    if total == 0:
        print("[WARN] no sequences passed filters; nothing to write.")
        return

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

# python3 cds_triplets_split.py \
#   --fasta cds.fa \
#   --train_tsv out/train_3nt.tsv \
#   --holdout_tsv out/holdout_3nt.tsv \
#   --train_frac 0.7
