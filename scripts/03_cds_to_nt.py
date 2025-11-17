#!/usr/bin/env python3
# Convert CDS FASTA to per-transcript base sequences (A/C/G/T) for 1-nt HMM,
# and split into train (first 70%) and holdout (last 30%).

from Bio import SeqIO
import argparse, re, os

def nts_from_cds(seq):
    s = str(seq).upper().replace("U", "T")
    s = re.sub(r"[^ACGT]", "", s)
    return s  # no min_len filter anymore

def main():
    ap = argparse.ArgumentParser(
        description="Emit per-transcript base sequences from CDS FASTA, "
                    "split into train (first 70%) and holdout (last 30%)."
    )
    ap.add_argument("--cds_fasta", required=True,
                    help="Input CDS FASTA file.")
    ap.add_argument("--train_tsv", required=True,
                    help="Output TSV for training (first 70% of sequences).")
    ap.add_argument("--holdout_tsv", required=True,
                    help="Output TSV for holdout (last 30% of sequences).")
    ap.add_argument("--train_frac", type=float, default=0.7,
                    help="Fraction of sequences to use for training (default: 0.7).")
    args = ap.parse_args()

    lines = []
    for rec in SeqIO.parse(args.cds_fasta, "fasta"):
        tid = rec.id.split(".")[0]
        nts = nts_from_cds(rec.seq)
        if not nts:
            # if sequence becomes empty after cleaning, skip
            continue
        line = tid + "\t" + " ".join(list(nts))
        lines.append(line)

    total = len(lines)
    if total == 0:
        print("[WARN] no sequences passed cleaning; nothing to write.")
        return

    # split first train_frac as train, rest as holdout
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

# python3 cds_1nt_split.py \
#   --cds_fasta cds.fa \
#   --train_tsv out/train_1nt.tsv \
#   --holdout_tsv out/holdout_1nt.tsv \
#   --train_frac 0.7