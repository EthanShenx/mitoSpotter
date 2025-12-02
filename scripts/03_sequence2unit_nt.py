#!/usr/bin/env python3

from Bio import SeqIO
import argparse, os

# ==============================
# Common utilities
# ==============================

def clean_nt(s):
    """Uppercase, convert U->T, remove non-ACGT."""
    return "".join([c for c in str(s).upper().replace("U", "T") if c in "ACGT"])


# ==============================
# Tokenization utilities
# ==============================

def tokenize_1nt(seq):
    return list(seq)

def tokenize_2nt(seq):
    return [seq[i:i+2] for i in range(len(seq) - 1)]

def tokenize_3nt(seq):
    L = (len(seq) // 3) * 3
    return [seq[i:i+3] for i in range(0, L, 3)]


# ==============================
# Main processing
# ==============================

def main():

    parser = argparse.ArgumentParser(
        description="Unified token generator: 1nt, 2nt, 3nt"
    )
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["1nt", "2nt", "3nt"])
    parser.add_argument("--train_tsv", required=True)
    parser.add_argument("--holdout_tsv", required=True)
    parser.add_argument("--train_frac", type=float, default=0.7)

    args = parser.parse_args()

    results = []

    for rec in SeqIO.parse(args.fasta, "fasta"):
        tid = rec.id.split(".")[0]

        seq = clean_nt(rec.seq)
        if not seq:
            continue

        # -----------------------
        # Modes
        # -----------------------

        if args.mode == "1nt":
            toks = tokenize_1nt(seq)

        elif args.mode == "2nt":
            toks = tokenize_2nt(seq)
            if not toks:
                continue

        elif args.mode == "3nt":
            toks = tokenize_3nt(seq)
            if not toks:
                continue

        # Build line
        line = tid + "\t" + " ".join(toks)
        results.append(line)

    # ==============================
    # Split train/holdout
    # ==============================

    total = len(results)
    split_idx = int(total * args.train_frac)
    if split_idx < 0:
        split_idx = 0
    elif split_idx > total:
        split_idx = total

    train_lines = results[:split_idx]
    hold_lines  = results[split_idx:]

    os.makedirs(os.path.dirname(args.train_tsv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.holdout_tsv) or ".", exist_ok=True)

    with open(args.train_tsv, "w") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open(args.holdout_tsv, "w") as f:
        for line in hold_lines:
            f.write(line + "\n")

    print(f"[OK] mode={args.mode}, total={total}, train={len(train_lines)}, holdout={len(hold_lines)}")
    print(f"train -> {args.train_tsv}")
    print(f"holdout -> {args.holdout_tsv}")

if __name__ == "__main__":
    main()