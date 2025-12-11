#!/usr/bin/env python3

# Import BioPython's SeqIO for reading FASTA files
from Bio import SeqIO
import argparse, os

# ==============================
# Common utilities
# ==============================

def clean_nt(s):
    """Uppercase, convert U->T, remove non-ACGT."""
    # Converts sequence to uppercase, replaces RNA (U) with DNA (T)
    # Filters out any characters that aren't A, C, G, or T
    return "".join([c for c in str(s).upper().replace("U", "T") if c in "ACGT"])


# ==============================
# Tokenization utilities
# ==============================

def tokenize_1nt(seq):
    # Splits sequence into individual nucleotides (single characters)
    return list(seq)

def tokenize_2nt(seq):
    # Creates overlapping dinucleotide tokens (sliding window of size 2)
    # Example: "ACGT" -> ["AC", "CG", "GT"]
    return [seq[i:i+2] for i in range(len(seq) - 1)]

def tokenize_3nt(seq):
    # Creates non-overlapping trinucleotide tokens (codon-like units)
    # Truncates sequence to nearest multiple of 3
    # Example: "ACGTAC" -> ["ACG", "TAC"]
    L = (len(seq) // 3) * 3
    return [seq[i:i+3] for i in range(0, L, 3)]


# ==============================
# Main processing
# ==============================

def main():

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Unified token generator: 1nt, 2nt, 3nt"
    )
    # Input FASTA file containing sequences
    parser.add_argument("--fasta", required=True)
    # Tokenization mode: single nucleotides, dinucleotides, or trinucleotides
    parser.add_argument("--mode", required=True,
                        choices=["1nt", "2nt", "3nt"])
    # Output file for training data
    parser.add_argument("--train_tsv", required=True)
    # Output file for holdout/validation data
    parser.add_argument("--holdout_tsv", required=True)
    # Fraction of data to use for training (default 70%)
    parser.add_argument("--train_frac", type=float, default=0.7)

    args = parser.parse_args()

    # Store all processed sequences
    results = []

    # Process each sequence record in the FASTA file
    for rec in SeqIO.parse(args.fasta, "fasta"):
        # Extract transcript ID (everything before first dot)
        tid = rec.id.split(".")[0]

        # Clean the sequence (normalize and filter nucleotides)
        seq = clean_nt(rec.seq)
        # Skip empty sequences
        if not seq:
            continue

        # -----------------------
        # Modes
        # -----------------------

        # Apply the selected tokenization mode
        if args.mode == "1nt":
            toks = tokenize_1nt(seq)

        elif args.mode == "2nt":
            toks = tokenize_2nt(seq)
            # Skip if no tokens generated (sequence too short)
            if not toks:
                continue

        elif args.mode == "3nt":
            toks = tokenize_3nt(seq)
            # Skip if no tokens generated (sequence too short)
            if not toks:
                continue

        # Create tab-separated line: transcript_id <tab> space-separated tokens
        line = tid + "\t" + " ".join(toks)
        results.append(line)

    # ==============================
    # Split train/holdout
    # ==============================

    # Calculate split point based on train_frac
    total = len(results)
    split_idx = int(total * args.train_frac)
    # Ensure split index is within valid bounds
    if split_idx < 0:
        split_idx = 0
    elif split_idx > total:
        split_idx = total

    # Split results into training and holdout sets
    train_lines = results[:split_idx]
    hold_lines  = results[split_idx:]

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.train_tsv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.holdout_tsv) or ".", exist_ok=True)

    # Write training data to file
    with open(args.train_tsv, "w") as f:
        for line in train_lines:
            f.write(line + "\n")

    # Write holdout data to file
    with open(args.holdout_tsv, "w") as f:
        for line in hold_lines:
            f.write(line + "\n")

    # Print summary statistics
    print(f"[OK] mode={args.mode}, total={total}, train={len(train_lines)}, holdout={len(hold_lines)}")
    print(f"train -> {args.train_tsv}")
    print(f"holdout -> {args.holdout_tsv}")

# Standard Python idiom: run main() when script is executed directly
if __name__ == "__main__":
    main()