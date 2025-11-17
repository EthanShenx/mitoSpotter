#!/usr/bin/env python3
# Translate ORF into amino acid token sequences (per-transcript TSV),
# and split into train (first 70%) and holdout (last 30%).
# Output one row per transcript: <transcript_id>\tA A A ...

from Bio import SeqIO
import argparse, re, os, sys

# --- Codon tables (explicit) ---
# Standard nuclear code
STD_CODON_TO_AA = {
 "TTT":"F","TTC":"F","TTA":"L","TTG":"L","TCT":"S","TCC":"S","TCA":"S","TCG":"S",
 "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","TGT":"C","TGC":"C","TGA":"*","TGG":"W",
 "CTT":"L","CTC":"L","CTA":"L","CTG":"L","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
 "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
 "ATT":"I","ATC":"I","ATA":"I","ATG":"M","ACT":"T","ACC":"T","ACA":"T","ACG":"T",
 "AAT":"N","AAC":"N","AAA":"K","AAG":"K","AGT":"S","AGC":"S","AGA":"R","AGG":"R",
 "GTT":"V","GTC":"V","GTA":"V","GTG":"V","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
 "GAT":"D","GAC":"D","GAA":"E","GAG":"E","GGT":"G","GGC":"G","GGA":"G","GGG":"G"
}

# Vertebrate mitochondrial code (selected differences)
# Note: TGA -> W, AGA/AGG -> Stop, ATA -> M
MITO_CODON_TO_AA = STD_CODON_TO_AA.copy()
MITO_CODON_TO_AA.update({
 "TGA":"W", "AGA":"*", "AGG":"*", "ATA":"M"
})

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical AAs

# --- Utilities ---
def clean_nt(s):
    s = str(s).upper().replace("U", "T")
    return re.sub(r"[^ACGT]", "", s)

def find_longest_orf_nt(seq, stop_codons):
    # Find longest ORF across 3 frames (stop codons excluded)
    # No min length filter anymore.
    s = clean_nt(seq)
    best = ""
    for frame in (0, 1, 2):
        start = frame
        i = frame
        while i + 3 <= len(s):
            codon = s[i:i+3]
            if codon in stop_codons:
                if i - start > len(best):
                    best = s[start:i]
                start = i + 3
            i += 3
        # tail
        tail = s[start: len(s) - ((len(s) - start) % 3)]
        if len(tail) > len(best):
            best = tail
    return best  # may be empty string if no valid ORF

def nt_to_codons(nt):
    return [nt[i:i+3] for i in range(0, len(nt) - 2, 3)]

def codons_to_aa(codons, table):
    aas = []
    for c in codons:
        if len(c) != 3:
            return None
        aa = table.get(c, "X")
        if aa == "*" or aa == "X":  # stop or unknown -> stop translation or invalid
            # skip stop codons (do not include) but if unknown, abort
            if aa == "X":
                return None
            else:
                continue
        aas.append(aa)
    # filter non-canonical AA
    if not all(a in AA_VOCAB for a in aas):
        return None
    return aas

# --- CLI ---
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Translate longest ORF into AA token sequences, "
            "split into train (first 70%) and holdout (last 30%)."
        )
    )
    ap.add_argument("--fasta", required=True, help="Input cDNA or CDS FASTA.")
    ap.add_argument(
        "--genetic_code",
        choices=["standard", "vertebrate_mito", "auto"],
        default="auto",
        help=(
            "Which code to use for translation; auto tries mito stops first, "
            "then standard stops if needed."
        ),
    )
    # segment is kept only as an explicit 'orf' flag for compatibility
    ap.add_argument(
        "--segment",
        choices=["orf"],
        default="orf",
        help="Only 'orf' is supported: find longest ORF by stops then translate.",
    )
    # removed --min_nt and all length-based filtering
    ap.add_argument(
        "--train_tsv",
        required=True,
        help="Output TSV for training (first 70% of AA sequences).",
    )
    ap.add_argument(
        "--holdout_tsv",
        required=True,
        help="Output TSV for holdout (last 30% of AA sequences).",
    )
    ap.add_argument(
        "--train_frac",
        type=float,
        default=0.7,
        help="Fraction of sequences to use for training (default: 0.7).",
    )
    args = ap.parse_args()

    # For translation table: keep original behavior (standard vs non-standard/auto)
    table = STD_CODON_TO_AA if args.genetic_code == "standard" else MITO_CODON_TO_AA

    lines = []

    for rec in SeqIO.parse(args.fasta, "fasta"):
        tid = rec.id.split(".")[0]

        # Only ORF mode is supported now
        if args.genetic_code == "standard":
            stops = {"TAA", "TAG", "TGA"}
        elif args.genetic_code == "vertebrate_mito":
            stops = {"TAA", "TAG", "AGA", "AGG"}
        else:  # auto: try mito stops first
            stops = {"TAA", "TAG", "AGA", "AGG"}

        nt = find_longest_orf_nt(rec.seq, stops)

        if not nt and args.genetic_code == "auto":
            # fallback to standard stops
            nt = find_longest_orf_nt(rec.seq, {"TAA", "TAG", "TGA"})
        if not nt:
            # no ORF found
            continue

        cods = nt_to_codons(nt)
        aas = codons_to_aa(cods, table)
        if not aas:
            continue

        line = tid + "\t" + " ".join(aas)
        lines.append(line)

    total = len(lines)
    if total == 0:
        print("[WARN] no AA sequences passed filters; nothing to write.")
        return

    # --- split 70% / 30% ---
    split_idx = int(total * args.train_frac)
    if split_idx <= 0:
        split_idx = 0
    elif split_idx >= total:
        split_idx = total

    train_lines = lines[:split_idx]
    holdout_lines = lines[split_idx:]

    # --- write outputs ---
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

# python3 translate_orf_split_aa.py \
#   --fasta transcripts.fa \
#   --genetic_code standard \
#   --train_tsv out/train_aa_70.tsv \
#   --holdout_tsv out/holdout_aa_30.tsv \
#   --train_frac 0.7