#!/usr/bin/env python3
# Unified feature generator for HMM project:
# Supports: ORF-codons (3nt_orf), ORF-AA (aa)

from Bio import SeqIO
import argparse, os

# ==============================
# Common utilities
# ==============================

def clean_nt(s):
    """Uppercase, convert U->T, remove non-ACGT."""
    return "".join([c for c in str(s).upper().replace("U", "T") if c in "ACGT"])


# ==============================
# ORF detection
# ==============================

def find_longest_orf_nt(seq, stop_codons):
    """Find longest ORF across 3 forward frames, return nt string."""
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

        tail = s[start: len(s) - ((len(s) - start) % 3)]
        if len(tail) > len(best):
            best = tail

    return best


def nt_to_codons(nt):
    return [nt[i:i+3] for i in range(0, len(nt) - 2, 3)]


# ==============================
# AA translation tables
# ==============================

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

MITO_CODON_TO_AA = STD_CODON_TO_AA.copy()
MITO_CODON_TO_AA.update({"TGA":"W", "AGA":"*", "AGG":"*", "ATA":"M"})

AA_VOCAB = set(list("ACDEFGHIKLMNPQRSTVWY"))


def codons_to_aa(codons, table):
    aas = []
    for c in codons:
        if len(c) != 3:
            return None
        aa = table.get(c, "X")
        if aa == "X":
            return None
        if aa == "*":  # stop
            continue
        aas.append(aa)

    if not all(a in AA_VOCAB for a in aas):
        return None
    return aas


# ==============================
# Main processing
# ==============================

def main():

    parser = argparse.ArgumentParser(
        description="Unified token generator: 3nt_orf, aa"
    )
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["3nt_orf", "aa"])
    parser.add_argument("--genetic_code", required=True,
                        choices=["nuclear", "mito"],
                        help="Genetic code table (required): 'nuclear' or 'mito'")
    parser.add_argument("--train_tsv", required=True)
    parser.add_argument("--holdout_tsv", required=True)
    parser.add_argument("--train_frac", type=float, default=0.7)

    args = parser.parse_args()

    # Set translation table and stop codons based on genetic code
    if args.genetic_code == "nuclear":
        table = STD_CODON_TO_AA
        stops = {"TAA", "TAG", "TGA"}
    else:  # mito
        table = MITO_CODON_TO_AA
        stops = {"TAA", "TAG", "AGA", "AGG"}

    results = []

    for rec in SeqIO.parse(args.fasta, "fasta"):
        tid = rec.id.split(".")[0]

        seq = clean_nt(rec.seq)
        if not seq:
            continue

        # Find longest ORF
        nt_orf = find_longest_orf_nt(seq, stops)
        if not nt_orf:
            continue

        codons = nt_to_codons(nt_orf)

        if args.mode == "3nt_orf":
            toks = codons
        else:  # aa mode
            aas = codons_to_aa(codons, table)
            if not aas:
                continue
            toks = aas

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

    print(f"[OK] mode={args.mode}, genetic_code={args.genetic_code}, total={total}, train={len(train_lines)}, holdout={len(hold_lines)}")
    print(f"train -> {args.train_tsv}")
    print(f"holdout -> {args.holdout_tsv}")

if __name__ == "__main__":
    main()