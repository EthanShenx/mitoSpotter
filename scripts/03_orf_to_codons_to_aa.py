#!/usr/bin/env python3
# Translate ORF or CDS into amino acid token sequences (per-transcript TSV).
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

def find_longest_orf_nt(seq, stop_codons, min_nt=150):
    # Find longest ORF across 3 frames (stop codons excluded)
    s = clean_nt(seq)
    best = ""
    for frame in (0,1,2):
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
        tail = s[start: len(s) - ((len(s)-start) % 3)]
        if len(tail) > len(best): best = tail
    return best if len(best) >= min_nt else ""

def nt_to_codons(nt):
    return [nt[i:i+3] for i in range(0, len(nt)-2, 3)]

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
    ap = argparse.ArgumentParser(description="Translate CDS/ORF into AA token sequences.")
    ap.add_argument("--fasta", required=True, help="Input cDNA or CDS FASTA.")
    ap.add_argument("--genetic_code", choices=["standard","vertebrate_mito","auto"], default="auto",
                    help="Which code to use for translation; auto tries mito if seq from chrM earlier (not implemented) - use explicit.")
    ap.add_argument("--segment", choices=["cds","orf"], default="orf",
                    help="cds=translate full cleaned input; orf=find longest ORF by stops then translate.")
    ap.add_argument("--min_nt", type=int, default=150)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    table = STD_CODON_TO_AA if args.genetic_code == "standard" else MITO_CODON_TO_AA

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    kept = 0
    with open(args.out_tsv, "w") as fo:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            tid = rec.id.split(".")[0]
            if args.segment == "cds":
                nt = clean_nt(rec.seq)
                if len(nt) < args.min_nt: 
                    continue
            else:
                # choose stop set based on requested genetic code when segment==orf
                if args.genetic_code == "standard":
                    stops = {"TAA","TAG","TGA"}
                elif args.genetic_code == "vertebrate_mito":
                    stops = {"TAA","TAG","AGA","AGG"}
                else:
                    # auto: attempt vertebrate mito stops first, then standard
                    stops = {"TAA","TAG","AGA","AGG"}
                nt = find_longest_orf_nt(rec.seq, stops, min_nt=args.min_nt)
                if not nt and args.genetic_code == "auto":
                    # fallback to standard stops
                    nt = find_longest_orf_nt(rec.seq, {"TAA","TAG","TGA"}, min_nt=args.min_nt)
                if not nt:
                    continue
            cods = nt_to_codons(nt)
            aas = codons_to_aa(cods, table)
            if not aas: 
                continue
            fo.write(tid + "\t" + " ".join(aas) + "\n")
            kept += 1
    print(f"[OK] wrote {kept} aa rows -> {args.out_tsv}")

if __name__ == "__main__":
    main()

