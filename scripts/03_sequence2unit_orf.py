#!/usr/bin/env python3
from Bio import SeqIO
from Bio.Data import CodonTable
import argparse, os, random

def clean(seq):
    return "".join([c for c in str(seq).upper() if c in "ACGT"])

def translate_cds(seq, code):
    table = CodonTable.unambiguous_dna_by_id[1 if code=="nuclear" else 2]
    aa = ""
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        if len(codon) < 3: break
        if codon in table.stop_codons:
            aa += "*"
        elif codon in table.forward_table:
            aa += table.forward_table[codon]
        else:
            aa += "X"
    return aa

def write_lines(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")

def main():
    ap = argparse.ArgumentParser(description="Split CDS FASTA into train/holdout and translate to AA.")
    ap.add_argument("--fasta", required=True, help="Input CDS FASTA")
    ap.add_argument("--code", choices=["nuclear","mito"], required=True)
    ap.add_argument("--out_prefix", required=True, help="Output file prefix")
    ap.add_argument("--train_frac", type=float, default=0.7)
    args = ap.parse_args()

    records = list(SeqIO.parse(args.fasta, "fasta"))
    random.shuffle(records)

    split = int(len(records) * args.train_frac)
    train = records[:split]
    hold  = records[split:]

    train_lines = []
    hold_lines  = []

    for rec in train:
        cds = clean(rec.seq)
        aa = translate_cds(cds, args.code)
        train_lines.append(f"{rec.id} {aa}")

    for rec in hold:
        cds = clean(rec.seq)
        aa = translate_cds(cds, args.code)
        hold_lines.append(f"{rec.id} {aa}")

    write_lines(args.out_prefix + "_train_aa.txt", train_lines)
    write_lines(args.out_prefix + "_holdout_aa.txt", hold_lines)

    print(f"[OK] Total: {len(records)}, Train: {len(train)}, Holdout: {len(hold)}")
    print(f"[OUT] {args.out_prefix}_train_aa.txt")
    print(f"[OUT] {args.out_prefix}_holdout_aa.txt")

if __name__ == "__main__":
    main()
