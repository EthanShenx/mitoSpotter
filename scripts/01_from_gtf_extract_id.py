#!/usr/bin/env python3
import re, sys, argparse, os

# Parse GTF attributes into dict
def parse_attrs(s):
    d = {}
    for m in re.finditer(r'(\S+)\s+"([^"]+)"', s):
        d[m.group(1)] = m.group(2)
    return d

def main():
    ap = argparse.ArgumentParser(description="Make mito/nuclear transcript ID lists from GTF.")
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--protein_coding_only",
        action="store_true",
        default=True,
        help="Keep only protein_coding transcripts (default: True). Use --no-protein_coding_only to disable."
    )
    ap.add_argument(
        "--no-protein_coding_only",
        dest="protein_coding_only",
        action="store_false"
    )
    ap.add_argument(
        "--prefix",
        default="",
        help="Prefix for all output filenames (default: none)."
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    mito_ids, nuc_ids = set(), set()

    mito_chroms = {"MT", "chrM", "M", "mt", "ChrM", "mitochondria", "MITOCHONDRIA"}
    with open(args.gtf) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            chrom, feature, attrs = cols[0], cols[2], cols[8]
            if feature != "transcript":
                continue
            A = parse_attrs(attrs)
            tid = A.get("transcript_id")
            if not tid:
                continue
            # Drop Ensembl version suffix: ENSTxxx.y -> ENSTxxx
            tid_novers = tid.split(".")[0]
            if args.protein_coding_only:
                tb = A.get("transcript_biotype") or A.get("gene_biotype") or ""
                if tb != "protein_coding":
                    continue
            if chrom in mito_chroms:
                mito_ids.add(tid_novers)
            else:
                nuc_ids.add(tid_novers)

    # Build filenames with optional prefix
    mito_fname = args.prefix + "ids_mito.txt"
    nuc_fname = args.prefix + "ids_nuclear.txt"
    mito_path = os.path.join(args.outdir, mito_fname)
    nuc_path = os.path.join(args.outdir, nuc_fname)

    with open(mito_path, "w") as fo:
        fo.write("\n".join(sorted(mito_ids)) + ("\n" if mito_ids else ""))
    with open(nuc_path, "w") as fo:
        fo.write("\n".join(sorted(nuc_ids)) + ("\n" if nuc_ids else ""))

    print(f"[OK] mito IDs: {len(mito_ids)} -> {mito_path}")
    print(f"[OK] nuclear IDs: {len(nuc_ids)} -> {nuc_path}")

if __name__ == "__main__":
    main()
