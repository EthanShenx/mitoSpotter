#!/usr/bin/env python3
# Script to extract transcript IDs from GTF files and separate them into mitochondrial and nuclear lists
# Optionally filters for protein-coding transcripts only

import re, sys, argparse, os

# Parse GTF attributes into dict
# GTF format has attributes in column 9 as key-value pairs: key1 "value1"; key2 "value2"
def parse_attrs(s):
    d = {}
    # Use regex to extract all attribute key-value pairs from the GTF attributes string
    for m in re.finditer(r'(\S+)\s+"([^"]+)"', s):
        d[m.group(1)] = m.group(2)
    return d

def main():
    # Set up command-line argument parser
    ap = argparse.ArgumentParser(description="Make mito/nuclear transcript ID lists from GTF.")
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--outdir", required=True)
    # Default behavior: keep only protein-coding transcripts
    ap.add_argument(
        "--protein_coding_only",
        action="store_true",
        default=True,
        help="Keep only protein_coding transcripts (default: True). Use --no-protein_coding_only to disable."
    )
    # Allow user to disable protein-coding filter
    ap.add_argument(
        "--no-protein_coding_only",
        dest="protein_coding_only",
        action="store_false"
    )
    # Optional prefix for output filenames
    ap.add_argument(
        "--prefix",
        default="",
        help="Prefix for all output filenames (default: none)."
    )
    args = ap.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    # Initialize sets to store unique transcript IDs
    mito_ids, nuc_ids = set(), set()

    # Define all possible chromosome names that represent mitochondria
    mito_chroms = {"MT", "chrM", "M", "mt", "ChrM", "mitochondria", "MITOCHONDRIA"}
    # Open and parse the GTF file
    with open(args.gtf) as fh:
        for line in fh:
            # Skip empty lines and comment lines
            if not line or line.startswith("#"):
                continue
            # Split GTF line into tab-separated columns
            cols = line.rstrip("\n").split("\t")
            # GTF files should have 9 columns; skip malformed lines
            if len(cols) < 9:
                continue
            # Extract chromosome name (column 1), feature type (column 3), and attributes (column 9)
            chrom, feature, attrs = cols[0], cols[2], cols[8]
            # Only process lines that represent transcripts (not genes, exons, etc.)
            if feature != "transcript":
                continue
            # Parse the attributes column into a dictionary
            A = parse_attrs(attrs)
            # Extract the transcript_id attribute
            tid = A.get("transcript_id")
            if not tid:
                continue
            # Drop Ensembl version suffix: ENSTxxx.y -> ENSTxxx
            # This normalizes transcript IDs for compatibility across different Ensembl releases
            tid_novers = tid.split(".")[0]
            # Apply protein-coding filter if enabled
            if args.protein_coding_only:
                # Try to get biotype from transcript_biotype or gene_biotype attribute
                tb = A.get("transcript_biotype") or A.get("gene_biotype") or ""
                # Skip non-protein-coding transcripts
                if tb != "protein_coding":
                    continue
            # Classify transcript as mitochondrial or nuclear based on chromosome name
            if chrom in mito_chroms:
                mito_ids.add(tid_novers)
            else:
                nuc_ids.add(tid_novers)

    # Build filenames with optional prefix
    mito_fname = args.prefix + "ids_mito.txt"
    nuc_fname = args.prefix + "ids_nuclear.txt"
    # Create full output paths
    mito_path = os.path.join(args.outdir, mito_fname)
    nuc_path = os.path.join(args.outdir, nuc_fname)

    # Write sorted mitochondrial IDs to file (one per line)
    with open(mito_path, "w") as fo:
        fo.write("\n".join(sorted(mito_ids)) + ("\n" if mito_ids else ""))
    # Write sorted nuclear IDs to file (one per line)
    with open(nuc_path, "w") as fo:
        fo.write("\n".join(sorted(nuc_ids)) + ("\n" if nuc_ids else ""))

    # Print summary of results
    print(f"[OK] mito IDs: {len(mito_ids)} -> {mito_path}")
    print(f"[OK] nuclear IDs: {len(nuc_ids)} -> {nuc_path}")

# Standard Python idiom to run main() when script is executed directly
if __name__ == "__main__":
    main()