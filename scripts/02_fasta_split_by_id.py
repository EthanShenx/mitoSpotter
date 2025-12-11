#!/usr/bin/env python3
# Script to split a CDS FASTA file into mitochondrial and nuclear sequences based on provided ID lists

# Import BioPython's SeqIO for handling FASTA files
from Bio import SeqIO
# Import modules for command-line arguments, file operations, and system functions
import argparse, os, sys

# Function to load transcript IDs from a file into a set
def load_ids(p):
    # Read file, strip whitespace from each line, filter out empty lines, and return as a set
    return set(x.strip() for x in open(p) if x.strip())

def main():
    # Set up argument parser with description
    ap = argparse.ArgumentParser(description="Split CDS FASTA by mito/nuclear transcript IDs.")
    # Define required input FASTA file argument
    ap.add_argument("--fasta", required=True, help="Input CDS FASTA file.")
    # Define required mitochondrial IDs file argument
    ap.add_argument("--mito_ids", required=True, help="File containing mitochondrial transcript IDs.")
    # Define required nuclear IDs file argument
    ap.add_argument("--nuc_ids", required=True, help="File containing nuclear transcript IDs.")
    # Define required output directory argument
    ap.add_argument("--outdir", required=True, help="Directory to write output files.")
    # Define optional prefix argument for output filenames
    ap.add_argument(
        "--prefix",
        default="",
        help="Prefix for all output filenames (default: none)."
    )
    # Parse command-line arguments
    args = ap.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Load mitochondrial and nuclear transcript IDs from files into sets
    mito_ids = load_ids(args.mito_ids)
    nuc_ids  = load_ids(args.nuc_ids)

    # Output filenames with optional prefix
    # Construct mitochondrial output filename
    mito_name = args.prefix + "mito_cds.fa"
    # Construct nuclear output filename
    nuc_name  = args.prefix + "nuclear_cds.fa"

    # Create full paths for output files
    mito_out = os.path.join(args.outdir, mito_name)
    nuc_out  = os.path.join(args.outdir, nuc_name)

    # Open output file handles for writing
    fm = open(mito_out, "w")
    fn = open(nuc_out, "w")

    # Initialize counters for mitochondrial and nuclear sequences
    km = kn = 0
    # Iterate through each sequence record in the input FASTA file
    for rec in SeqIO.parse(args.fasta, "fasta"):
        # Extract transcript ID by removing version number (everything after the dot)
        tid = rec.id.split(".")[0]  # remove version
        # Check if transcript ID is in mitochondrial set
        if tid in mito_ids:
            # Write sequence to mitochondrial output file
            SeqIO.write(rec, fm, "fasta")
            # Increment mitochondrial counter
            km += 1
        # Check if transcript ID is in nuclear set
        elif tid in nuc_ids:
            # Write sequence to nuclear output file
            SeqIO.write(rec, fn, "fasta")
            # Increment nuclear counter
            kn += 1

    # Close output file handles
    fm.close()
    fn.close()

    # Print summary of sequences written to each file
    print(f"[OK] mito sequences: {km} -> {mito_out}")
    print(f"[OK] nuclear sequences: {kn} -> {nuc_out}")

# Standard Python idiom to run main() when script is executed directly
if __name__ == "__main__":
    main()