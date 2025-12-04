#!/bin/bash
# ============================================================================
# run_benchmark.sh
# Example commands to run the HMM benchmarking pipeline
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration - EDIT THESE PATHS FOR YOUR SYSTEM
# ============================================================================

# Project root directory
PROJECT_ROOT="/Users/coellearth/Library/Mobile Documents/com~apple~CloudDocs/mitoSpotter"

# Number of parallel workers (set to number of CPU cores for maximum speed)
N_WORKERS=4

# ============================================================================
# Step 0: Verify directory structure
# ============================================================================

echo "=============================================="
echo "Verifying project structure..."
echo "=============================================="

# Check required directories exist
required_dirs=(
    "$PROJECT_ROOT/out_dir/01_ids"
    "$PROJECT_ROOT/out_dir/03_unit/train"
    "$PROJECT_ROOT/out_dir/03_unit/holdout"
    "$PROJECT_ROOT/scripts"
)

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Required directory not found: $dir"
        exit 1
    fi
done

# Check required files exist
required_files=(
    "$PROJECT_ROOT/out_dir/01_ids/human_protein_coding_ids_mito.txt"
    "$PROJECT_ROOT/out_dir/01_ids/human_protein_coding_ids_nuclear.txt"
    "$PROJECT_ROOT/scripts/04_train_hmm_nt.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file not found: $file"
        exit 1
    fi
done

echo "✓ All required directories and files found"

# ============================================================================
# Step 1: Copy the modified decoder script to the project
# ============================================================================

echo ""
echo "=============================================="
echo "Setting up modified decoder script..."
echo "=============================================="

# Copy 05_decode_path_nt_v2.py to the scripts directory
cp 05_decode_path_nt_v2.py "$PROJECT_ROOT/scripts/"
echo "✓ Copied 05_decode_path_nt_v2.py to scripts/"

# ============================================================================
# Step 2: Run the benchmark pipeline
# ============================================================================

echo ""
echo "=============================================="
echo "Running benchmark pipeline..."
echo "=============================================="

python 06_benchmark_pipeline.py \
    --project_root "$PROJECT_ROOT" \
    --train_script "$PROJECT_ROOT/scripts/04_train_hmm_nt.py" \
    --decode_script "$PROJECT_ROOT/scripts/05_decode_path_nt_v2.py" \
    --n_workers "$N_WORKERS"

# ============================================================================
# Step 3: Generate visualization plots (optional)
# ============================================================================

echo ""
echo "=============================================="
echo "Generating visualization plots..."
echo "=============================================="

python 07_visualize_benchmark.py \
    --results_csv "$PROJECT_ROOT/out_dir/04_model/benchmark_results.csv" \
    --output_dir "$PROJECT_ROOT/out_dir/04_model/plots"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - Models:      $PROJECT_ROOT/out_dir/04_model/{1,2,3}nt/*"
echo "  - Predictions: $PROJECT_ROOT/out_dir/04_model/predictions/"
echo "  - CSV:         $PROJECT_ROOT/out_dir/04_model/benchmark_results.csv"
echo "  - Plots:       $PROJECT_ROOT/out_dir/04_model/plots/"
echo ""
