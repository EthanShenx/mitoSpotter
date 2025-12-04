#!/bin/bash

set -e 

# Project root dir
PROJECT_ROOT="/Users/coellearth/Library/Mobile Documents/com~apple~CloudDocs/mitoSpotter"

# Number of parallel workers (set to number of CPU cores for maximum speed)
N_WORKERS=2

echo "=============================================="
echo "Verifying project structure..."
echo "=============================================="

# Check
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

# Check
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

echo "All required directories and files found."

# ==================================
# ==================================
# ==================================

echo ""
echo "=============================================="
echo "Running benchmark pipeline..."
echo "=============================================="

python 06_benchmark_pipeline.py \
    --project_root "$PROJECT_ROOT" \
    --train_script "$PROJECT_ROOT/scripts/04_train_hmm_nt.py" \
    --decode_script "$PROJECT_ROOT/scripts/05_decode_path_nt_v2.py" \
    --n_workers "$N_WORKERS"

echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - 04_Models:      $PROJECT_ROOT/out_dir/04_model/{1,2,3}nt/*"
echo "  - 05_Predictions: $PROJECT_ROOT/out_dir/04_model/predictions/"
echo "  - CSV:         $PROJECT_ROOT/out_dir/04_model/benchmark_results.csv"
echo ""