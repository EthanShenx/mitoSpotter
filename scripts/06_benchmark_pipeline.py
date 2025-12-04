#!/usr/bin/env python3
"""
Benchmarking Pipeline for HMM-based Mitochondrial vs Nuclear CDS Classifier

This script runs a rigorous benchmark of 15 HMM model variants:
- 3 unit sizes (1nt, 2nt, 3nt)
- 5 training regimes per unit size:
  (a) pure_em_20:       EM 20 iters, Viterbi 0
  (b) pure_viterbi_20:  EM 0, Viterbi 20
  (c) hybrid_em5_vit15: EM 5, Viterbi 15
  (d) hybrid_em10_vit10: EM 10, Viterbi 10
  (e) hybrid_em15_vit5: EM 15, Viterbi 5

For each variant, the pipeline:
1. Trains the HMM model, measuring wall-clock time and peak memory
2. Decodes holdout sequences using Viterbi
3. Computes classification metrics (accuracy, precision, recall, F1)
4. Aggregates results into a final CSV

Usage:
    python 06_benchmark_pipeline.py \
        --project_root /path/to/mitoSpotter \
        --output_dir /path/to/mitoSpotter/out_dir/04_model \
        --n_workers 4

Author: Benchmarking pipeline for mitoSpotter project
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingRegime:
    """Configuration for a single training regime."""
    name: str
    train_method: str  # "em", "viterbi", or "hybrid"
    n_em_iter: int
    n_viterbi_iter: int


# Define all 5 training regimes (total 20 iterations each)
TRAINING_REGIMES: List[TrainingRegime] = [
    TrainingRegime("pure_em_20", "em", 20, 0),
    TrainingRegime("pure_viterbi_20", "viterbi", 0, 20),
    TrainingRegime("hybrid_em5_vit15", "hybrid", 5, 15),
    TrainingRegime("hybrid_em10_vit10", "hybrid", 10, 10),
    TrainingRegime("hybrid_em15_vit5", "hybrid", 15, 5),
]

UNIT_SIZES = [1, 2, 3]


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single model variant."""
    unit_size: int
    regime_name: str
    train_method: str
    n_em_iter: int
    n_viterbi_iter: int
    
    # Training metrics
    train_time_sec: float = 0.0
    train_max_mem_mb: float = 0.0
    
    # Decoding metrics
    decode_time_sec: float = 0.0
    decode_max_mem_mb: float = 0.0
    
    # Dataset sizes
    n_mito_holdout: int = 0
    n_nuclear_holdout: int = 0
    
    # Confusion matrix (M = mitochondrial is positive)
    TP: int = 0  # Predicted M, Actual M
    TN: int = 0  # Predicted N, Actual N
    FP: int = 0  # Predicted M, Actual N
    FN: int = 0  # Predicted N, Actual M
    
    # Derived metrics
    accuracy: float = 0.0
    precision_M: float = 0.0
    recall_M: float = 0.0
    f1_M: float = 0.0
    
    # Status
    status: str = "pending"
    error_message: str = ""


# ============================================================================
# Utility Functions
# ============================================================================

def get_peak_memory_mb() -> float:
    """
    Get peak memory usage of child processes in MB.
    
    On Linux: ru_maxrss is in KB
    On macOS: ru_maxrss is in bytes
    """
    rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if platform.system() == "Darwin":
        return rusage.ru_maxrss / (1024 * 1024)
    else:
        return rusage.ru_maxrss / 1024


def reset_rusage_baseline():
    """
    Get current child rusage to use as baseline.
    Returns max RSS so we can compute delta after subprocess.
    """
    rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if platform.system() == "Darwin":
        return rusage.ru_maxrss / (1024 * 1024)
    else:
        return rusage.ru_maxrss / 1024


def run_subprocess_timed(cmd: List[str], cwd: Optional[str] = None) -> Tuple[float, float, int, str, str]:
    """
    Run a subprocess and measure wall-clock time and peak memory.
    
    Returns:
        (elapsed_sec, peak_mem_mb, returncode, stdout, stderr)
    """
    baseline_mem = reset_rusage_baseline()
    
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    elapsed = time.perf_counter() - start
    
    peak_mem = get_peak_memory_mb()
    # Memory delta (approximate - child processes accumulate)
    mem_used = max(0, peak_mem - baseline_mem) if baseline_mem > 0 else peak_mem
    
    return elapsed, mem_used, result.returncode, result.stdout, result.stderr


def load_ground_truth_ids(ids_dir: Path) -> Tuple[set, set]:
    """
    Load ground truth IDs from ID files.
    
    Returns:
        (mito_ids, nuclear_ids) as sets
    """
    mito_file = ids_dir / "human_protein_coding_ids_mito.txt"
    nuclear_file = ids_dir / "human_protein_coding_ids_nuclear.txt"
    
    mito_ids = set()
    nuclear_ids = set()
    
    with open(mito_file) as f:
        for line in f:
            line = line.strip()
            if line:
                mito_ids.add(line)
    
    with open(nuclear_file) as f:
        for line in f:
            line = line.strip()
            if line:
                nuclear_ids.add(line)
    
    return mito_ids, nuclear_ids


def parse_holdout_tsv(tsv_path: Path) -> List[Tuple[str, str]]:
    """
    Parse holdout TSV file.
    
    Returns:
        List of (seq_id, sequence_tokens_string)
    """
    sequences = []
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                seq_id, tokens = parts
                sequences.append((seq_id, tokens))
    return sequences


def parse_predictions_tsv(tsv_path: Path) -> Dict[str, str]:
    """
    Parse decoder output TSV.
    
    Returns:
        Dict mapping seq_id -> call ("nuclear" or "mitochondrial")
    """
    predictions = {}
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 5:
                seq_id = parts[0]
                call = parts[4]  # call column
                predictions[seq_id] = call
    return predictions


def compute_metrics(result: BenchmarkResult) -> None:
    """Compute derived metrics from confusion matrix."""
    total = result.TP + result.TN + result.FP + result.FN
    
    if total > 0:
        result.accuracy = (result.TP + result.TN) / total
    
    # Precision for M (mitochondrial)
    if (result.TP + result.FP) > 0:
        result.precision_M = result.TP / (result.TP + result.FP)
    
    # Recall for M (mitochondrial)
    if (result.TP + result.FN) > 0:
        result.recall_M = result.TP / (result.TP + result.FN)
    
    # F1 for M
    if (result.precision_M + result.recall_M) > 0:
        result.f1_M = 2 * (result.precision_M * result.recall_M) / (result.precision_M + result.recall_M)


# ============================================================================
# Core Benchmark Functions
# ============================================================================

def train_model(
    train_script: Path,
    regime: TrainingRegime,
    unit_size: int,
    nuclear_tsv: Path,
    mito_tsv: Path,
    output_dir: Path,
    n_workers: int = 1
) -> Tuple[float, float, bool, str]:
    """
    Train a single HMM model variant.
    
    Returns:
        (train_time_sec, train_max_mem_mb, success, error_msg)
    """
    model_dir = output_dir / f"{unit_size}nt" / regime.name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_json = model_dir / "model.json"
    vocab_json = model_dir / "vocab.json"
    states_json = model_dir / "states.json"
    
    cmd = [
        sys.executable, str(train_script),
        "--nuclear_nt_tsv", str(nuclear_tsv),
        "--mito_nt_tsv", str(mito_tsv),
        "--out_model_json", str(model_json),
        "--out_vocab_json", str(vocab_json),
        "--out_states_json", str(states_json),
        "--ngram", str(unit_size),
        "--train_method", regime.train_method,
        "--n_em_iter", str(regime.n_em_iter),
        "--n_viterbi_iter", str(regime.n_viterbi_iter),
        "--n_workers", str(n_workers),
    ]
    
    elapsed, mem_mb, retcode, stdout, stderr = run_subprocess_timed(cmd)
    
    success = retcode == 0 and model_json.exists()
    error_msg = "" if success else f"Return code: {retcode}\n{stderr}"
    
    return elapsed, mem_mb, success, error_msg


def decode_holdout(
    decode_script: Path,
    unit_size: int,
    regime: TrainingRegime,
    model_dir: Path,
    holdout_mito_tsv: Path,
    holdout_nuclear_tsv: Path,
    predictions_dir: Path,
    min_len: int = 0
) -> Tuple[float, float, Path, bool, str]:
    """
    Decode holdout sequences using a trained model.
    
    Returns:
        (decode_time_sec, decode_max_mem_mb, predictions_path, success, error_msg)
    """
    model_json = model_dir / "model.json"
    vocab_json = model_dir / "vocab.json"
    states_json = model_dir / "states.json"
    
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_tsv = predictions_dir / f"{unit_size}nt_{regime.name}_pred.tsv"
    
    # Collect all holdout sequences
    mito_seqs = parse_holdout_tsv(holdout_mito_tsv)
    nuclear_seqs = parse_holdout_tsv(holdout_nuclear_tsv)
    all_seqs = mito_seqs + nuclear_seqs
    
    # Build command with all sequences
    cmd = [
        sys.executable, str(decode_script),
        "--method", "viterbi",
        "--ngram", str(unit_size),
        "--model_json", str(model_json),
        "--vocab_json", str(vocab_json),
        "--states_json", str(states_json),
        "--out_tsv", str(pred_tsv),
        "--min_len", str(min_len),
    ]
    
    # Add each sequence as --seq and --seq_id
    for seq_id, tokens in all_seqs:
        cmd.extend(["--seq", tokens, "--seq_id", seq_id])
    
    elapsed, mem_mb, retcode, stdout, stderr = run_subprocess_timed(cmd)
    
    success = retcode == 0 and pred_tsv.exists()
    error_msg = "" if success else f"Return code: {retcode}\n{stderr}"
    
    return elapsed, mem_mb, pred_tsv, success, error_msg


def evaluate_predictions(
    predictions_path: Path,
    mito_ids: set,
    nuclear_ids: set,
    result: BenchmarkResult
) -> None:
    """
    Evaluate predictions against ground truth and populate result metrics.
    """
    predictions = parse_predictions_tsv(predictions_path)
    
    for seq_id, pred_call in predictions.items():
        actual_mito = seq_id in mito_ids
        actual_nuclear = seq_id in nuclear_ids
        
        if not (actual_mito or actual_nuclear):
            # Unknown ID, skip
            continue
        
        pred_mito = (pred_call == "mitochondrial")
        
        if actual_mito:
            result.n_mito_holdout += 1
            if pred_mito:
                result.TP += 1
            else:
                result.FN += 1
        else:  # actual_nuclear
            result.n_nuclear_holdout += 1
            if pred_mito:
                result.FP += 1
            else:
                result.TN += 1
    
    compute_metrics(result)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_benchmark(
    project_root: Path,
    output_dir: Path,
    train_script: Path,
    decode_script: Path,
    n_workers: int = 1,
    verbose: bool = True
) -> List[BenchmarkResult]:
    """
    Run the complete benchmark pipeline.
    
    Returns:
        List of BenchmarkResult for all 15 model variants
    """
    # Resolve paths
    data_dir = project_root / "out_dir" / "03_unit"
    train_dir = data_dir / "train"
    holdout_dir = data_dir / "holdout"
    ids_dir = project_root / "out_dir" / "01_ids"
    predictions_dir = output_dir / "predictions"
    
    # Load ground truth
    if verbose:
        print("=" * 70)
        print("MITOSPOTTER HMM BENCHMARKING PIPELINE")
        print("=" * 70)
        print(f"\nProject root: {project_root}")
        print(f"Output dir:   {output_dir}")
        print(f"Workers:      {n_workers}")
        print("\nLoading ground truth IDs...")
    
    mito_ids, nuclear_ids = load_ground_truth_ids(ids_dir)
    
    if verbose:
        print(f"  Mitochondrial IDs: {len(mito_ids)}")
        print(f"  Nuclear IDs:       {len(nuclear_ids)}")
    
    results: List[BenchmarkResult] = []
    total_variants = len(UNIT_SIZES) * len(TRAINING_REGIMES)
    variant_num = 0
    
    for unit_size in UNIT_SIZES:
        # Resolve training and holdout files for this unit size
        nuclear_train_tsv = train_dir / f"human_nuclear_{unit_size}nt_train.tsv"
        mito_train_tsv = train_dir / f"human_mito_{unit_size}nt_train.tsv"
        nuclear_holdout_tsv = holdout_dir / f"human_nuclear_{unit_size}nt_holdout.tsv"
        mito_holdout_tsv = holdout_dir / f"human_mito_{unit_size}nt_holdout.tsv"
        
        # Verify files exist
        for p in [nuclear_train_tsv, mito_train_tsv, nuclear_holdout_tsv, mito_holdout_tsv]:
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")
        
        for regime in TRAINING_REGIMES:
            variant_num += 1
            
            if verbose:
                print(f"\n[{variant_num}/{total_variants}] "
                      f"{unit_size}nt / {regime.name}")
                print("-" * 50)
            
            result = BenchmarkResult(
                unit_size=unit_size,
                regime_name=regime.name,
                train_method=regime.train_method,
                n_em_iter=regime.n_em_iter,
                n_viterbi_iter=regime.n_viterbi_iter,
            )
            
            # Step 1: Train
            if verbose:
                print(f"  Training ({regime.train_method}: "
                      f"EM={regime.n_em_iter}, Viterbi={regime.n_viterbi_iter})...")
            
            train_time, train_mem, train_ok, train_err = train_model(
                train_script=train_script,
                regime=regime,
                unit_size=unit_size,
                nuclear_tsv=nuclear_train_tsv,
                mito_tsv=mito_train_tsv,
                output_dir=output_dir,
                n_workers=n_workers,
            )
            
            result.train_time_sec = train_time
            result.train_max_mem_mb = train_mem
            
            if not train_ok:
                result.status = "train_failed"
                result.error_message = train_err
                if verbose:
                    print(f"  [FAILED] Training failed: {train_err[:100]}...")
                results.append(result)
                continue
            
            if verbose:
                print(f"    Time: {train_time:.2f}s, Memory: {train_mem:.1f} MB")
            
            # Step 2: Decode
            if verbose:
                print(f"  Decoding holdout set (Viterbi)...")
            
            model_dir = output_dir / f"{unit_size}nt" / regime.name
            
            decode_time, decode_mem, pred_path, decode_ok, decode_err = decode_holdout(
                decode_script=decode_script,
                unit_size=unit_size,
                regime=regime,
                model_dir=model_dir,
                holdout_mito_tsv=mito_holdout_tsv,
                holdout_nuclear_tsv=nuclear_holdout_tsv,
                predictions_dir=predictions_dir,
                min_len=0,  # No minimum length filter for benchmarking
            )
            
            result.decode_time_sec = decode_time
            result.decode_max_mem_mb = decode_mem
            
            if not decode_ok:
                result.status = "decode_failed"
                result.error_message = decode_err
                if verbose:
                    print(f"  [FAILED] Decoding failed: {decode_err[:100]}...")
                results.append(result)
                continue
            
            if verbose:
                print(f"    Time: {decode_time:.2f}s, Memory: {decode_mem:.1f} MB")
            
            # Step 3: Evaluate
            if verbose:
                print(f"  Evaluating predictions...")
            
            evaluate_predictions(pred_path, mito_ids, nuclear_ids, result)
            result.status = "success"
            
            if verbose:
                print(f"    Mito holdout:    {result.n_mito_holdout}")
                print(f"    Nuclear holdout: {result.n_nuclear_holdout}")
                print(f"    TP={result.TP}, TN={result.TN}, FP={result.FP}, FN={result.FN}")
                print(f"    Accuracy:    {result.accuracy:.4f}")
                print(f"    Precision_M: {result.precision_M:.4f}")
                print(f"    Recall_M:    {result.recall_M:.4f}")
                print(f"    F1_M:        {result.f1_M:.4f}")
            
            results.append(result)
    
    return results


def save_results_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to CSV."""
    fieldnames = [
        "unit_size", "regime_name", "train_method", "n_em_iter", "n_viterbi_iter",
        "train_time_sec", "train_max_mem_mb", "decode_time_sec", "decode_max_mem_mb",
        "n_mito_holdout", "n_nuclear_holdout",
        "TP", "TN", "FP", "FN",
        "accuracy", "precision_M", "recall_M", "f1_M",
        "status", "error_message"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def print_summary_table(results: List[BenchmarkResult]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"{'Unit':<6} {'Regime':<20} {'Train(s)':<10} {'Decode(s)':<10} "
          f"{'Acc':<8} {'Prec_M':<8} {'Rec_M':<8} {'F1_M':<8} {'Status':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.unit_size}nt    {r.regime_name:<20} "
              f"{r.train_time_sec:<10.2f} {r.decode_time_sec:<10.2f} "
              f"{r.accuracy:<8.4f} {r.precision_M:<8.4f} {r.recall_M:<8.4f} {r.f1_M:<8.4f} "
              f"{r.status:<10}")
    
    print("=" * 100)
    
    # Best performer by F1
    successful = [r for r in results if r.status == "success"]
    if successful:
        best = max(successful, key=lambda x: x.f1_M)
        print(f"\nBest model by F1: {best.unit_size}nt / {best.regime_name} "
              f"(F1={best.f1_M:.4f})")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmarking pipeline for HMM-based mito/nuclear classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python 06_benchmark_pipeline.py \\
        --project_root /Users/coellearth/Library/Mobile\\ Documents/com~apple~CloudDocs/mitoSpotter \\
        --train_script scripts/04_train_hmm_nt.py \\
        --decode_script scripts/05_decode_path_nt_v2.py \\
        --n_workers 4

Output:
    out_dir/04_model/
    ├── 1nt/, 2nt/, 3nt/  (model directories)
    ├── predictions/       (prediction TSVs)
    └── benchmark_results.csv
        """
    )
    
    parser.add_argument(
        "--project_root",
        type=Path,
        required=True,
        help="Root directory of mitoSpotter project"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for models and results (default: project_root/out_dir/04_model)"
    )
    parser.add_argument(
        "--train_script",
        type=Path,
        default=None,
        help="Path to training script (default: project_root/scripts/04_train_hmm_nt.py)"
    )
    parser.add_argument(
        "--decode_script",
        type=Path,
        default=None,
        help="Path to decoding script (default: project_root/scripts/05_decode_path_nt_v2.py)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers for training (default: 1)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Resolve defaults
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or project_root / "out_dir" / "04_model").resolve()
    train_script = (args.train_script or project_root / "scripts" / "04_train_hmm_nt.py").resolve()
    decode_script = (args.decode_script or project_root / "scripts" / "05_decode_path_nt_v2.py").resolve()
    
    # Validate paths
    if not project_root.exists():
        raise SystemExit(f"Project root not found: {project_root}")
    if not train_script.exists():
        raise SystemExit(f"Training script not found: {train_script}")
    if not decode_script.exists():
        raise SystemExit(f"Decoding script not found: {decode_script}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    results = run_benchmark(
        project_root=project_root,
        output_dir=output_dir,
        train_script=train_script,
        decode_script=decode_script,
        n_workers=args.n_workers,
        verbose=not args.quiet,
    )
    
    # Save results
    results_csv = output_dir / "benchmark_results.csv"
    save_results_csv(results, results_csv)
    
    if not args.quiet:
        print_summary_table(results)
        print(f"\nResults saved to: {results_csv}")
    
    # Return exit code based on any failures
    failed = [r for r in results if r.status != "success"]
    if failed:
        print(f"\n[WARNING] {len(failed)} model(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
