#!/usr/bin/env python3
# Unified training script for a 2-state Multinomial HMM on 1-nt sequences (A/C/G/T).
# Supports three training methods:
#   - EM       : Baum-Welch (forward-backward, soft EM)
#   - Viterbi  : Viterbi training (hard EM)
#   - hybrid   : EM first, then Viterbi fine-tuning

# Example usage:
# python merged_hmm_train.py \
#     --nuclear_tsv nuclear_1nt.tsv \
#     --mito_tsv mito_1nt.tsv \
#     --ngram 1 \
#     --train_method em \
#     --out_model_json model_1nt.json

# -------------------- Loading Animation -------------------- #
import sys
import threading
import time as _time

class LoadingSpinner:
    """A simple loading spinner animation using only standard library."""
    
    def __init__(self, message="Training initiating"):
        self.message = message
        self.spinning = False
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.ascii_frames = ["|", "/", "-", "\\"]
    
    def _spin(self):
        """Internal spinning loop."""
        idx = 0
        try:
            sys.stderr.write(self.frames[0])
            sys.stderr.flush()
            frames = self.frames
        except (UnicodeEncodeError, UnicodeError):
            frames = self.ascii_frames
        
        while self.spinning:
            frame = frames[idx % len(frames)]
            sys.stderr.write(f"\r{self.message}... {frame} ")
            sys.stderr.flush()
            _time.sleep(0.1)
            idx += 1
    
    def start(self):
        """Start the spinner."""
        self.spinning = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
    
    def stop(self, final_message=None):
        """Stop the spinner and optionally print a final message."""
        self.spinning = False
        if self.thread:
            self.thread.join(timeout=0.5)
        sys.stderr.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stderr.flush()
        if final_message:
            print(final_message, file=sys.stderr)

# Only show banner and spinner in the main process
if __name__ == "__main__":
    print("""
┏┏ ┛━┏┛┏━┃┏━┛┏━┃┏━┃━┏┛━┏┛┏━┛┏━┃
┃┃┃┃ ┃ ┃ ┃━━┃┏━┛┃ ┃ ┃  ┃ ┏━┛┏┏┛
┛┛┛┛ ┛ ━━┛━━┛┛  ━━┛ ┛  ┛ ━━┛┛ ┛
""")
    spinner = LoadingSpinner("Training initiating")
    spinner.start()

# -------------------- Import Phase -------------------- #
import argparse
import json
import os
import time
from datetime import datetime
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

# Stop spinner only in main process
if __name__ == "__main__":
    spinner.stop("Training initiating... done!")

NUC, MITO = 0, 1
VOCAB = ["A", "C", "G", "T"]

# -------------------- Timing and Memory Utilities -------------------- #

def format_timestamp(ts):
    """Format a timestamp as a human-readable string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds):
    """Format duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} min {secs:.2f} sec"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)} hr {int(minutes)} min {secs:.2f} sec"


def format_memory(bytes_value):
    """Format memory size in bytes to a human-readable string."""
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.2f} KB"
    elif bytes_value < 1024 ** 3:
        return f"{bytes_value / (1024 ** 2):.2f} MB"
    else:
        return f"{bytes_value / (1024 ** 3):.2f} GB"


def report_timing(start_time, end_time):
    """Print timing report to stderr."""
    duration = end_time - start_time
    print("\n" + "=" * 60, file=sys.stderr)
    print("TIMING REPORT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Start time : {format_timestamp(start_time)}", file=sys.stderr)
    print(f"  End time   : {format_timestamp(end_time)}", file=sys.stderr)
    print(f"  Total time : {format_duration(duration)}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def report_memory(peak_memory):
    """Print memory report to stderr."""
    print("\n" + "-" * 60, file=sys.stderr)
    print("MEMORY REPORT", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print(f"  Peak memory usage : {format_memory(peak_memory)}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)


# -------------------- Data loading & helpers -------------------- #

def setup_vocab(ngram):
    """Setup vocabulary based on ngram size"""
    if ngram == 1:
        IDX = {b:i for i,b in enumerate(VOCAB)}
        return VOCAB, IDX, "base_order"
    elif ngram == 2:
        VOCAB_NG = [a+b for a in VOCAB for b in VOCAB]  # AA, AC, ..., TT
        IDX = {d:i for i,d in enumerate(VOCAB_NG)}
        return VOCAB_NG, IDX, "dinuc_order"
    elif ngram == 3:
        VOCAB_NG = [a+b+c for a in VOCAB for b in VOCAB for c in VOCAB]  # AAA, AAC, ..., TTT
        IDX = {t:i for i,t in enumerate(VOCAB_NG)}
        return VOCAB_NG, IDX, "trinuc_order"
    else:
        raise ValueError(f"Unsupported ngram size: {ngram}")


def read_nt_tsv(p, IDX):
    """
    Read sequences from a TSV file.
    Each line format: id \t A C G T ...
    Returns a list of np.ndarray(int64) with values in [0, M-1].
    """
    seqs = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            xs = [IDX[c] for c in parts[1].split() if c in IDX]
            if xs:
                seqs.append(np.asarray(xs, dtype=np.int64))
    return seqs


def freq_from_seqs(seqs, VOCAB, alpha=1.0):
    """
    Laplace-smoothed mono-nucleotide frequencies over VOCAB.
    """
    cnt = np.full(len(VOCAB), alpha, dtype=np.float64)
    for s in seqs:
        for x in s:
            cnt[x] += 1.0
    return cnt / cnt.sum()


# -------------------- Forward-backward (soft EM) -------------------- #

def forward_backward_scaled(pi, A, B, obs):
    """
    Scaled forward-backward for numerical stability.

    pi:  (N,)
    A:   (N, N)
    B:   (N, M)
    obs: (T,) int64

    Returns:
      gamma: (T, N)
      xi:    (T-1, N, N)
      ll:    float, log-likelihood log p(obs)
    """
    T = len(obs)
    N = A.shape[0]
    alpha = np.zeros((T, N))
    beta = np.zeros((T, N))
    c = np.zeros(T)

    # forward
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum() or 1e-300
    alpha[0] /= c[0]

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum() or 1e-300
        alpha[t] /= c[t]

    # backward
    beta[-1] = 1.0 / c[-1]
    for t in range(T - 2, -1, -1):
        beta[t] = (A * B[:, obs[t + 1]]).dot(beta[t + 1])
        beta[t] /= c[t]

    # gamma
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

    # xi
    xi = np.zeros((T - 1, N, N))
    for t in range(T - 1):
        denom = (alpha[t] @ A * B[:, obs[t + 1]]).dot(beta[t + 1]) or 1e-300
        xi[t] = (alpha[t][:, None] * A) * (
            B[:, obs[t + 1]] * beta[t + 1]
        )[None, :] / denom

    ll = np.sum(np.log(c + 1e-300))
    return gamma, xi, ll


# -------------------- Parallel Processing Helpers -------------------- #

def _process_sequence_em(args):
    """
    Worker function for parallel EM processing.
    Processes a single sequence and returns accumulated statistics.
    """
    obs, pi, A, B = args
    N, M = B.shape
    
    gamma, xi, ll = forward_backward_scaled(pi, A, B, obs)
    
    # Accumulate statistics for this sequence
    pi_num = gamma[0].copy()
    A_num = xi.sum(0)
    A_den = gamma[:-1].sum(0)
    
    B_num = np.zeros((N, M))
    B_den = np.zeros(N)
    for t, o in enumerate(obs):
        B_num[:, o] += gamma[t]
        B_den += gamma[t]
    
    return pi_num, A_num, A_den, B_num, B_den, ll


def _process_sequence_viterbi(args):
    """
    Worker function for parallel Viterbi processing.
    Processes a single sequence and returns accumulated counts.
    """
    obs, pi, A, B = args
    N, M = B.shape
    
    path, ll = viterbi_decode(pi, A, B, obs)
    
    # Accumulate counts for this sequence
    pi_cnt = np.zeros(N)
    pi_cnt[path[0]] = 1.0
    
    A_cnt = np.zeros((N, N))
    for t in range(len(path) - 1):
        A_cnt[path[t], path[t + 1]] += 1.0
    
    B_cnt = np.zeros((N, M))
    for t, o in enumerate(obs):
        B_cnt[path[t], o] += 1.0
    
    return pi_cnt, A_cnt, B_cnt, ll


def baum_welch(
    seqs,
    pi,
    A,
    B,
    n_iter=20,
    learn="t",
    emis_alpha=1.0,
    show_progress=True,
    tol=1e-4,
    n_workers=1,
):
    """
    Standard Baum-Welch EM (soft EM) with optional parallel processing.

    seqs: list of np.ndarray(int64)
    pi, A, B: initial parameters
    n_iter: maximum number of EM iterations
    learn:
      "t"   -> learn transitions only
      "et"  -> learn transitions + emissions
      "none" (not used here) -> skip EM
    emis_alpha: Laplace smoothing for emission probabilities
    tol: relative tolerance for early stopping on log-likelihood
    n_workers: number of parallel workers (1 = sequential)
    """
    N, M = B.shape
    outer = tqdm(
        range(n_iter),
        desc="EM",
        unit="iter",
        disable=not sys.stderr.isatty() or not show_progress,
    )
    
    use_parallel = n_workers > 1 and len(seqs) > n_workers

    prev_ll = None
    for it in outer:
        pi_num = np.zeros(N)
        A_num = np.zeros((N, N))
        A_den = np.zeros(N)
        B_num = np.full((N, M), emis_alpha)
        B_den = np.full(N, emis_alpha * M)
        tot_ll = 0.0

        if use_parallel:
            # Parallel processing
            args_list = [(obs, pi, A, B) for obs in seqs]
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_sequence_em, args): i 
                          for i, args in enumerate(args_list)}
                
                inner = tqdm(
                    as_completed(futures),
                    total=len(seqs),
                    desc=f"iter {it+1}/{n_iter}",
                    unit="seq",
                    leave=False,
                    disable=not sys.stderr.isatty() or not show_progress,
                )
                
                for future in inner:
                    p_pi, p_A, p_A_den, p_B, p_B_den, ll = future.result()
                    pi_num += p_pi
                    A_num += p_A
                    A_den += p_A_den
                    B_num += p_B
                    B_den += p_B_den
                    tot_ll += ll
        else:
            # Sequential processing (original behavior)
            inner = tqdm(
                seqs,
                desc=f"iter {it+1}/{n_iter}",
                unit="seq",
                leave=False,
                disable=not sys.stderr.isatty() or not show_progress,
            )

            for obs in inner:
                gamma, xi, ll = forward_backward_scaled(pi, A, B, obs)
                tot_ll += ll

                pi_num += gamma[0]
                A_num += xi.sum(0)
                A_den += gamma[:-1].sum(0)

                for t, o in enumerate(obs):
                    B_num[:, o] += gamma[t]
                    B_den += gamma[t]

        # normalize
        pi = pi_num / (pi_num.sum() or 1e-300)
        if learn in ("t", "et"):
            A = A_num / (A_den[:, None] + 1e-300)
            A = (A.T / (A.sum(1) + 1e-300)).T
        if learn == "et":
            B = B_num / (B_den[:, None] + 1e-300)
            B = (B.T / (B.sum(1) + 1e-300)).T

        if hasattr(outer, "set_postfix"):
            outer.set_postfix({"loglik": f"{tot_ll:.1f}"})
        else:
            print(f"[EM] iter={it+1} loglik={tot_ll:.3f}")

        # early stopping
        if prev_ll is not None and abs(tot_ll - prev_ll) < tol * (1 + abs(prev_ll)):
            break
        prev_ll = tot_ll

    return pi, A, B


# -------------------- Viterbi decode & hard EM -------------------- #

def viterbi_decode(pi, A, B, obs):
    """
    Viterbi decoding: find the most likely hidden state path.

    pi:  (N,)
    A:   (N, N)
    B:   (N, M)
    obs: (T,) int64

    Returns:
      path:   (T,) int64
      loglik: float, log p(x, z*)
    """
    T = len(obs)
    N = A.shape[0]

    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_B = np.log(B + 1e-300)

    dp = np.full((T, N), -np.inf)
    ptr = np.zeros((T, N), dtype=np.int64)

    # t = 0
    o0 = obs[0]
    dp[0] = log_pi + log_B[:, o0]

    # t = 1..T-1
    for t in range(1, T):
        o = obs[t]
        for j in range(N):
            scores = dp[t - 1] + log_A[:, j]
            best_prev = np.argmax(scores)
            dp[t, j] = scores[best_prev] + log_B[j, o]
            ptr[t, j] = best_prev

    path = np.zeros(T, dtype=np.int64)
    last_state = np.argmax(dp[T - 1])
    path[T - 1] = last_state
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    loglik = dp[T - 1, last_state]
    return path, loglik


def viterbi_train(
    seqs,
    pi,
    A,
    B,
    n_iter=20,
    emis_alpha=1.0,
    trans_alpha=1.0,
    show_progress=True,
    tol=1e-4,
    n_workers=1,
):
    """
    Viterbi training (hard EM) with optional parallel processing.

    seqs: list of np.ndarray(int64)
    pi, A, B: initial parameters
    emis_alpha: Laplace smoothing for emission counts
    trans_alpha: Laplace smoothing for transition counts
    tol: relative tolerance for early stopping on joint log-likelihood
    n_workers: number of parallel workers (1 = sequential)
    """
    N, M = B.shape
    outer = tqdm(
        range(n_iter),
        desc="Viterbi-EM",
        unit="iter",
        disable=not sys.stderr.isatty() or not show_progress,
    )
    
    use_parallel = n_workers > 1 and len(seqs) > n_workers

    prev_ll = None
    for it in outer:
        pi_cnt = np.full(N, 1e-12)  # tiny prior for initial distribution
        A_cnt = np.full((N, N), trans_alpha)
        B_cnt = np.full((N, M), emis_alpha)

        tot_ll = 0.0

        if use_parallel:
            # Parallel processing
            args_list = [(obs, pi, A, B) for obs in seqs]
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_sequence_viterbi, args): i 
                          for i, args in enumerate(args_list)}
                
                inner = tqdm(
                    as_completed(futures),
                    total=len(seqs),
                    desc=f"iter {it+1}/{n_iter}",
                    unit="seq",
                    leave=False,
                    disable=not sys.stderr.isatty() or not show_progress,
                )
                
                for future in inner:
                    p_pi, p_A, p_B, ll = future.result()
                    pi_cnt += p_pi
                    A_cnt += p_A
                    B_cnt += p_B
                    tot_ll += ll
        else:
            # Sequential processing (original behavior)
            inner = tqdm(
                seqs,
                desc=f"iter {it+1}/{n_iter}",
                unit="seq",
                leave=False,
                disable=not sys.stderr.isatty() or not show_progress,
            )

            for obs in inner:
                path, ll = viterbi_decode(pi, A, B, obs)
                tot_ll += ll

                # initial state
                pi_cnt[path[0]] += 1.0

                # transitions
                for t in range(len(path) - 1):
                    A_cnt[path[t], path[t + 1]] += 1.0

                # emissions
                for t, o in enumerate(obs):
                    B_cnt[path[t], o] += 1.0

        # normalize
        pi = pi_cnt / (pi_cnt.sum() + 1e-300)
        A = A_cnt / (A_cnt.sum(axis=1, keepdims=True) + 1e-300)
        B = B_cnt / (B_cnt.sum(axis=1, keepdims=True) + 1e-300)

        if hasattr(outer, "set_postfix"):
            outer.set_postfix({"joint_ll": f"{tot_ll:.1f}"})
        else:
            print(f"[Viterbi] iter={it+1} joint_loglik={tot_ll:.3f}")

        # early stopping on joint likelihood
        if prev_ll is not None and abs(tot_ll - prev_ll) < tol * (1 + abs(prev_ll)):
            break
        prev_ll = tot_ll

    return pi, A, B


# -------------------- Main: unified training interface -------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Train a 2-state 1-nt/2-nt/3-nt HMM (A/C/G/T) from nuclear/mito CDS using EM, Viterbi, or hybrid."
    )
    parser.add_argument("--nuclear_nt_tsv", required=True)
    parser.add_argument("--mito_nt_tsv", required=True)
    parser.add_argument("--out_model_json", required=True)
    parser.add_argument("--out_vocab_json", required=True)
    parser.add_argument("--out_states_json", required=True)
    parser.add_argument("--ngram", type=int, choices=[1,2,3], default=1)

    # Training method
    parser.add_argument("--train_method", choices=["em", "viterbi", "hybrid"], default="em")
    # EM-related options
    parser.add_argument("--learn", choices=["t", "et", "none"], default="t")
    parser.add_argument("--n_em_iter", type=int, default=20)

    # Viterbi-related options
    parser.add_argument("--n_viterbi_iter", type=int, default=20)
    parser.add_argument("--trans_smooth", type=float, default=1.0)

    # General HMM hyper-parameters
    parser.add_argument("--self_loop", type=float, default=0.995)
    parser.add_argument("--emis_smooth", type=float, default=1.0)

    # Sampling option for nuclear sequences
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fraction of nuclear sequences to use for training (0.0-1.0). "
             "E.g., 0.3 uses 30%% of nuclear sequences. Default is 1.0 (use all)."
    )

    # Memory tracking option - only enabled when explicitly set
    parser.add_argument(
        "--track_memory",
        action="store_true",
        help="Enable memory tracking and report peak memory usage after training. "
             "Requires tracemalloc module (included in Python standard library)."
    )
    
    # Parallel processing option (similar to R's future package)
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers for training. "
             "Set to number of CPU cores for maximum parallelism. "
             "Use 1 for sequential processing (default). "
             "Similar to R's future::plan(multisession, workers=N)."
    )

    args = parser.parse_args()
    
    # Determine number of workers
    n_workers = args.n_workers
    max_workers = multiprocessing.cpu_count()
    
    if n_workers < 1:
        n_workers = 1
    elif n_workers > max_workers:
        print(f"[WARNING] Requested {n_workers} workers but only {max_workers} CPUs available. "
              f"Using {max_workers} workers.", file=sys.stderr)
        n_workers = max_workers
    
    if n_workers > 1:
        print(f"[INFO] Parallel processing enabled with {n_workers} workers "
              f"(max available: {max_workers}).", file=sys.stderr)

    # -------------------- Start timing -------------------- #
    start_time = time.time()

    # -------------------- Start memory tracking if requested -------------------- #
    if args.track_memory:
        if TRACEMALLOC_AVAILABLE:
            tracemalloc.start()
            print("[INFO] Memory tracking enabled.", file=sys.stderr)
        else:
            print("[WARNING] tracemalloc not available. Memory tracking disabled.", file=sys.stderr)
            args.track_memory = False

    # Load data
    VOCAB_USED, IDX, order_key = setup_vocab(args.ngram)
    nuc = read_nt_tsv(args.nuclear_nt_tsv, IDX)
    
    # Sample nuclear sequences if requested
    if args.sample < 1.0:
        if args.sample <= 0.0:
            raise SystemExit("--sample must be > 0.0")
        n_original = len(nuc)
        n_sample = max(1, int(n_original * args.sample))
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(n_original, size=n_sample, replace=False)
        nuc = [nuc[i] for i in indices]
        print(f"[INFO] Sampled {n_sample} / {n_original} nuclear sequences "
              f"({args.sample * 100:.1f}%).", file=sys.stderr)
    
    mit = read_nt_tsv(args.mito_nt_tsv, IDX)
    allseq = nuc + mit
    
    if not allseq:
        raise SystemExit("No sequences to train.")

    # Initialize emissions from class frequencies
    nuc_emit = freq_from_seqs(nuc, VOCAB_USED, args.emis_smooth)
    mit_emit = freq_from_seqs(mit, VOCAB_USED, args.emis_smooth)
    B = np.vstack([nuc_emit, mit_emit]).astype(np.float64)

    # Initialize transitions and start distribution
    s = args.self_loop
    A = np.array([[s, 1.0 - s],
                  [1.0 - s, s]], dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)

    # -------------------- Training selection -------------------- #

    train_method = args.train_method

    if train_method == "em":
        # Pure EM (soft)
        if args.learn != "none" and args.n_em_iter > 0:
            pi, A, B = baum_welch(
                allseq,
                pi,
                A,
                B,
                n_iter=args.n_em_iter,
                learn=args.learn,
                emis_alpha=args.emis_smooth,
                show_progress=True,
                tol=1e-4,
                n_workers=n_workers,
            )
        else:
            print("[INFO] EM training skipped (learn='none' or n_em_iter <= 0).")

    elif train_method == "viterbi":
        # Pure Viterbi (hard)
        if args.n_viterbi_iter > 0:
            pi, A, B = viterbi_train(
                allseq,
                pi,
                A,
                B,
                n_iter=args.n_viterbi_iter,
                emis_alpha=args.emis_smooth,
                trans_alpha=args.trans_smooth,
                show_progress=True,
                tol=1e-4,
                n_workers=n_workers,
            )
        else:
            print("[INFO] Viterbi training skipped (n_viterbi_iter <= 0).")

    elif train_method == "hybrid":
        # 1) Optional EM, then 2) Viterbi fine-tuning
        if args.learn != "none" and args.n_em_iter > 0:
            pi, A, B = baum_welch(
                allseq,
                pi,
                A,
                B,
                n_iter=args.n_em_iter,
                learn=args.learn,
                emis_alpha=args.emis_smooth,
                show_progress=True,
                tol=1e-4,
                n_workers=n_workers,
            )
        else:
            print("[INFO] EM phase skipped in hybrid (learn='none' or n_em_iter <= 0).")

        if args.n_viterbi_iter > 0:
            pi, A, B = viterbi_train(
                allseq,
                pi,
                A,
                B,
                n_iter=args.n_viterbi_iter,
                emis_alpha=args.emis_smooth,
                trans_alpha=args.trans_smooth,
                show_progress=True,
                tol=1e-4,
                n_workers=n_workers,
            )
        else:
            print("[INFO] Viterbi phase skipped in hybrid (n_viterbi_iter <= 0).")

    else:
        raise SystemExit(f"Unknown train_method: {train_method}")

    # -------------------- Save artifacts -------------------- #

    os.makedirs(os.path.dirname(args.out_model_json) or ".", exist_ok=True)
    model = {
        "n_states": 2,
        "n_observations": len(VOCAB_USED),
        "startprob": pi.tolist(),
        "transmat": A.tolist(),
        "emissionprob": B.tolist(),
        order_key: VOCAB_USED,
        "ngram": args.ngram,
        "training": {
            "train_method": args.train_method,
            "learn": args.learn,
            "n_em_iter": args.n_em_iter,
            "n_viterbi_iter": args.n_viterbi_iter,
            "self_loop": args.self_loop,
            "emis_smooth": args.emis_smooth,
            "trans_smooth": args.trans_smooth,
            "n_workers": n_workers,
            "sample": args.sample,
        },
    }
    with open(args.out_model_json, "w") as f:
        json.dump(model, f, indent=2)

    if args.ngram == 1:
        vocab_data = {"bases": VOCAB_USED, "index": IDX}
    elif args.ngram == 2:
        vocab_data = {"dinucs": VOCAB_USED, "index": IDX}
    elif args.ngram == 3:
        vocab_data = {"trinucs": VOCAB_USED, "index": IDX}
    with open(args.out_vocab_json,"w") as f: json.dump(vocab_data,f,indent=2)

    with open(args.out_states_json, "w") as f:
        json.dump({"0": "nuclear", "1": "mitochondrial"}, f, indent=2)

    print(f"[OK] model  -> {args.out_model_json}")
    print(f"[OK] vocab  -> {args.out_vocab_json}")
    print(f"[OK] states -> {args.out_states_json}")

    # -------------------- End timing -------------------- #
    end_time = time.time()

    # -------------------- Report memory usage if tracking was enabled -------------------- #
    if args.track_memory:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report_memory(peak)

    # -------------------- Always report timing -------------------- #
    report_timing(start_time, end_time)


if __name__ == "__main__":
    main()