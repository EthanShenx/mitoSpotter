"""
Training runner module for mitoSpotter web UI.
Provides a wrapper around the HMM training functionality with progress callbacks
for real-time progress reporting via Server-Sent Events.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -------------------- Vocabulary Setup -------------------- #

VOCAB = ["A", "C", "G", "T"]


def setup_vocab(ngram: int):
    """Setup vocabulary based on ngram size"""
    if ngram == 1:
        IDX = {b: i for i, b in enumerate(VOCAB)}
        return VOCAB, IDX, "base_order"
    elif ngram == 2:
        VOCAB_NG = [a + b for a in VOCAB for b in VOCAB]
        IDX = {d: i for i, d in enumerate(VOCAB_NG)}
        return VOCAB_NG, IDX, "dinuc_order"
    elif ngram == 3:
        VOCAB_NG = [a + b + c for a in VOCAB for b in VOCAB for c in VOCAB]
        IDX = {t: i for i, t in enumerate(VOCAB_NG)}
        return VOCAB_NG, IDX, "trinuc_order"
    else:
        raise ValueError(f"Unsupported ngram size: {ngram}")


def read_nt_tsv(filepath: Path, IDX: dict) -> List[np.ndarray]:
    """Read sequences from a TSV file."""
    seqs = []
    with open(filepath) as fh:
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


def freq_from_seqs(seqs, VOCAB_USED, alpha=1.0):
    """Laplace-smoothed mono-nucleotide frequencies over VOCAB."""
    cnt = np.full(len(VOCAB_USED), alpha, dtype=np.float64)
    for s in seqs:
        for x in s:
            cnt[x] += 1.0
    return cnt / cnt.sum()


# -------------------- Forward-backward (soft EM) -------------------- #

def forward_backward_scaled(pi, A, B, obs):
    """Scaled forward-backward for numerical stability."""
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


# -------------------- Viterbi decode -------------------- #

def viterbi_decode(pi, A, B, obs):
    """Viterbi decoding: find the most likely hidden state path."""
    T = len(obs)
    N = A.shape[0]

    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_B = np.log(B + 1e-300)

    dp = np.full((T, N), -np.inf)
    ptr = np.zeros((T, N), dtype=np.int64)

    dp[0] = log_pi + log_B[:, obs[0]]

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


# -------------------- Training with Progress Callbacks -------------------- #

def baum_welch_with_progress(
    seqs,
    pi,
    A,
    B,
    n_iter=20,
    learn="t",
    emis_alpha=1.0,
    tol=1e-4,
    progress_callback: Optional[Callable[[dict], None]] = None,
):
    """Baum-Welch EM with progress callback support."""
    N, M = B.shape
    prev_ll = None

    for it in range(n_iter):
        pi_num = np.zeros(N)
        A_num = np.zeros((N, N))
        A_den = np.zeros(N)
        B_num = np.full((N, M), emis_alpha)
        B_den = np.full(N, emis_alpha * M)
        tot_ll = 0.0

        for seq_idx, obs in enumerate(seqs):
            gamma, xi, ll = forward_backward_scaled(pi, A, B, obs)
            tot_ll += ll

            pi_num += gamma[0]
            A_num += xi.sum(0)
            A_den += gamma[:-1].sum(0)

            for t, o in enumerate(obs):
                B_num[:, o] += gamma[t]
                B_den += gamma[t]

            # Report sequence progress within iteration
            if progress_callback and seq_idx % max(1, len(seqs) // 20) == 0:
                progress_callback({
                    "type": "sequence_progress",
                    "iteration": it + 1,
                    "total_iterations": n_iter,
                    "sequence": seq_idx + 1,
                    "total_sequences": len(seqs),
                    "phase": "em"
                })

        # normalize
        pi = pi_num / (pi_num.sum() or 1e-300)
        if learn in ("t", "et"):
            A = A_num / (A_den[:, None] + 1e-300)
            A = (A.T / (A.sum(1) + 1e-300)).T
        if learn == "et":
            B = B_num / (B_den[:, None] + 1e-300)
            B = (B.T / (B.sum(1) + 1e-300)).T

        # Report iteration progress
        if progress_callback:
            progress_callback({
                "type": "iteration_complete",
                "iteration": it + 1,
                "total_iterations": n_iter,
                "loglik": float(tot_ll),
                "phase": "em"
            })

        # early stopping
        if prev_ll is not None and abs(tot_ll - prev_ll) < tol * (1 + abs(prev_ll)):
            if progress_callback:
                progress_callback({
                    "type": "early_stop",
                    "iteration": it + 1,
                    "reason": "convergence",
                    "phase": "em"
                })
            break
        prev_ll = tot_ll

    return pi, A, B


def viterbi_train_with_progress(
    seqs,
    pi,
    A,
    B,
    n_iter=20,
    emis_alpha=1.0,
    trans_alpha=1.0,
    tol=1e-4,
    progress_callback: Optional[Callable[[dict], None]] = None,
):
    """Viterbi training with progress callback support."""
    N, M = B.shape
    prev_ll = None

    for it in range(n_iter):
        pi_cnt = np.full(N, 1e-12)
        A_cnt = np.full((N, N), trans_alpha)
        B_cnt = np.full((N, M), emis_alpha)
        tot_ll = 0.0

        for seq_idx, obs in enumerate(seqs):
            path, ll = viterbi_decode(pi, A, B, obs)
            tot_ll += ll

            pi_cnt[path[0]] += 1.0

            for t in range(len(path) - 1):
                A_cnt[path[t], path[t + 1]] += 1.0

            for t, o in enumerate(obs):
                B_cnt[path[t], o] += 1.0

            # Report sequence progress within iteration
            if progress_callback and seq_idx % max(1, len(seqs) // 20) == 0:
                progress_callback({
                    "type": "sequence_progress",
                    "iteration": it + 1,
                    "total_iterations": n_iter,
                    "sequence": seq_idx + 1,
                    "total_sequences": len(seqs),
                    "phase": "viterbi"
                })

        # normalize
        pi = pi_cnt / (pi_cnt.sum() + 1e-300)
        A = A_cnt / (A_cnt.sum(axis=1, keepdims=True) + 1e-300)
        B = B_cnt / (B_cnt.sum(axis=1, keepdims=True) + 1e-300)

        # Report iteration progress
        if progress_callback:
            progress_callback({
                "type": "iteration_complete",
                "iteration": it + 1,
                "total_iterations": n_iter,
                "loglik": float(tot_ll),
                "phase": "viterbi"
            })

        # early stopping
        if prev_ll is not None and abs(tot_ll - prev_ll) < tol * (1 + abs(prev_ll)):
            if progress_callback:
                progress_callback({
                    "type": "early_stop",
                    "iteration": it + 1,
                    "reason": "convergence",
                    "phase": "viterbi"
                })
            break
        prev_ll = tot_ll

    return pi, A, B


# -------------------- Training Configuration -------------------- #

@dataclass
class TrainingConfig:
    """Configuration for HMM training."""
    nuclear_tsv: Path
    mito_tsv: Path
    output_dir: Path
    ngram: int = 1
    train_method: str = "em"  # em, viterbi, hybrid
    learn: str = "et"  # t, et, none
    n_em_iter: int = 20
    n_viterbi_iter: int = 20
    self_loop: float = 0.995
    emis_smooth: float = 1.0
    trans_smooth: float = 1.0
    sample: float = 1.0  # Fraction of nuclear sequences to use


@dataclass
class TrainingJob:
    """Represents a training job with progress tracking."""
    job_id: str
    config: TrainingConfig
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    current_phase: str = ""
    current_iteration: int = 0
    total_iterations: int = 0
    loglik: float = 0.0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "current_phase": self.current_phase,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "loglik": self.loglik,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# -------------------- Training Runner -------------------- #

class TrainingRunner:
    """Manages HMM training jobs with progress tracking."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def create_job(self, config: TrainingConfig) -> TrainingJob:
        """Create a new training job."""
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(job_id=job_id, config=config)

        with self._lock:
            self.jobs[job_id] = job

        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        with self._lock:
            return self.jobs.get(job_id)

    def run_training(
        self,
        job: TrainingJob,
        progress_callback: Optional[Callable[[dict], None]] = None
    ) -> Dict[str, Any]:
        """Run the training job."""
        config = job.config

        try:
            job.status = "running"
            job.started_at = datetime.now().isoformat()

            # Calculate total iterations for progress
            total_iters = 0
            if config.train_method in ("em", "hybrid"):
                total_iters += config.n_em_iter
            if config.train_method in ("viterbi", "hybrid"):
                total_iters += config.n_viterbi_iter
            job.total_iterations = total_iters

            # Notify start
            if progress_callback:
                progress_callback({
                    "type": "start",
                    "job_id": job.job_id,
                    "total_iterations": total_iters,
                    "train_method": config.train_method
                })

            # Setup vocabulary
            VOCAB_USED, IDX, order_key = setup_vocab(config.ngram)

            # Load data
            job.current_phase = "loading"
            if progress_callback:
                progress_callback({"type": "phase", "phase": "loading", "message": "Loading sequences..."})

            nuc = read_nt_tsv(config.nuclear_tsv, IDX)

            # Sample nuclear sequences if requested
            if config.sample < 1.0:
                n_original = len(nuc)
                n_sample = max(1, int(n_original * config.sample))
                np.random.seed(42)
                indices = np.random.choice(n_original, size=n_sample, replace=False)
                nuc = [nuc[i] for i in indices]
                if progress_callback:
                    progress_callback({
                        "type": "info",
                        "message": f"Sampled {n_sample}/{n_original} nuclear sequences ({config.sample*100:.1f}%)"
                    })

            mit = read_nt_tsv(config.mito_tsv, IDX)
            allseq = nuc + mit

            if not allseq:
                raise ValueError("No sequences to train.")

            if progress_callback:
                progress_callback({
                    "type": "info",
                    "message": f"Loaded {len(nuc)} nuclear + {len(mit)} mito sequences"
                })

            # Initialize emissions from class frequencies
            nuc_emit = freq_from_seqs(nuc, VOCAB_USED, config.emis_smooth)
            mit_emit = freq_from_seqs(mit, VOCAB_USED, config.emis_smooth)
            B = np.vstack([nuc_emit, mit_emit]).astype(np.float64)

            # Initialize transitions and start distribution
            s = config.self_loop
            A = np.array([[s, 1.0 - s], [1.0 - s, s]], dtype=np.float64)
            pi = np.array([0.5, 0.5], dtype=np.float64)

            # Track iterations for hybrid
            completed_iters = 0

            def update_progress(info: dict):
                """Internal progress updater."""
                nonlocal completed_iters

                if info.get("type") == "iteration_complete":
                    iter_num = info["iteration"]
                    phase = info.get("phase", "")

                    if config.train_method == "hybrid" and phase == "viterbi":
                        completed_iters = config.n_em_iter + iter_num
                    else:
                        completed_iters = iter_num

                    job.current_iteration = completed_iters
                    job.progress = (completed_iters / total_iters) * 100 if total_iters > 0 else 0
                    job.loglik = info.get("loglik", 0)
                    job.current_phase = phase

                if progress_callback:
                    info["job_id"] = job.job_id
                    info["overall_progress"] = job.progress
                    progress_callback(info)

            # Training
            if config.train_method == "em":
                job.current_phase = "em"
                if progress_callback:
                    progress_callback({"type": "phase", "phase": "em", "message": "Starting EM training..."})

                pi, A, B = baum_welch_with_progress(
                    allseq, pi, A, B,
                    n_iter=config.n_em_iter,
                    learn=config.learn,
                    emis_alpha=config.emis_smooth,
                    progress_callback=update_progress
                )

            elif config.train_method == "viterbi":
                job.current_phase = "viterbi"
                if progress_callback:
                    progress_callback({"type": "phase", "phase": "viterbi", "message": "Starting Viterbi training..."})

                pi, A, B = viterbi_train_with_progress(
                    allseq, pi, A, B,
                    n_iter=config.n_viterbi_iter,
                    emis_alpha=config.emis_smooth,
                    trans_alpha=config.trans_smooth,
                    progress_callback=update_progress
                )

            elif config.train_method == "hybrid":
                # EM phase
                job.current_phase = "em"
                if progress_callback:
                    progress_callback({"type": "phase", "phase": "em", "message": "Starting EM phase..."})

                pi, A, B = baum_welch_with_progress(
                    allseq, pi, A, B,
                    n_iter=config.n_em_iter,
                    learn=config.learn,
                    emis_alpha=config.emis_smooth,
                    progress_callback=update_progress
                )

                # Viterbi phase
                job.current_phase = "viterbi"
                if progress_callback:
                    progress_callback({"type": "phase", "phase": "viterbi", "message": "Starting Viterbi fine-tuning..."})

                pi, A, B = viterbi_train_with_progress(
                    allseq, pi, A, B,
                    n_iter=config.n_viterbi_iter,
                    emis_alpha=config.emis_smooth,
                    trans_alpha=config.trans_smooth,
                    progress_callback=update_progress
                )

            # Save model
            job.current_phase = "saving"
            if progress_callback:
                progress_callback({"type": "phase", "phase": "saving", "message": "Saving model..."})

            output_dir = config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate output filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{config.ngram}nt_{config.train_method}_{timestamp}"

            model_path = output_dir / f"{model_name}.json"
            vocab_path = output_dir / f"{model_name}_vocab.json"
            states_path = output_dir / f"{model_name}_states.json"

            # Save model JSON
            model = {
                "n_states": 2,
                "n_observations": len(VOCAB_USED),
                "startprob": pi.tolist(),
                "transmat": A.tolist(),
                "emissionprob": B.tolist(),
                order_key: VOCAB_USED,
                "ngram": config.ngram,
                "training": {
                    "train_method": config.train_method,
                    "learn": config.learn,
                    "n_em_iter": config.n_em_iter,
                    "n_viterbi_iter": config.n_viterbi_iter,
                    "self_loop": config.self_loop,
                    "emis_smooth": config.emis_smooth,
                    "trans_smooth": config.trans_smooth,
                    "sample": config.sample,
                },
            }
            with open(model_path, "w") as f:
                json.dump(model, f, indent=2)

            # Save vocab JSON
            if config.ngram == 1:
                vocab_data = {"bases": VOCAB_USED, "index": IDX}
            elif config.ngram == 2:
                vocab_data = {"dinucs": VOCAB_USED, "index": IDX}
            else:
                vocab_data = {"trinucs": VOCAB_USED, "index": IDX}
            with open(vocab_path, "w") as f:
                json.dump(vocab_data, f, indent=2)

            # Save states JSON
            with open(states_path, "w") as f:
                json.dump({"0": "nuclear", "1": "mitochondrial"}, f, indent=2)

            # Mark complete
            job.status = "completed"
            job.progress = 100.0
            job.completed_at = datetime.now().isoformat()
            job.result = {
                "model_path": str(model_path),
                "vocab_path": str(vocab_path),
                "states_path": str(states_path),
                "n_sequences": len(allseq),
                "n_nuclear": len(nuc),
                "n_mito": len(mit),
            }

            if progress_callback:
                progress_callback({
                    "type": "complete",
                    "job_id": job.job_id,
                    "result": job.result
                })

            return job.result

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now().isoformat()

            if progress_callback:
                progress_callback({
                    "type": "error",
                    "job_id": job.job_id,
                    "error": str(e)
                })

            raise
