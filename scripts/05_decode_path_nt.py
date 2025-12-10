#!/usr/bin/env python3
"""
HMM Viterbi decoder for nuclear vs mitochondrial classification.

Accepts explicit --model_json, --vocab_json, --states_json paths
for benchmarking custom-trained models.
"""

import argparse
import json
import os
import os.path as op
import re
import sys
import time
from datetime import datetime
import numpy as np
from Bio import SeqIO

# Optional memory tracking - only used when explicitly requested
try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False

# Optional plotting - only used when explicitly requested
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Nucleotide vocabulary
BASES = ["A", "C", "G", "T"]

# Maximum number of sequences for per-sequence bar charts
MAX_SEQS_FOR_BAR_CHART = 100


# -------------------- Color Palette -------------------- #

# Professional color scheme
COLORS = {
    'nuclear': '#1F78B4',       # Blue
    'mitochondrial': '#E31A1C', # Red
    'gc': '#33A02C',            # Green
    'at': '#FF7F00',            # Orange/Gold
    'background': '#f8f9fa',    # Light gray
    'grid': '#dee2e6',          # Medium gray
    'text': '#000000',          # Black
}


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


# -------------------- Sequence Processing -------------------- #

def clean_nt(s):
    """Clean nucleotide string: uppercase, replace U->T, keep only A/C/G/T."""
    return re.sub(r"[^ACGT]", "", str(s).upper().replace("U", "T"))


def calculate_gc_content(sequence):
    """Calculate GC content of a cleaned nucleotide sequence."""
    s = clean_nt(sequence)
    if len(s) == 0:
        return 0.0
    gc_count = s.count('G') + s.count('C')
    return gc_count / len(s)


def load_model(model_json, vocab_json):
    """Load HMM parameters pi, A, B from JSON."""
    M = json.load(open(model_json))
    pi = np.array(M["startprob"], dtype=np.float64)
    A = np.array(M["transmat"], dtype=np.float64)
    B = np.array(M["emissionprob"], dtype=np.float64)
    
    # Load vocabulary mapping
    vocab_data = json.load(open(vocab_json))
    if "bases" in vocab_data:  # 1-nt model
        idx = {b: i for i, b in enumerate(vocab_data["bases"])}
    elif "dinucs" in vocab_data:  # 2-nt model
        idx = {d: i for i, d in enumerate(vocab_data["dinucs"])}
    elif "trinucs" in vocab_data:  # 3-nt model
        idx = {t: i for i, t in enumerate(vocab_data["trinucs"])}
    else:  # Fallback
        idx = vocab_data.get("index", {})

    return pi, A, B, idx


def load_states(p):
    """Load state name mapping, and resolve nuclear/mitochondrial IDs."""
    S = json.load(open(p))
    S = {int(k): v for k, v in S.items()}
    inv = {v: k for k, v in S.items()}
    nuc_id = inv.get("nuclear", 0)
    mito_id = inv.get("mitochondrial", 1)
    return S, nuc_id, mito_id


# -------------------- HMM Algorithms -------------------- #

def viterbi(pi, A, B, obs):
    """
    Viterbi decoding: find most likely state path.
    
    Returns:
        viterbi_ll: log-likelihood of best path
        path: state sequence
    """
    T = len(obs)
    N = A.shape[0]
    logA = np.log(A + 1e-300)
    logB = np.log(B + 1e-300)
    logpi = np.log(pi + 1e-300)
    
    d = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=np.int64)
    
    d[0] = logpi + logB[:, obs[0]]
    psi[0] = -1
    
    for t in range(1, T):
        for j in range(N):
            s = d[t-1] + logA[:, j]
            psi[t, j] = int(np.argmax(s))
            d[t, j] = s[psi[t, j]] + logB[j, obs[t]]
    
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(d[-1]))
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    
    return float(d[-1, path[-1]]), path


def forward_ll(pi, A, B, obs):
    """
    Forward algorithm with scaling to compute log-likelihood.
    
    Returns:
        ll: log-likelihood of the observation sequence
    """
    T = len(obs)
    N = A.shape[0]
    
    alpha = np.zeros((T, N), dtype=np.float64)
    c = np.zeros(T, dtype=np.float64)
    
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum() or 1e-300
    alpha[0] /= c[0]
    
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum() or 1e-300
        alpha[t] /= c[t]
    
    return float(np.sum(np.log(c + 1e-300)))


# -------------------- Summary Functions -------------------- #

def summarize_viterbi(path, nuc_id=0, mito_id=1, state_names=None):
    """
    Summarize Viterbi path with winner call.
    
    Returns:
        dict with nuclear_frac, mito_frac, call (winner)
    """
    if len(path) == 0:
        return {"nuclear_frac": 0.0, "mito_frac": 0.0, "call": "NA"}
    
    nuc = float(np.mean(path == nuc_id))
    mito = float(np.mean(path == mito_id))
    winner_id = nuc_id if nuc >= mito else mito_id
    
    name = state_names.get(
        winner_id,
        "nuclear" if winner_id == nuc_id else "mitochondrial"
    ) if state_names else (
        "nuclear" if winner_id == nuc_id else "mitochondrial"
    )
    
    return {
        "nuclear_frac": round(nuc, 4),
        "mito_frac": round(mito, 4),
        "call": name
    }


# -------------------- Input Iteration -------------------- #

def iter_inputs(args):
    """Iterate over all input sequences based on CLI options."""
    if args.seq:
        for i, s in enumerate(args.seq, 1):
            sid = (
                args.seq_id[i-1]
                if args.seq_id and i <= len(args.seq_id)
                else f"seq{i}"
            )
            yield sid, s
    
    if args.stdin:
        raw = sys.stdin.read()
        if raw:
            yield args.stdin_id, raw
    
    if args.fasta:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            yield rec.id, str(rec.seq)


def sequence_to_observations(sequence, ngram, vocab_idx):
    """
    Convert sequence to observation indices based on ngram size.
    
    Returns:
        obs: numpy array of observation indices
        token_count: number of tokens generated
        nt_length: original nucleotide length (after cleaning)
    """
    s = clean_nt(sequence)
    nt_length = len(s)
    
    if ngram == 1:
        tokens = list(s)
    elif ngram == 2:
        tokens = [s[i:i+2] for i in range(len(s)-1)]
    elif ngram == 3:
        L = (len(s)//3) * 3
        tokens = [s[i:i+3] for i in range(0, L, 3)]
    else:
        tokens = list(s)

    obs = [vocab_idx[t] for t in tokens if t in vocab_idx]
    return np.array(obs, dtype=np.int64), len(tokens), nt_length


# -------------------- Plotting Functions -------------------- #

def setup_plot_style():
    """Configure matplotlib for professional-looking plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.facecolor': COLORS['background'],
        'axes.edgecolor': COLORS['grid'],
        'axes.linewidth': 1.2,
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })


def create_plot_directory():
    """Create a timestamped directory for plot outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = f"Plot_res_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def plot_single_gc_pie(gc_content, seq_id, plot_dir):
    """Create a pie chart for GC content of a single sequence."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    at_content = 1 - gc_content
    sizes = [gc_content * 100, at_content * 100]
    labels = ['GC', 'AT']
    colors = [COLORS['gc'], COLORS['at']]
    explode = (0.02, 0.02)
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        colors=colors,
        explode=explode,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.6,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax.set_title(f'GC Content\n{seq_id}', fontsize=14, fontweight='bold', 
                 color=COLORS['text'], pad=20)
    
    ax.set_aspect('equal')
    
    plt.tight_layout()
    filepath = op.join(plot_dir, 'gc_content_pie.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.close()
    return filepath


def plot_single_state_pie(nuclear_frac, mito_frac, seq_id, plot_dir):
    """Create a pie chart for M/N state proportions of a single sequence."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sizes = [nuclear_frac * 100, mito_frac * 100]
    labels = ['Nuclear (N)', 'Mitochondrial (M)']
    colors = [COLORS['nuclear'], COLORS['mitochondrial']]
    explode = (0.02, 0.02)
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        colors=colors,
        explode=explode,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.6,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax.set_title(f'HMM State Proportions\n{seq_id}', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=20)
    
    ax.set_aspect('equal')
    
    plt.tight_layout()
    filepath = op.join(plot_dir, 'state_proportions_pie.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.close()
    return filepath


def plot_multi_gc_stacked_bar(results, plot_dir):
    """Create a stacked bar chart of GC content for multiple sequences (<=100)."""
    n_seqs = len(results)
    # Cap figure width: min 10, max 20 inches
    fig_width = min(20, max(10, n_seqs * 0.15))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    seq_ids = [r['id'] for r in results]
    gc_values = [r['gc_content'] * 100 for r in results]
    at_values = [(1 - r['gc_content']) * 100 for r in results]
    
    x = np.arange(n_seqs)
    bar_width = 0.8
    
    # Create stacked bars
    ax.bar(x, gc_values, bar_width, label='GC', color=COLORS['gc'],
           edgecolor='white', linewidth=0.3)
    ax.bar(x, at_values, bar_width, bottom=gc_values, label='AT',
           color=COLORS['at'], edgecolor='white', linewidth=0.3)
    
    ax.set_xlabel('Sequence', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('GC/AT Content by Sequence', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    
    # X-axis labels - show fewer for readability
    if n_seqs <= 20:
        ax.set_xticks(x)
        ax.set_xticklabels(seq_ids, rotation=45, ha='right', fontsize=8)
    elif n_seqs <= 50:
        step = max(1, n_seqs // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([seq_ids[i] for i in range(0, n_seqs, step)], 
                          rotation=45, ha='right', fontsize=8)
    else:
        # For many sequences, just show index numbers
        step = max(1, n_seqs // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(i+1) for i in range(0, n_seqs, step)], fontsize=8)
        ax.set_xlabel('Sequence Index', fontsize=12, fontweight='bold', color=COLORS['text'])
    
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, n_seqs - 0.5)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add horizontal reference line at 50%
    ax.axhline(y=50, color=COLORS['text'], linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    filepath = op.join(plot_dir, 'gc_content_stacked_bar.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.close()
    return filepath


def plot_multi_state_stacked_bar(results, plot_dir):
    """Create a stacked bar chart of M/N state proportions for multiple sequences (<=100)."""
    n_seqs = len(results)
    # Cap figure width: min 10, max 20 inches
    fig_width = min(20, max(10, n_seqs * 0.15))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    seq_ids = [r['id'] for r in results]
    nuclear_values = [r['nuclear_frac'] * 100 for r in results]
    mito_values = [r['mito_frac'] * 100 for r in results]
    
    x = np.arange(n_seqs)
    bar_width = 0.8
    
    # Create stacked bars
    ax.bar(x, nuclear_values, bar_width, label='Nuclear (N)', 
           color=COLORS['nuclear'], edgecolor='white', linewidth=0.3)
    ax.bar(x, mito_values, bar_width, bottom=nuclear_values,
           label='Mitochondrial (M)', color=COLORS['mitochondrial'],
           edgecolor='white', linewidth=0.3)
    
    ax.set_xlabel('Sequence', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('HMM State Proportions by Sequence', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    
    # X-axis labels
    if n_seqs <= 20:
        ax.set_xticks(x)
        ax.set_xticklabels(seq_ids, rotation=45, ha='right', fontsize=8)
    elif n_seqs <= 50:
        step = max(1, n_seqs // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([seq_ids[i] for i in range(0, n_seqs, step)],
                          rotation=45, ha='right', fontsize=8)
    else:
        step = max(1, n_seqs // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(i+1) for i in range(0, n_seqs, step)], fontsize=8)
        ax.set_xlabel('Sequence Index', fontsize=12, fontweight='bold', color=COLORS['text'])
    
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, n_seqs - 0.5)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add horizontal reference line at 50%
    ax.axhline(y=50, color=COLORS['text'], linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    filepath = op.join(plot_dir, 'state_proportions_stacked_bar.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.close()
    return filepath


def plot_gc_distribution(results, plot_dir):
    """Create a horizontal stacked bar chart of GC/AT content for large datasets."""
    n_seqs = len(results)
    
    # Calculate figure height based on number of sequences
    # Minimum height of 8, scale up for more sequences, cap at reasonable size
    fig_height = min(max(8, n_seqs * 0.04), 50)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    seq_ids = [r['id'] for r in results]
    gc_values = [r['gc_content'] for r in results]
    at_values = [1 - r['gc_content'] for r in results]
    
    y = np.arange(n_seqs)
    bar_height = 0.8
    
    # Create horizontal stacked bars
    ax.barh(y, gc_values, bar_height, label='GC', color=COLORS['gc'],
            edgecolor='white', linewidth=0.2)
    ax.barh(y, at_values, bar_height, left=gc_values, label='AT',
            color=COLORS['at'], edgecolor='white', linewidth=0.2)
    
    ax.set_ylabel('Sequence', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_xlabel('Proportion', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('GC/AT Content by Sequence', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    
    # Y-axis labels - show fewer for readability
    if n_seqs <= 50:
        ax.set_yticks(y)
        ax.set_yticklabels(seq_ids, fontsize=7)
    else:
        # For many sequences, show tick marks but fewer labels
        step = max(1, n_seqs // 30)
        ax.set_yticks(y[::step])
        ax.set_yticklabels([str(i+1) for i in range(0, n_seqs, step)], fontsize=7)
        ax.set_ylabel('Sequence Index', fontsize=12, fontweight='bold', color=COLORS['text'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_seqs - 0.5)
    
    # Invert y-axis so first sequence is at top
    ax.invert_yaxis()
    
    # Grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    # Legend at bottom
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Add vertical reference line at 0.5
    ax.axvline(x=0.5, color=COLORS['text'], linestyle=':', alpha=0.5, linewidth=1)
    
    # Add statistics annotation
    mean_gc = np.mean(gc_values)
    median_gc = np.median(gc_values)
    std_gc = np.std(gc_values)
    
    stats_text = f'Mean GC: {mean_gc:.1%}\nMedian GC: {median_gc:.1%}\nStd: {std_gc:.1%}\nn = {n_seqs}'
    ax.text(0.98, 0.02, stats_text,
           transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                    edgecolor=COLORS['grid']),
           family='monospace')
    
    plt.tight_layout()
    filepath = op.join(plot_dir, 'gc_content_distribution.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.close()
    return filepath


def plot_state_distribution(results, plot_dir):
    """Create a horizontal stacked bar chart of N/M state proportions for large datasets."""
    n_seqs = len(results)
    
    # Calculate figure height based on number of sequences
    fig_height = min(max(8, n_seqs * 0.04), 50)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    seq_ids = [r['id'] for r in results]
    nuclear_values = [r['nuclear_frac'] for r in results]
    mito_values = [r['mito_frac'] for r in results]
    
    y = np.arange(n_seqs)
    bar_height = 0.8
    
    # Create horizontal stacked bars
    ax.barh(y, nuclear_values, bar_height, label='Nuclear (N)', 
            color=COLORS['nuclear'], edgecolor='white', linewidth=0.2)
    ax.barh(y, mito_values, bar_height, left=nuclear_values,
            label='Mitochondrial (M)', color=COLORS['mitochondrial'],
            edgecolor='white', linewidth=0.2)
    
    ax.set_ylabel('Sequence', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_xlabel('Proportion', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('HMM State Proportions by Sequence', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    
    # Y-axis labels
    if n_seqs <= 50:
        ax.set_yticks(y)
        ax.set_yticklabels(seq_ids, fontsize=7)
    else:
        step = max(1, n_seqs // 30)
        ax.set_yticks(y[::step])
        ax.set_yticklabels([str(i+1) for i in range(0, n_seqs, step)], fontsize=7)
        ax.set_ylabel('Sequence Index', fontsize=12, fontweight='bold', color=COLORS['text'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_seqs - 0.5)
    
    # Invert y-axis so first sequence is at top
    ax.invert_yaxis()
    
    # Grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Add classification threshold line at 0.5
    ax.axvline(x=0.5, color=COLORS['text'], linestyle='-', linewidth=1.5, alpha=0.7,
               label='Classification threshold')
    
    # Add statistics annotation
    mean_nuc = np.mean(nuclear_values)
    median_nuc = np.median(nuclear_values)
    n_nuclear = sum(1 for r in results if r['call'] == 'nuclear')
    n_mito = sum(1 for r in results if r['call'] == 'mitochondrial')
    
    stats_text = f'Mean N: {mean_nuc:.1%}\nMedian N: {median_nuc:.1%}\nClassified N: {n_nuclear}\nClassified M: {n_mito}'
    ax.text(0.98, 0.02, stats_text,
           transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                    edgecolor=COLORS['grid']),
           family='monospace')
    
    plt.tight_layout()
    filepath = op.join(plot_dir, 'state_proportions_distribution.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.close()
    return filepath


def plot_classification_counts(results, plot_dir):
    """Create a bar chart showing count of sequences classified as M or N."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    calls = [r['call'] for r in results]
    nuclear_count = sum(1 for c in calls if c == 'nuclear')
    mito_count = sum(1 for c in calls if c == 'mitochondrial')
    
    categories = ['Nuclear (N)', 'Mitochondrial (M)']
    counts = [nuclear_count, mito_count]
    colors = [COLORS['nuclear'], COLORS['mitochondrial']]
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=2,
                  width=0.6)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=14, fontweight='bold', color=COLORS['text'])
    
    ax.set_xlabel('Classification', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('Classification Summary', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    # Set y-axis to start at 0
    ax.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 1)
    
    # Add percentage annotation
    total = nuclear_count + mito_count
    if total > 0:
        nuc_pct = nuclear_count / total * 100
        mito_pct = mito_count / total * 100
        ax.text(0.98, 0.95, f'N: {nuc_pct:.1f}%  |  M: {mito_pct:.1f}%',
               transform=ax.transAxes, fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                        edgecolor=COLORS['grid']))
    
    plt.tight_layout()
    filepath = op.join(plot_dir, 'classification_counts.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.close()
    return filepath


def plot_loglikelihood_distribution(results, plot_dir):
    """Create a histogram of log-likelihood values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    logliks = [r['loglik'] for r in results]
    
    # Determine number of bins
    n_bins = min(50, max(10, len(logliks) // 5))
    
    # Create histogram
    n, bins, patches = ax.hist(logliks, bins=n_bins, color=COLORS['nuclear'],
                               edgecolor='white', linewidth=0.8, alpha=0.85)
    
    # Add a gradient effect by coloring bins
    cm = plt.cm.get_cmap('Blues')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min() + 1e-10)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(0.3 + c * 0.5))
    
    ax.set_xlabel('Log-Likelihood', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('Distribution of Log-Likelihood Values', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    # Add statistics annotation
    mean_ll = np.mean(logliks)
    median_ll = np.median(logliks)
    std_ll = np.std(logliks)
    
    stats_text = f'Mean: {mean_ll:.2f}\nMedian: {median_ll:.2f}\nStd: {std_ll:.2f}\nn = {len(logliks)}'
    ax.text(0.98, 0.95, stats_text,
           transform=ax.transAxes, fontsize=10, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                    edgecolor=COLORS['grid']),
           family='monospace')
    
    # Add vertical lines for mean and median
    ax.axvline(mean_ll, color=COLORS['mitochondrial'], linestyle='--', linewidth=2,
               label=f'Mean ({mean_ll:.2f})')
    ax.axvline(median_ll, color=COLORS['gc'], linestyle=':', linewidth=2,
               label=f'Median ({median_ll:.2f})')
    
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    filepath = op.join(plot_dir, 'loglikelihood_distribution.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.close()
    return filepath


def generate_plots(results, plot_dir):
    """
    Generate all plots based on number of sequences.
    
    Args:
        results: List of dictionaries with sequence results
        plot_dir: Directory to save plots
    
    Returns:
        List of generated file paths
    """
    setup_plot_style()
    generated_files = []
    n_seqs = len(results)
    
    if n_seqs == 1:
        # Single sequence: pie charts
        r = results[0]
        
        # GC content pie chart
        filepath = plot_single_gc_pie(r['gc_content'], r['id'], plot_dir)
        generated_files.append(filepath)
        
        # State proportions pie chart
        filepath = plot_single_state_pie(r['nuclear_frac'], r['mito_frac'], r['id'], plot_dir)
        generated_files.append(filepath)
        
    elif n_seqs <= MAX_SEQS_FOR_BAR_CHART:
        # Moderate number of sequences: stacked bar charts
        
        # GC content stacked bar chart
        filepath = plot_multi_gc_stacked_bar(results, plot_dir)
        generated_files.append(filepath)
        
        # State proportions stacked bar chart
        filepath = plot_multi_state_stacked_bar(results, plot_dir)
        generated_files.append(filepath)
        
        # Classification counts
        filepath = plot_classification_counts(results, plot_dir)
        generated_files.append(filepath)
        
        # Log-likelihood distribution
        filepath = plot_loglikelihood_distribution(results, plot_dir)
        generated_files.append(filepath)
        
    else:
        # Large dataset: use distribution plots instead of per-sequence bars
        print(f"[INFO] Large dataset ({n_seqs} sequences) - using distribution plots", 
              file=sys.stderr)
        
        # GC content distribution histogram
        filepath = plot_gc_distribution(results, plot_dir)
        generated_files.append(filepath)
        
        # State proportions distribution histogram
        filepath = plot_state_distribution(results, plot_dir)
        generated_files.append(filepath)
        
        # Classification counts
        filepath = plot_classification_counts(results, plot_dir)
        generated_files.append(filepath)
        
        # Log-likelihood distribution
        filepath = plot_loglikelihood_distribution(results, plot_dir)
        generated_files.append(filepath)
    
    return generated_files


# -------------------- Main -------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=(
            "HMM Viterbi decoder for nuclear vs mitochondrial classification. "
            "Requires explicit model path options."
        )
    )
    
    # Core parameters
    ap.add_argument("--ngram", type=int, choices=[1,2,3], default=1)
    
    # Model loading - explicit paths required
    ap.add_argument("--model_json", required=True,
                    help="Path to model JSON file")
    ap.add_argument("--vocab_json", required=True,
                    help="Path to vocabulary JSON file")
    ap.add_argument("--states_json", required=True,
                    help="Path to states JSON file")
    
    # Input options
    ap.add_argument("--fasta")
    ap.add_argument("--seq", action="append")
    ap.add_argument("--seq_id", action="append")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--stdin_id", default="stdin_seq")
    
    # Sequence processing
    ap.add_argument(
        "--min_len", 
        type=int, 
        default=150, 
        help="Minimum sequence length in nucleotides (applied before tokenization)"
    )
    
    # Output
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument(
        "--emit_path",
        action="store_true",
        help="Emit decoded path per sequence."
    )
    
    # Visualization option
    ap.add_argument(
        "--plotting",
        action="store_true",
        help="Generate visualization plots. Creates a timestamped 'Plot_res_*' folder "
             "with charts showing GC content, state proportions, and classification results."
    )
    
    # Memory tracking option - only enabled when explicitly set
    ap.add_argument(
        "--track_memory",
        action="store_true",
        help="Enable memory tracking and report peak memory usage after decoding. "
             "Requires tracemalloc module (included in Python standard library)."
    )
    
    args = ap.parse_args()
    
    if not (args.fasta or args.seq or args.stdin):
        raise SystemExit("Provide input via --fasta or --seq (repeatable) or --stdin.")
    
    # Verify model files exist
    for p in (args.model_json, args.vocab_json, args.states_json):
        if not op.exists(p):
            raise SystemExit(f"File not found: {p}")
    
    # Check matplotlib availability if plotting requested
    if args.plotting and not MATPLOTLIB_AVAILABLE:
        print("[WARNING] matplotlib not available. Plotting disabled.", file=sys.stderr)
        print("[WARNING] Install with: pip install matplotlib", file=sys.stderr)
        args.plotting = False
    
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
    
    # Load model and assets
    pi, A, B, vocab_idx = load_model(args.model_json, args.vocab_json)
    state_names, nuc_id, mito_id = load_states(args.states_json)
    
    # Track processing statistics
    n_processed = 0
    n_skipped = 0
    
    # Collect results for plotting
    plot_results = []
    
    with open(args.out_tsv, "w") as fo:
        
        fo.write(f"#id\tloglik\tnuclear_frac\tmito_frac\tcall\tseq_len_nt\tn_tokens_{args.ngram}mer\n")
        
        for rid, raw in iter_inputs(args):
            obs, token_count, nt_length = sequence_to_observations(raw, args.ngram, vocab_idx)
            
            # Apply min_len filter to nucleotide length
            if nt_length < args.min_len:
                n_skipped += 1
                continue
        
            if obs.size == 0:
                n_skipped += 1
                continue
            
            n_processed += 1
            
            ll = forward_ll(pi, A, B, obs)
            _, path = viterbi(pi, A, B, obs)
            
            summ = summarize_viterbi(path, nuc_id, mito_id, state_names)
            
            fo.write(
                f"{rid}\t{ll:.3f}\t"
                f"{summ['nuclear_frac']}\t{summ['mito_frac']}\t"
                f"{summ['call']}\t{nt_length}\t{token_count}\n"
            )
            
            if args.emit_path:
                path_str = " ".join(map(str, path.tolist()))
                fo.write(f"#{rid}_PATH\t{path_str}\n")
            
            # Collect data for plotting
            if args.plotting:
                gc_content = calculate_gc_content(raw)
                plot_results.append({
                    'id': rid,
                    'loglik': ll,
                    'nuclear_frac': summ['nuclear_frac'],
                    'mito_frac': summ['mito_frac'],
                    'call': summ['call'],
                    'gc_content': gc_content
                })
    
    # -------------------- Generate plots if requested -------------------- #
    if args.plotting and plot_results:
        plot_dir = create_plot_directory()
        print(f"[INFO] Generating plots in: {plot_dir}", file=sys.stderr)
        
        generated_files = generate_plots(plot_results, plot_dir)
        
        print(f"[OK] Generated {len(generated_files)} plot(s):", file=sys.stderr)
        for f in generated_files:
            print(f"     - {f}", file=sys.stderr)
    elif args.plotting and not plot_results:
        print("[WARNING] No sequences processed, skipping plot generation.", file=sys.stderr)
    
    # -------------------- End timing -------------------- #
    end_time = time.time()
    
    # Print summary
    print(f"[OK] Viterbi decode -> {args.out_tsv}")
    print(f"[OK] Processed: {n_processed} sequences, Skipped: {n_skipped} sequences")
    
    # -------------------- Report memory usage if tracking was enabled -------------------- #
    if args.track_memory:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report_memory(peak)
    
    # -------------------- Always report timing -------------------- #
    report_timing(start_time, end_time)


if __name__ == "__main__":
    main()