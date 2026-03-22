'''
Convert a list of BenchmarkResult objects into charts and a summary csv

Charts generated: 
exec_time
speedup
memory
cpu
crossover - 
'''

import os
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from metrics import BenchmarkResult

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

COLORS = {
    "single":       "#4C72B0",   # blue
    "multiprocess": "#DD8452",   # orange
    "distributed":  "#55A868",   # green
}

def _color(label: str) -> str:
    '''
    Look up the color chart for a given result label
    '''
    for key in COLORS:
        if key in label:
            return COLORS[key]
    return '#888888'

def result_to_df(results: List[BenchmarkResult]) -> pd.DataFrame:
    '''
    Convert a list of BenchmarkResult dataclasses into a flat pandas DataFrame for easy filtering, grouping, and exporting
    '''
    return pd.DataFrame([
        {
            'label': r.label,
            'algorithm': r.algorithm,
            'size': r.size,
            'num_workers': r.num_workers,
            'elapsed_sec': r.elapsed_sec,
            'peak_memory_mb': r.peak_memory_mb,
            'avg_cpu_pct': r.avg_cpu_pct,
            'speedup': r.speedup
        }
        for r in results
    ])

def plot_exec_time(df: pd.DataFrame, algorithm: str, size: int) -> None:
    '''
    Bar chart showing wall-clock exec time for each mode
    '''
    #Only keep rows matching the selected alg and size
    subset = df[(df['algorithm'] == algorithm) & (df['size'] == size)].sort_values('num_workers')

    fig, ax = plt.subplots(figsize=(8, 5))

    #One bar per exec mode
    bars = ax.bar(subset["label"], subset["elapsed_sec"], color=[_color(l) for l in subset["label"]], edgecolor="white")

    #Show the exact runtime above each bar
    ax.bar_label(bars, fmt="%.2fs", padding=4, fontsize=9)

    ax.set_title(f'Execution Time - {algorithm} (size = {size})', fontweight='bold')
    ax.set_xlabel('Execution Mode')
    ax.set_ylabel('Wall-clock Time (s)')
    ax.grid(axis='y', alpha=0.3)

    #Rotate labels a bit so long names dont overlap
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    path = os.path.join(RESULTS_DIR, f'exec_time_{algorithm}_{size}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_speedup(df: pd.DataFrame, algorithm: str, size: int) -> None:
    '''
    Line chart of speedup (relative to single process as a baseline) vs worker count
    '''
    #Exclude the single-process baselines because speedup lines are only useful for runs with more than one worker
    subset = df[(df["algorithm"] == algorithm) & (df["size"] == size) & (df["num_workers"] > 1)]

    if subset.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))

    #Plot a separate line for each parallel exec strategy
    for mode in ['multiprocess', 'distributed']:
        grp = subset[subset["label"].str.startswith(mode)].sort_values("num_workers")

        if not grp.empty:
            ax.plot(grp["num_workers"], grp["speedup"], marker="o", label=mode.capitalize(), color=COLORS.get(mode), linewidth=2)

    #Ideal linear speedup reference - if scaling was perfect, speedup would be exactly equal to worker count.
    #Real results fall below this due to overhead and communication costs.
    max_w = int(subset['num_workers'].max())
    ax.plot(range(1, max_w + 1), range(1, max_w + 1), "k--", alpha=0.35, label="Ideal (linear)")

    ax.set_title(f'Speedup vs Workers - {algorithm} (size={size})', fontweight='bold')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Speedup (x)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    path = os.path.join(RESULTS_DIR, f'speedup_{algorithm}_{size}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')

def plot_memory(df: pd.DataFrame, algorithm: str, size: int) -> None:
    """
    Bar chart showing peak RSS memory usage per execution mode.

    Memory typically increases with worker count due to process overhead
    and data duplication (each worker holds its own copy of inputs).

    Args:
        df:        Full results DataFrame.
        algorithm: Which algorithm to plot.
        size:      Which workload size to plot.
    """
    subset: pd.DataFrame = df[
        (df["algorithm"] == algorithm) & (df["size"] == size)
    ].sort_values("num_workers")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        subset["label"], subset["peak_memory_mb"],
        color=[_color(l) for l in subset["label"]], edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.1f MB", padding=4, fontsize=9)
    ax.set_title(f"Peak Memory — {algorithm} (size={size})", fontweight="bold")
    ax.set_xlabel("Execution Mode"); ax.set_ylabel("Peak RSS Memory (MB)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()

    path: str = os.path.join(RESULTS_DIR, f"memory_{algorithm}_{size}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_cpu(df: pd.DataFrame, algorithm: str, size: int) -> None:
    """
    Bar chart showing average CPU utilisation (across all cores) per mode.

    Single-process runs should show low overall CPU % (one core active out of N).
    Multiprocess runs should approach 100% when fully utilised across all cores.

    Args:
        df:        Full results DataFrame.
        algorithm: Which algorithm to plot.
        size:      Which workload size to plot.
    """
    subset: pd.DataFrame = df[
        (df["algorithm"] == algorithm) & (df["size"] == size)
    ].sort_values("num_workers")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        subset["label"], subset["avg_cpu_pct"],
        color=[_color(l) for l in subset["label"]], edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_title(f"Avg CPU Utilisation — {algorithm} (size={size})", fontweight="bold")
    ax.set_xlabel("Execution Mode"); ax.set_ylabel("Avg CPU % (all cores)")
    ax.set_ylim(0, 105)  # cap at 105 so bar labels don't get clipped at 100%
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()

    path: str = os.path.join(RESULTS_DIR, f"cpu_{algorithm}_{size}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def generate_all(results: List[BenchmarkResult]) -> None:
    '''
    Generate every chart and summary csv for the full result set
    '''

    df = result_to_df(results)

    #Save raw data first
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    csv_path = os.path.join(RESULTS_DIR, 'benchmark_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}\n')

    #For each alg/size combo, create the charts
    for combo in df[["algorithm", "size"]].drop_duplicates().values.tolist():
        algo = combo[0]
        size = combo[1]

        print(f"[{algo} | size={size}]")
        plot_exec_time(df, algo, size)
        plot_speedup(df, algo, size)
        plot_memory(df, algo, size)
        plot_cpu(df, algo, size)