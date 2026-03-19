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
