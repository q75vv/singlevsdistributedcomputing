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

