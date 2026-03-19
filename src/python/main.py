import sys
import os
import multiprocessing
from typing import Dict, List, Tuple, Optional

from single_runner import bench_prime_single, bench_matrix_multiplication_single
from distributed_runner import bench_primes_multiprocessing, bench_matrix_multiplication_distributed
from metrics import BenchmarkResult
import visualizations as vis

#Worker counts to test for both multiprocess and distributed modes
WORKER_COUNTS = [2,4,8,10]

#Standard benchmark - handful of sizes for main comparison charts
SIZES = {
    'primes': [500_000, 2_000_000],
    'matrix multiplication': [512, 1024]
}

#Crossover sweeps - many sizes to find exact point where parallelism starts paying off
SWEEP_SIZES = {
    "primes": [
        10_000, 25_000, 50_000, 75_000,
        100_000, 250_000, 500_000, 750_000,
        1_000_000, 1_500_000, 2_000_000,
    ],
    "matrix multiplication": [
        64, 128, 192, 256,
        384, 512, 768, 1024,
    ],
}

def find_crossover(results: List[BenchmarkResult], mode_prefix: str, algorithm: str) -> Optional[str]:
    '''
    Find smallest workload size at which a given paralell mode first achieves speedup > 1
    '''

    #Collect all results for this mode and algo that have a computed speedup
    relevant = [r for r in results if r.algorithm == algorithm and r.label.startswith(mode_prefix) and r.speedup > 0.0]

    #Sort ascending
    relevant.sort(key=lambda r: r.size)

    for r in relevant:
        if r.speedup > 1.0:
            return r.size
        
    return None #parallelism never overcame its overhead in the tested range

def report_crossovers(results: List[BenchmarkResult]) -> None:
    '''
    Human readable crossover summary
    '''

    algorithms = sorted({r.algorithm for r in results})
    modes = ['multiprocess', 'distributed']

    print('\n Crossover Points (smallest points where parallelism beats single-process)')
    for algo in algorithms:
        for mode in modes:
            #Only report modes that actually appear in results
            if not any(r.label.startswith(mode) for r in results):
                continue
            crossover = find_crossover(results, mode, algo)
            if crossover is not None:
                print(f'{algo:<8} {mode:<15} --> first beats single at size {crossover:<10,}')
            else:
                print(f'{algo:<8} {mode:<15} --> never beat single process in tested range')

def compute_speedup(results: List[BenchmarkResult]) -> List[BenchmarkResult]:
    '''
    Populate speedup field on each result by dividing the single process baseline time by each mode elapsed time
    '''

    #Make lookup of single process time keyed by (alg, size)
    baselines = {}
    for r in results:
        if r.label == 'single':
            baselines[(r.algorithm, r.size)] = r.elapsed_sec

    #Divide baseline time by each result to get the speedup multiplier
    for r in results:
        key = (r.algorithm, r.size)
        if key in baselines and baselines[key] > 0:
            r.speedup = round(baselines[key] / r.elapsed_sec, 4)
    return results

def _header(msg: str) -> None:
    '''Print a section divider with title'''
    print('\n' + '='*60)
    print(f'{msg}')
    print('='*60)

def _result_line(r: BenchmarkResult) -> None:
    '''Print a single result on one line'''
    print(f"  [{r.label:<22}]  {r.elapsed_sec:>7.3f}s mem={r.peak_memory_mb:>7.1f}MB cpu={r.avg_cpu_pct:>5.1f}%")


