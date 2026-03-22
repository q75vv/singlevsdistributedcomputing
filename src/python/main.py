import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import multiprocessing
from typing import Dict, List, Tuple, Optional

from single_runner import bench_prime_single, bench_prime_trial_division_single, bench_matrix_multiplication_single
from multithread_runner import bench_primes_multiprocessing, bench_matrix_multiplication_distributed
from distributed_runner import bench_primes_distributed, bench_matmul_distributed, RAY_AVAILABLE
from metrics import BenchmarkResult
import visualizations as vis

#Worker counts to test for both multiprocess and distributed modes
WORKER_COUNTS = [2,4,8,10]


#Standard benchmark - handful of sizes for main comparison charts
SIZES = {
    'primes': [500_000, 2_000_000, 3_000_000, 5_000_000],
    'matrix multiplication': [512, 1024, 2048, 4096]
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

SWEEP_WORKERS = 4

def find_crossover(results: List[BenchmarkResult], mode_prefix: str, algorithm: str) -> Optional[int]:
    '''
    Find smallest workload size at which a given parallel mode first achieves speedup > 1
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
    Populate speedup field on each result by dividing the single process baseline time by each mode elapsed time.

    Uses 'single-td' as the baseline for 'primes-td' results, and 'single' for everything else.
    '''

    #Make lookup of single process time keyed by (alg, size)
    baselines = {}
    for r in results:
        if r.label in ('single', 'single-td'):
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

def run(sizes: Dict[str, List[int]], fixed_workers: Optional[int]=None) -> List[BenchmarkResult]:
    '''
    Execute every benchmark combination and return all results
    '''

    all_results = []

    #Cap worker count at available cpu count
    max_workers = min(max(WORKER_COUNTS), multiprocessing.cpu_count())
    workers = [min(fixed_workers, max_workers)] if fixed_workers is not None else [w for w in WORKER_COUNTS if w <= max_workers] or [2]

    #Primes
    for limit in sizes['primes']:
        _header(f'PRIMES limit={limit:,}')

        #Sieve baseline (fast, not directly comparable to parallel trial division)
        #r = bench_prime_single(limit)
        #all_results.append(r); _result_line(r)

        #Trial division baseline (same algorithm as parallel workers - fair comparison)
        r = bench_prime_trial_division_single(limit)
        all_results.append(r); _result_line(r)

        #Multiprocess (trial division)
        for w in workers:
            r = bench_primes_multiprocessing(limit, w)
            all_results.append(r); _result_line(r)

        #Distributed (Ray, trial division)
        if RAY_AVAILABLE:
            for w in workers:
                r = bench_primes_distributed(limit, w)
                all_results.append(r); _result_line(r)
        else:
            print('  [distributed] skipped - Ray not installed')

    #Matrix Mult
    for size in sizes['matrix multiplication']:
        _header(f'MATRIX MULTIPLICATION size={size}x{size}')

        #Single process
        r = bench_matrix_multiplication_single(size)
        all_results.append(r); _result_line(r)

        #Multiprocess
        for w in workers:
            r = bench_matrix_multiplication_distributed(size, w)
            all_results.append(r); _result_line(r)

        #Distributed (Ray)
        if RAY_AVAILABLE:
            for w in workers:
                r = bench_matmul_distributed(size, w)
                all_results.append(r); _result_line(r)
        else:
            print('  [distributed] skipped - Ray not installed')

    return all_results

#Entry point

def main() -> None:
    _header("STANDARD BENCHMARK")
    results = run(SIZES)
    results = compute_speedup(results)

    _header('CREATING STANDARD CHARTS')
    vis.generate_all(results)

    #Crossover sweep
    _header(f'CROSSOVER SWEEP ({SWEEP_WORKERS}) workers')
    sweep_results = run(SWEEP_SIZES, fixed_workers=SWEEP_WORKERS)
    sweep_results = compute_speedup(sweep_results)

    _header('CROSSOVER ANALYSIS')
    report_crossovers(sweep_results)

    _header('CREATING CROSSOVER CHARTS')
    #NEED THIS CODE

    if RAY_AVAILABLE:
        import ray
        ray.shutdown()

    print('\n DONE, CHARTS SAVED')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()