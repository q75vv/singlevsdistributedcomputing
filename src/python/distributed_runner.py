'''
Parallel runner using Python's multiprocessing.Pool. Shared mem parallelism where all workers run on the same machine and can use
multiple CPU cores simultaneously

Work is split into equal-sized chunks so each worker get almost the same amount of work

Use pool.map() because it blocks until all workers are done to make timing easier.

The Pool is opened and closed inside the timed lambda in run_benchmark, meaning pool creation overhead IS included in the elapsed time.
This reflects real-world cost.
'''

import math
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from multiprocessing import Pool

from algorithms import primes_chunk, matrix_multiplication_chunk
from metrics import run_benchmark, BenchmarkResult


def _run_primes_multiprocessing(limit: int, num_workers: int) -> List[int]:
    '''
    Divide the range[2, limit] into num_workers roughly equal sub-ranges, give each sub-range to a worker, and merge and sort results.
    '''

    #Divide search space into equal chunks for each worker
    chunk_size = math.ceil(limit / num_workers)
    ranges = [(max(2, i * chunk_size), min((i + 1) * chunk_size - 1, limit)) for i in range(num_workers)]

    #pool.map blocks until all workers finish, then return results in order
    with Pool(processes=num_workers) as pool:
        results = pool.map(primes_chunk, ranges)

    #Flatten the list of lists and sort - workers return unsorted sublists
    return sorted(p for chunk in results for p in chunk)

def bench_primes_multiprocessing(limit: int, num_workers: int) -> BenchmarkResult:
    '''
    Benchmark multiprocess primes with given number of workers
    '''
    return run_benchmark(label=f'multiprocess-{num_workers=}', algorithm='primes-td', size=limit, num_workers=num_workers, 
                         fn=lambda: _run_primes_multiprocessing(limit, num_workers))

def _run_matrix_multiplication_multiprocessing(size: int, num_workers: int) -> np.ndarray:
    '''
    Split matrix A into horizontal slices, compute each slices contribution to A @ B in seperate worker processes, 
    and combine result
    '''

    rows_per_worker = math.ceil(size / num_workers)

    #Build list of (row_indicies, size, seed) tuples for each worker
    chunks = [
        (
            list(range(i * rows_per_worker, min((i + 1) * rows_per_worker, size))),
            size,
            42,   # fixed seed — must match matrix_multiply() so A and B are identical
        )
        for i in range(num_workers)
        if i * rows_per_worker < size  # skip empty chunks if workers > rows
    ]

    with Pool(processes=num_workers) as pool:
        partial_results = pool.map(matrix_multiplication_chunk, chunks)

    #Stack partial row-band results vertically to make full result matrix
    return np.vstack(partial_results)

def bench_matrix_multiplication_distributed(size: int, num_workers: int) -> BenchmarkResult:
    '''
    Benchmark multiprocess matrix multiplication with given number of workers
    '''

    return run_benchmark(label=f'multiprocess-{num_workers}', algorithm='matrix multiplication', size=size, num_workers=num_workers, 
                         fn=lambda: _run_matrix_multiplication_multiprocessing(size, num_workers))