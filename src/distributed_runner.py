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
    return run_benchmark(label=f'multiprocess-{num_workers=}', algorithm='primes', size=limit, num_workers=num_workers, 
                         fn=lambda: _run_primes_multiprocessing(limit, num_workers))