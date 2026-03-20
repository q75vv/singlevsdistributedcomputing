'''
Executes both algorithms on a single process using a single CPU core.
'''

from algorithms import prime_sieve, prime_trial_division, matrix_multiplication
from metrics import run_benchmark, BenchmarkResult

def bench_prime_single(limit: int) -> BenchmarkResult:
    '''
    Benchmark the prime sieve on a single process
    '''
    return run_benchmark(label='single', algorithm='primes', size=limit, num_workers=1, fn=lambda: prime_sieve(limit))

def bench_prime_trial_division_single(limit: int) -> BenchmarkResult:
    '''
    Benchmark trial division on a single process.

    Uses the same algorithm as the parallel workers so speedup comparisons are fair.
    Results are stored under algorithm='primes-td' to keep them separate from the
    sieve baseline in charts and speedup calculations.
    '''
    return run_benchmark(label='single-td', algorithm='primes-td', size=limit, num_workers=1, fn=lambda: prime_trial_division(limit))

def bench_matrix_multiplication_single(size: int) -> BenchmarkResult:
    '''
    Benchmark the matrix multiplication on a single process
    '''
    return run_benchmark(label='single', algorithm='matrix multiplication', size=size, num_workers=1, fn=lambda: matrix_multiplication(size))