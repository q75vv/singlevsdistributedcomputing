'''
Executes both algorithms on a single process using a single CPU core.
'''

from algorithms import prime_sieve, matrix_multiplication
from metrics import run_benchmark, BenchmarkResult

def bench_prime_single(limit: int) -> BenchmarkResult:
    '''
    Benchmark the prime sieve on a single process
    '''
    return run_benchmark(label='single', algorithm='primes', size=limit, num_workers=1, fn=lambda: prime_sieve(limit))