'''
Defines a a few CPU intensive algorithms
Prime number generation via the Sieve of Eratosthenes (https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes#Algorithm_and_variants)
Matrix Multiplication via NumPy
'''

import math
import numpy as np
from typing import Tuple, List

#Prime number genreation

def prime_sieve(limit: int) -> List[int]:
    """
    Return all prime numbers up to limit (inclusive) using the Sieve of Eratosthenes
    - https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes#Algorithm_and_variants

    Start with boolean arr of size limit + 1, set all True, meaning each number is possibly prime.
    Mark 0 & 1 as not prime.
    For each number i starting at 2, if i is still marked prime, mark all multiples of i starting at i^2 as not prime.
    Only iterate up to sqrt(limit). Any non-prime number must have at leaast one factor that is <= its square root, so after that point the remaining unmarked numbers are prime.

    """

    if limit < 2:
        return []
    
    #Using bytearray instead of list of bools is much more memory efficient
    sieve = bytearray([1]) * (limit + 1)

    #0 & 1 are not prime numbers
    sieve[0] = 0
    sieve[1] = 0

    #Only iterate up to sqrt(limit)
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            #Zero out every multiple of i starting at i*i (all smaller multiples were already handled by earlier primes)
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))

    #Collect the indices that are still marked as prime
    return [i for i, v in enumerate(sieve) if v]

def prime_trial_division(limit: int) -> List[int]:
    """
    Return all prime numbers up to limit (inclusive) using trial division.

    This is the single-process equivalent of primes_chunk — it uses the same algorithm
    as the parallel workers so that speedup comparisons are fair. Each number is tested
    individually by checking divisibility against all integers up to its square root.

    This is intentionally slower than the sieve. The point is to give multiprocess and
    distributed runners a realistic baseline to beat.
    """

    results = []
    for n in range(2, limit + 1):
        if all(n % d != 0 for d in range(2, int(math.isqrt(n)) + 1)):
            results.append(n)
    return results
    
def primes_chunk(args: Tuple[int, int]) -> List[int]:
    '''
    Finds all primes within subrange [lo, hi] using trial division.

    This is a worker function to be used by multiprocessing and distributed runners. It uses a trial division approach instead of
    a sieve because each worker only recieves a slice of the number line, and building a fill sieve for every worker would be wasteful and 
    defeat the purpose of splitting the work.

    We pass in a tuple because multiprocessing.Pool.map and Ray can only pass a single argument to worker functions.
    '''

    lo, hi = args
    results = []

    for n in range(max(lo, 2), hi + 1):
        if n < 2:
            continue
        #Trial division: n is prime if no integer in [2, sqrt(n)] divides it
        if all(n % d != 0 for d in range(2, int(math.isqrt(n)) + 1)):
            results.append(n)

    return results


#Matrix multiplication
def matrix_multiplication(size: int) -> np.ndarray:
    '''
    Multiply two randomly generated (size x size) float64 matricies and return the result

    Use a fixed random seed to make sure every runner (single, multiprocess, distributed) operates on the same data
    '''
    #Random number generator with specified seed
    rng = np.random.default_rng(seed=2026)

    #Create random size x size matricies
    A = rng.random((size, size))
    B = rng.random((size, size))

    
    return A @ B #Shorthand for np.matmul(A, B). Uses Basic Linear Algebra Subprograms under the hood.\

def matrix_multiplication_chunk(args: Tuple[List[int], int, int]) -> np.ndarray:
    '''
    Multiply a horizontal slice of matrix A by the full matrix B

    Worker function for parallel and distributed matrix multiplication. Each worker is given a subset of A's rows and computes only those rows of the final product.
    The results are later stacked in order by the coordinator to make the full result matrix.
    '''

    row_indicies, size, seed = args

    rng = np.random.default_rng(seed=seed)
    A_full = rng.random((size, size))
    B  = rng.random((size, size)) #Use the same rng sequence.

    #Slice out only this worker's rows before multiplying
    A_chunk = A_full[row_indicies]

    return A_chunk @ B