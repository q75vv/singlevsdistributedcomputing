'''
Defines a a few CPU intensive algorithms
Prime number generation via the Sieve of Eratosthenes (https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes#Algorithm_and_variants)
Matrix Multiplication via NumPy
'''

import math
import numpy

#Prime number genreation
def prime_sieve(limit: int) -> list[int]:
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
    
