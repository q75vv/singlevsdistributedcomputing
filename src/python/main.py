import sys
import os
import argparse
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

