"""
distributed.py
Runs algorithms across a Ray cluster (distributed execution).

By default Ray starts a local cluster automatically. To point at a real cluster:
    ray.init(address="ray://<head-node-ip>:10001")
"""

import math
import numpy as np

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from algorithms import primes_chunk, matrix_multiplication_chunk
from metrics import run_benchmark, BenchmarkResult


def _ensure_ray():
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed. Run: pip install ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


@ray.remote
def _ray_primes_chunk(lo: int, hi: int) -> list[int]:
    return primes_chunk((lo, hi))


def _run_primes_distributed(limit: int, num_workers: int) -> list[int]:
    _ensure_ray()
    chunk_size = math.ceil(limit / num_workers)
    futures = [
        _ray_primes_chunk.remote(
            i * chunk_size,
            min((i + 1) * chunk_size - 1, limit),
        )
        for i in range(num_workers)
    ]
    results = ray.get(futures)
    return sorted(p for chunk in results for p in chunk)


def bench_primes_distributed(limit: int, num_workers: int) -> BenchmarkResult:
    return run_benchmark(
        label=f"distributed-{num_workers}",
        algorithm="primes-td",
        size=limit,
        num_workers=num_workers,
        fn=lambda: _run_primes_distributed(limit, num_workers),
    )


@ray.remote
def _ray_matmul_chunk(row_indices: list, size: int, seed: int) -> np.ndarray:
    return matrix_multiplication_chunk((row_indices, size, seed))


def _run_matmul_distributed(size: int, num_workers: int) -> np.ndarray:
    _ensure_ray()
    rows_per_worker = math.ceil(size / num_workers)
    futures = [
        _ray_matmul_chunk.remote(
            list(range(i * rows_per_worker, min((i + 1) * rows_per_worker, size))),
            size,
            42,
        )
        for i in range(num_workers)
        if i * rows_per_worker < size
    ]
    partial_results = ray.get(futures)
    return np.vstack(partial_results)


def bench_matmul_distributed(size: int, num_workers: int) -> BenchmarkResult:
    return run_benchmark(
        label=f"distributed-{num_workers}",
        algorithm="matrix multiplication",
        size=size,
        num_workers=num_workers,
        fn=lambda: _run_matmul_distributed(size, num_workers),
    )