"""
Benchmarking utilities for qdot functions.

    from benchmarks.timing import benchmark, compare, print_result

    result = benchmark("my_func", my_func, arg1, arg2, n_runs=5, warmup=1)
    print_result(result)

    compare(result_old, result_new)
"""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    times: list[float]
    n_sites: int | None = None

    @property
    def mean(self) -> float:
        return float(np.mean(self.times))

    @property
    def std(self) -> float:
        return float(np.std(self.times))

    @property
    def median(self) -> float:
        return float(np.median(self.times))

    @property
    def min(self) -> float:
        return float(np.min(self.times))

    @property
    def max(self) -> float:
        return float(np.max(self.times))

    @property
    def per_site(self) -> float | None:
        return self.mean / self.n_sites if self.n_sites else None


def benchmark(
    name: str,
    func,
    *args,
    n_runs: int = 5,
    warmup: int = 1,
    n_sites: int | None = None,
    **kwargs,
) -> BenchmarkResult:
    """
    Time func(*args, **kwargs) and return a BenchmarkResult.

    Args:
        name:    Label for this result.
        func:    Callable to time.
        *args:   Positional arguments forwarded to func.
        n_runs:  Number of timed runs.
        warmup:  Number of untimed warm-up calls before timing starts.
        n_sites: Optional site count — enables per-site timing in output.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        BenchmarkResult with timing statistics.
    """
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    return BenchmarkResult(name=name, times=times, n_sites=n_sites)


def compare(a: BenchmarkResult, b: BenchmarkResult) -> None:
    """
    Print a side-by-side comparison of two benchmark results.

    The second result (b) is treated as the candidate — the speedup shown
    is a.mean / b.mean, so values > 1 mean b is faster.
    """
    width = max(len(a.name), len(b.name)) + 2
    speedup = a.mean / b.mean if b.mean > 0 else float("inf")

    print("Comparison")
    print("-" * 50)
    print(f"  {a.name:<{width}} {_fmt(a.mean)} ± {_fmt(a.std)}")
    print(f"  {b.name:<{width}} {_fmt(b.mean)} ± {_fmt(b.std)}")
    print(f"  {'speedup':<{width}} {speedup:.1f}×  ", end="")
    if speedup >= 1:
        print(f"({b.name} is faster)")
    else:
        print(f"({a.name} is faster)")


def print_result(result: BenchmarkResult) -> None:
    """Print a formatted summary of a BenchmarkResult."""
    print(result.name)
    print("-" * len(result.name))
    print(f"  runs:    {len(result.times)}")
    print(f"  mean:    {_fmt(result.mean)}  ±  {_fmt(result.std)}")
    print(f"  median:  {_fmt(result.median)}")
    print(f"  range:   {_fmt(result.min)}  –  {_fmt(result.max)}")
    if result.per_site is not None:
        print(f"  per site: {_fmt(result.per_site)}  ({result.n_sites:,} sites)")


def _fmt(seconds: float) -> str:
    """Format a duration with auto-scaling units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"
