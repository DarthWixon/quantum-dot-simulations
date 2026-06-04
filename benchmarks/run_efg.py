"""
Benchmark the current calculate_efg implementation.

Generates synthetic strain arrays at several grid sizes matching the scale of
the real Sokolov dataset (full dot region: 441×1100 ≈ 485,000 sites) and
measures execution time per size.

Run from the repo root:
    python -m benchmarks.run_efg

The Sokolov dot region at step_size=1 is far too large to run many times with
the current implementation, so smaller sizes are timed properly (multiple
runs) and a single large run is used to project to real-world scale.
"""

import numpy as np

from qdot.efg import calculate_efg
from benchmarks.timing import benchmark, print_result, _fmt

# Synthetic strain values representative of the Sokolov dataset.
# Real data: ε_xx and ε_zz in [-0.02, 0.02], ε_xz in [-0.01, 0.01].
RNG = np.random.default_rng(42)

def make_strain(shape):
    xx = RNG.uniform(-0.02,  0.02, shape)
    xz = RNG.uniform(-0.01,  0.01, shape)
    zz = RNG.uniform(-0.02,  0.02, shape)
    return xx, xz, zz


# Sizes benchmarked with multiple runs.
# 200×200 = 40,000 sites is feasible; beyond that the loop takes minutes.
TIMED_SIZES = [
    (10,  10),
    (50,  50),
    (100, 100),
    (200, 200),
]

# One larger single-run measurement to anchor a projection to real scale.
PROJECTION_SIZE = (400, 400)

SPECIES   = "Ga69"
N_RUNS    = 3
WARMUP    = 1

# ---------------------------------------------------------------------------

print()
print("calculate_efg  —  current implementation")
print("=" * 55)
print(f"species: {SPECIES}   runs per size: {N_RUNS}  warmup: {WARMUP}")
print()

results = {}
for n, m in TIMED_SIZES:
    xx, xz, zz = make_strain((n, m))
    r = benchmark(
        f"calculate_efg  {n}×{m}",
        calculate_efg,
        SPECIES, xx, xz, zz,
        n_runs=N_RUNS,
        warmup=WARMUP,
        n_sites=n * m,
    )
    results[(n, m)] = r
    print_result(r)
    print()

# ---------------------------------------------------------------------------
# Single large run — not warmed up, results flagged as approximate.

print(f"Large single run  {PROJECTION_SIZE[0]}×{PROJECTION_SIZE[1]}  (no warmup — approximate)")
print("-" * 55)
n, m = PROJECTION_SIZE
xx, xz, zz = make_strain((n, m))
r_large = benchmark(
    f"calculate_efg  {n}×{m}",
    calculate_efg,
    SPECIES, xx, xz, zz,
    n_runs=1,
    warmup=0,
    n_sites=n * m,
)
print(f"  time:     {_fmt(r_large.mean)}")
print(f"  per site: {_fmt(r_large.per_site)}  ({r_large.n_sites:,} sites)")
print()

# ---------------------------------------------------------------------------
# Projection to full Sokolov dataset scale.

dot_sites    = 441 * 1100   # step_size=1, full dot region
fine_sites   = 441 * 1100   # same, shown for clarity
per_site_s   = r_large.per_site

print("Projections  (based on large single-run per-site time)")
print("-" * 55)
for label, sites in [
    ("dot region  step=1  (441×1100)", dot_sites),
    ("dot region  step=5  (88×220)",  dot_sites // 25),
    ("dot region  step=10 (44×110)",  dot_sites // 100),
]:
    projected = per_site_s * sites
    print(f"  {label:<38}  {_fmt(projected)}  ({sites:,} sites)")

print()
print("Note: projections assume linear scaling, which holds for the loop-based")
print("implementation since each site is processed independently.")
