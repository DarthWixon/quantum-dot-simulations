"""
Benchmark calculate_efg vs calculate_efg_vectorised.

Generates synthetic strain arrays at several grid sizes matching the scale of
the real Sokolov dataset (full dot region: 441×1100 ≈ 485,000 sites) and
measures execution time per size for both implementations.

Run from the repo root:
    python -m benchmarks.run_efg
"""

import numpy as np

from qdot.efg import calculate_efg, calculate_efg_vectorised
from benchmarks.timing import benchmark, compare, print_result, _fmt

# Synthetic strain values representative of the Sokolov dataset.
# Real data: ε_xx and ε_zz in [-0.02, 0.02], ε_xz in [-0.01, 0.01].
RNG = np.random.default_rng(42)


def make_strain(shape):
    xx = RNG.uniform(-0.02, 0.02, shape)
    xz = RNG.uniform(-0.01, 0.01, shape)
    zz = RNG.uniform(-0.02, 0.02, shape)
    return xx, xz, zz


TIMED_SIZES = [
    (10, 10),
    (50, 50),
    (100, 100),
    (200, 200),
]

PROJECTION_SIZE = (400, 400)

SPECIES = "Ga69"
N_RUNS = 3
WARMUP = 1

# ---------------------------------------------------------------------------

print()
print("calculate_efg  —  scalar vs vectorised")
print("=" * 60)
print(f"species: {SPECIES}   runs per size: {N_RUNS}  warmup: {WARMUP}")
print()

scalar_results = {}
vector_results = {}

for n, m in TIMED_SIZES:
    xx, xz, zz = make_strain((n, m))

    r_s = benchmark(
        f"scalar      {n}×{m}",
        calculate_efg,
        SPECIES,
        xx,
        xz,
        zz,
        n_runs=N_RUNS,
        warmup=WARMUP,
        n_sites=n * m,
    )
    r_v = benchmark(
        f"vectorised  {n}×{m}",
        calculate_efg_vectorised,
        SPECIES,
        xx,
        xz,
        zz,
        n_runs=N_RUNS,
        warmup=WARMUP,
        n_sites=n * m,
    )
    scalar_results[(n, m)] = r_s
    vector_results[(n, m)] = r_v

    print_result(r_s)
    print_result(r_v)
    compare(r_s, r_v)
    print()

# ---------------------------------------------------------------------------
# Single large run for both — unwarmed.

print(
    f"Large single run  {PROJECTION_SIZE[0]}×{PROJECTION_SIZE[1]}  (no warmup — approximate)"
)
print("-" * 60)
n, m = PROJECTION_SIZE
xx, xz, zz = make_strain((n, m))

r_s_large = benchmark(
    f"scalar      {n}×{m}",
    calculate_efg,
    SPECIES,
    xx,
    xz,
    zz,
    n_runs=1,
    warmup=0,
    n_sites=n * m,
)
r_v_large = benchmark(
    f"vectorised  {n}×{m}",
    calculate_efg_vectorised,
    SPECIES,
    xx,
    xz,
    zz,
    n_runs=1,
    warmup=0,
    n_sites=n * m,
)

print(
    f"  scalar      time: {_fmt(r_s_large.mean)}   per site: {_fmt(r_s_large.per_site)}"
)
print(
    f"  vectorised  time: {_fmt(r_v_large.mean)}   per site: {_fmt(r_v_large.per_site)}"
)
print(f"  speedup:    {r_s_large.mean / r_v_large.mean:.1f}×")
print()

# ---------------------------------------------------------------------------
# Projection to full Sokolov dataset scale.

dot_sites = 441 * 1100
s_per = r_s_large.per_site
v_per = r_v_large.per_site

print("Projections to real dataset  (based on 400×400 per-site time)")
print("-" * 60)
for label, sites in [
    ("dot region  step=1  (441×1100)", dot_sites),
    ("dot region  step=5  (88×220)", dot_sites // 25),
    ("dot region  step=10 (44×110)", dot_sites // 100),
]:
    print(f"  {label}")
    print(f"    scalar:      {_fmt(s_per * sites)}  ({sites:,} sites)")
    print(f"    vectorised:  {_fmt(v_per * sites)}")
    print(f"    speedup:     {s_per / v_per:.1f}×")
