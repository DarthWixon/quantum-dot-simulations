"""
Site sampling utilities for selecting lattice locations within a region.

Locations are (row, column) index pairs into the 2D arrays produced by the
EFG and strain loaders. Functions take array shapes or strain arrays directly
so they stay decoupled from any particular archive format.
"""

import numpy as np


def all_locations(shape: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Enumerate every lattice site in a region.

    Args:
        shape (tuple): (n_rows, n_cols) of the region arrays.

    Returns:
        list of (int, int): All (row, column) index pairs.
    """
    n_rows, n_cols = shape
    return [(i, j) for i in range(n_rows) for j in range(n_cols)]


def random_locations(
    n_locations: int,
    shape: tuple[int, int],
    rng: np.random.Generator | int | None = None,
) -> list[tuple[int, int]]:
    """
    Draw lattice sites uniformly at random from a region (with replacement).

    Args:
        n_locations (int): Number of sites to draw.
        shape (tuple): (n_rows, n_cols) of the region arrays.
        rng: numpy Generator, integer seed, or None for fresh entropy.

    Returns:
        list of (int, int): Random (row, column) index pairs.
    """
    rng = np.random.default_rng(rng)
    rows = rng.integers(shape[0], size=n_locations)
    cols = rng.integers(shape[1], size=n_locations)
    return list(zip(rows.tolist(), cols.tolist()))


def find_best_locations(
    n_locations: int,
    xx_array: np.ndarray,
    xz_array: np.ndarray,
    zz_array: np.ndarray,
    search_range: int,
    rng: np.random.Generator | int | None = None,
) -> list[tuple[int, int]]:
    """
    Draw random sites and snap each to the local maximum of the summed strain.

    For each random seed site, searches the (2·search_range + 1)² window
    around it in the metric ε_xx + ε_xz + ε_zz and returns the coordinates of
    the window maximum, so sampled sites sit on strongly-strained nuclei.

    The plain component sum as a strain metric is inherited from the original
    research code; see human-todo. before relying on it quantitatively.

    Args:
        n_locations (int): Number of sites to return.
        xx_array, xz_array, zz_array (ndarray): Strain components, shape (n, m).
        search_range (int): Half-width of the search window in sites.
        rng: numpy Generator, integer seed, or None for fresh entropy.

    Returns:
        list of (int, int): (row, column) index pairs at local strain maxima.
    """
    metric = xx_array + xz_array + zz_array
    n_rows, n_cols = metric.shape

    best = []
    for row, col in random_locations(n_locations, metric.shape, rng):
        row_min = max(row - search_range, 0)
        row_max = min(row + search_range + 1, n_rows)
        col_min = max(col - search_range, 0)
        col_max = min(col + search_range + 1, n_cols)

        window = metric[row_min:row_max, col_min:col_max]
        rel_row, rel_col = np.unravel_index(np.argmax(window), window.shape)
        best.append((row_min + int(rel_row), col_min + int(rel_col)))

    return best
