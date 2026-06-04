"""
Toy spring-mass strain model for an InGaAs lattice.

Models atomic positions as a grid of springs, minimises the potential energy
using scipy.optimize.minimize to find the strained lattice, then calculates
the strain tensor at each site.

Species encoding: 0 = Ga, 1 = As, 2 = In
"""

import math
import numpy as np
import scipy.optimize as opt
from scipy.spatial import distance_matrix as scipy_distance_matrix

# ---------------------------------------------------------------------------
# Bond parameters
# ---------------------------------------------------------------------------


def _bond_parameters(
    real_atoms: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (spring_constants, natural_lengths) dicts keyed by species-pair string."""
    if real_atoms:
        # Stiffnesses from Vurgaftman 2001, DOI: 10.1063/1.1368156
        k = {
            "00": 1,
            "01": 2.56,
            "02": 1.22,
            "10": 2.56,
            "11": 2.99,
            "12": 2.30,
            "20": 1.22,
            "21": 2.30,
            "22": 1.58,
        }
        n = {
            "00": 1,
            "01": 1,
            "02": 1,
            "10": 1,
            "11": 1,
            "12": 1.06,
            "20": 1,
            "21": 1.06,
            "22": 1,
        }
    else:
        k = {
            "00": 1,
            "01": 5,
            "02": 50,
            "10": 5,
            "11": 10,
            "12": 1000,
            "20": 50,
            "21": 1000,
            "22": 200,
        }
        n = {
            "00": 1,
            "01": 1,
            "02": 1,
            "10": 1,
            "11": 1,
            "12": 0.1,
            "20": 1,
            "21": 0.1,
            "22": 1,
        }
    return k, n


# ---------------------------------------------------------------------------
# Lattice generators
# ---------------------------------------------------------------------------


def _gaas_base_lattice(n_rows: int, n_cols: int) -> np.ndarray:
    """GaAs alternating lattice with a 2-cell border for boundary conditions."""
    large = np.zeros((n_rows + 2, n_cols + 2), dtype=int)
    for r in range(n_rows + 2):
        for c in range(n_cols + 2):
            large[r, c] = 1 if (r % 2 == 0) == (c % 2 == 0) else 0
    # swap so Ga (0) and As (1) are correctly ordered
    large = 1 - large
    return large


def gaas_lattice(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Pure GaAs lattice, no Indium."""
    large = _gaas_base_lattice(n_rows, n_cols)
    return large, large[1 : n_rows + 1, 1 : n_cols + 1]


def gaas_lattice_with_indium_centre(
    n_rows: int, n_cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """GaAs lattice with a single In atom at the centre."""
    large = _gaas_base_lattice(n_rows, n_cols)
    cr, cc = int(math.floor(n_rows / 2)) + 1, int(math.floor(n_cols / 2)) + 1
    if large[cr, cc] == 0:
        large[cr, cc] = 2
    return large, large[1 : n_rows + 1, 1 : n_cols + 1]


def gaas_lattice_with_indium(
    n_rows: int, n_cols: int, in_positions: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """
    GaAs lattice with In atoms placed at specified positions.

    Args:
        n_rows, n_cols (int): Visible grid dimensions.
        in_positions (list of [x, y]): Positions of In atoms (0-indexed, bottom-left origin).

    Returns:
        large_array (ndarray): Full grid including border.
        simulated_array (ndarray): Visible grid, shape (n_rows, n_cols).
    """
    large = _gaas_base_lattice(n_rows, n_cols)
    for pos in in_positions:
        x, y = pos[0] + 1, pos[1] + 1
        if large[y, x] == 0:
            large[y, x] = 2
    return large, large[1 : n_rows + 1, 1 : n_cols + 1]


def block_in_positions(bottom_left: list[int], top_right: list[int]) -> list[list[int]]:
    """Generate a rectangular block of In positions."""
    return [
        [i, j]
        for i in range(bottom_left[0], top_right[0])
        for j in range(bottom_left[1], top_right[1])
    ]


# ---------------------------------------------------------------------------
# Spring parameter extraction
# ---------------------------------------------------------------------------


def _n_springs(n_rows: int, n_cols: int) -> int:
    return 2 * n_rows * n_cols + n_rows + n_cols


def spring_constants_from_lattice(
    large_species_array: np.ndarray, real_atoms: bool = True
) -> np.ndarray:
    """Extract spring constant for each spring from the species array."""
    k_dict, _ = _bond_parameters(real_atoms)
    n_rows = large_species_array.shape[0] - 2
    n_cols = large_species_array.shape[1] - 2
    constants = np.zeros(_n_springs(n_rows, n_cols))

    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            sr, sc = r - 1, c - 1
            own = large_species_array[r, c]
            above_idx = 2 * (n_rows + 1) * sc + 2 * sr
            left_idx = 2 * (n_cols + 1) * sr + 2 * sc + 1

            constants[above_idx] = k_dict[f"{own}{large_species_array[r - 1, c]}"]
            constants[left_idx] = k_dict[f"{own}{large_species_array[r, c - 1]}"]
            if r == n_rows:
                constants[above_idx + 2] = k_dict[
                    f"{own}{large_species_array[r + 1, c]}"
                ]
            if c == n_cols:
                constants[left_idx + 2] = k_dict[
                    f"{own}{large_species_array[r, c + 1]}"
                ]

    return constants


def natural_lengths_from_lattice(
    large_species_array: np.ndarray, real_atoms: bool = True
) -> np.ndarray:
    """Extract natural (rest) length for each spring from the species array."""
    _, n_dict = _bond_parameters(real_atoms)
    n_rows = large_species_array.shape[0] - 2
    n_cols = large_species_array.shape[1] - 2
    lengths = np.zeros(_n_springs(n_rows, n_cols))

    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            sr, sc = r - 1, c - 1
            own = large_species_array[r, c]
            above_idx = 2 * (n_rows + 1) * sc + 2 * sr
            left_idx = 2 * (n_cols + 1) * sr + 2 * sc + 1

            lengths[above_idx] = n_dict[f"{own}{large_species_array[r - 1, c]}"]
            lengths[left_idx] = n_dict[f"{own}{large_species_array[r, c - 1]}"]
            if r == n_rows:
                lengths[above_idx + 2] = n_dict[f"{own}{large_species_array[r + 1, c]}"]
            if c == n_cols:
                lengths[left_idx + 2] = n_dict[f"{own}{large_species_array[r, c + 1]}"]

    return lengths


# ---------------------------------------------------------------------------
# Energy function and position ↔ spring-length conversion
# ---------------------------------------------------------------------------


def _positions_to_spring_lengths(
    coords: np.ndarray,
    box_width: float,
    box_height: float,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    coords = np.reshape(coords, (n_rows, n_cols, 2))
    n_s = _n_springs(n_rows, n_cols)
    lengths = np.full(n_s, 1000.0)

    for r in range(n_rows):
        for c in range(n_cols):
            above_idx = 2 * (n_rows + 1) * c + 2 * r
            left_idx = 2 * (n_cols + 1) * r + 2 * c + 1

            x, y = coords[r, c]
            x_top = coords[r - 1, c, 0] if r > 0 else (c + 1) * box_width / (n_cols + 1)
            y_top = coords[r - 1, c, 1] if r > 0 else 0
            x_left = coords[r, c - 1, 0] if c > 0 else 0
            y_left = (
                coords[r, c - 1, 1] if c > 0 else (r + 1) * box_height / (n_rows + 1)
            )

            lengths[above_idx] = np.sqrt((x - x_top) ** 2 + (y - y_top) ** 2)
            lengths[left_idx] = np.sqrt((x - x_left) ** 2 + (y - y_left) ** 2)

            if r == n_rows - 1:
                x_bot = (c + 1) * box_width / (n_cols + 1)
                lengths[above_idx + 2] = np.sqrt(
                    (x - x_bot) ** 2 + (y - box_height) ** 2
                )
            if c == n_cols - 1:
                y_right = (r + 1) * box_height / (n_rows + 1)
                lengths[left_idx + 2] = np.sqrt(
                    (x - box_width) ** 2 + (y - y_right) ** 2
                )

    return lengths


def _potential_energy(
    coords: np.ndarray,
    spring_k: np.ndarray,
    natural_l: np.ndarray,
    box_width: float,
    box_height: float,
    n_rows: int,
    n_cols: int,
) -> float:
    lengths = _positions_to_spring_lengths(
        coords, box_width, box_height, n_rows, n_cols
    )
    return np.sum(spring_k * (lengths - natural_l) ** 2)


def unstrained_positions(n_rows: int, n_cols: int) -> np.ndarray:
    """Equilibrium (unstrained) atomic positions on a regular grid."""
    box_h = n_rows + 2
    box_w = n_cols + 2
    coords = np.zeros((n_rows, n_cols, 2))
    for r in range(n_rows):
        for c in range(n_cols):
            coords[r, c] = [
                (c + 1) * box_w / (n_cols + 1),
                (r + 1) * box_h / (n_rows + 1),
            ]
    return coords


# ---------------------------------------------------------------------------
# Strain tensor
# ---------------------------------------------------------------------------


def strain_tensor(
    strained_coords: np.ndarray, unstrained_coords: np.ndarray
) -> np.ndarray:
    """
    Calculate the 2D strain tensor at each atomic site.

    Uses a least-squares deformation gradient approach (comparing strained
    to unstrained inter-atomic vectors).

    Args:
        strained_coords (ndarray): Shape (n_rows, n_cols, 2).
        unstrained_coords (ndarray): Shape (n_rows, n_cols, 2).

    Returns:
        ndarray: Shape (n_rows, n_cols, 2, 2). Symmetric strain tensor at each site.
    """
    n_rows, n_cols = strained_coords.shape[:2]
    result = np.zeros((n_rows, n_cols, 2, 2))

    for r in range(n_rows):
        for c in range(n_cols):
            W = np.zeros((2, 2))
            V = np.zeros((2, 2))
            base_u = unstrained_coords[r, c]
            base_s = strained_coords[r, c]

            for tr in range(n_rows):
                for tc in range(n_cols):
                    du = base_u - unstrained_coords[tr, tc]
                    ds = base_s - strained_coords[tr, tc]
                    W += np.outer(ds, du)
                    V += np.outer(du, du)

            F = W @ np.linalg.pinv(V)
            grad_U = F - np.eye(2)
            result[r, c] = 0.5 * (grad_U + grad_U.T)

    return result


# ---------------------------------------------------------------------------
# Top-level simulation
# ---------------------------------------------------------------------------


def run_strain_simulation(
    n_rows: int,
    n_cols: int,
    lattice_type: int = 0,
    in_positions: list[list[int]] | None = None,
    real_atoms: bool = True,
    decimal_places: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Find the relaxed strained lattice by energy minimisation.

    Args:
        n_rows, n_cols (int): Number of visible rows and columns.
        lattice_type (int): 0 = pure GaAs, 1 = single central In, 2 = scattered In.
        in_positions (list): In atom positions, used only when lattice_type=2.
        real_atoms (bool): Use physical bond parameters if True.
        decimal_places (int): Rounding precision for output statistics.

    Returns:
        simulated_species (ndarray): Species array for the visible grid.
        unstrained (ndarray): Unstrained positions, shape (n_rows, n_cols, 2).
        strained (ndarray): Relaxed strained positions, shape (n_rows, n_cols, 2).
        time_taken (float): Optimisation time in seconds.
        avg_row_diff (float): Average row-length deviation (%).
        avg_col_diff (float): Average column-length deviation (%).
    """
    import time

    box_h = n_rows + 2
    box_w = n_cols + 2

    if in_positions is None:
        in_positions = []

    if lattice_type == 0:
        large, simulated_species = gaas_lattice(n_rows, n_cols)
    elif lattice_type == 1:
        large, simulated_species = gaas_lattice_with_indium_centre(n_rows, n_cols)
    else:
        large, simulated_species = gaas_lattice_with_indium(
            n_rows, n_cols, in_positions
        )

    spring_k = spring_constants_from_lattice(large, real_atoms)
    natural_l = natural_lengths_from_lattice(large, real_atoms)
    unstr = unstrained_positions(n_rows, n_cols)

    t0 = time.time()
    result = opt.minimize(
        _potential_energy,
        unstr.flatten(),
        args=(spring_k, natural_l, box_w, box_h, n_rows, n_cols),
    )
    t1 = time.time()

    strained = np.reshape(result.x, (n_rows, n_cols, 2))
    time_taken = np.around(t1 - t0, decimal_places)

    lengths = _positions_to_spring_lengths(strained, box_w, box_h, n_rows, n_cols)
    n_s = _n_springs(n_rows, n_cols)
    row_lengths = np.array(
        [
            np.sum(
                lengths[1 + 2 * r * n_cols : 1 + 2 * r * n_cols + 2 * n_cols + 1 : 2]
            )
            for r in range(n_rows)
        ]
    )
    col_lengths = np.array(
        [
            np.sum(lengths[2 * c * n_rows : 2 * c * n_rows + 2 * n_rows + 1 : 2])
            for c in range(n_cols)
        ]
    )
    avg_row_diff = np.around(
        (np.sqrt(np.sum((row_lengths - box_w) ** 2)) / n_rows) * 100 / box_w,
        decimal_places,
    )
    avg_col_diff = np.around(
        (np.sqrt(np.sum((col_lengths - box_h) ** 2)) / n_cols) * 100 / box_h,
        decimal_places,
    )

    return simulated_species, unstr, strained, time_taken, avg_row_diff, avg_col_diff
