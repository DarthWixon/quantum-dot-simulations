"""
Electric field gradient (EFG) tensor calculations for nuclear spins in InGaAs quantum dots.

The EFG tensor at each lattice site is derived from the local strain tensor using
the gradient-elastic tensor components S11 and S44 for each nuclear species.
"""

import numpy as np
from qdot.isotopes import species_dict, old_species_dict


def euler_angles_from_rot_mat(rot_mat):
    """
    Extract Euler angles (X-Y-Z convention) from a rotation matrix.

    Finds one valid set of angles (alpha, beta, gamma) such that
    R = Rx(alpha) · Ry(beta) · Rz(gamma).

    Based on Slabaugh, "Computing Euler Angles from a Rotation Matrix."

    Args:
        rot_mat (ndarray): 3×3 rotation matrix.

    Returns:
        alpha, beta, gamma (float): Euler angles in radians.
    """
    if rot_mat[2, 0] != 1 and rot_mat[2, 0] != -1:
        beta = -np.arcsin(rot_mat[2, 0])
        alpha = np.arctan2(rot_mat[2, 1] / np.cos(beta), rot_mat[2, 2] / np.cos(beta))
        gamma = np.arctan2(rot_mat[1, 0] / np.cos(beta), rot_mat[0, 0] / np.cos(beta))
    else:
        gamma = 0.0
        if rot_mat[2, 0] == -1:
            beta = np.pi / 2
            alpha = gamma + np.arctan2(rot_mat[0, 1], rot_mat[0, 2])
        else:
            beta = -np.pi / 2
            alpha = -gamma + np.arctan2(-rot_mat[0, 1], -rot_mat[0, 2])
    return alpha, beta, gamma


def calculate_efg(nuclear_species, xx_array, xz_array, zz_array, use_sundfors=False):
    """
    Calculate the EFG tensor at every lattice site from the local strain tensor.

    Uses the gradient-elastic tensor components S11 and S44 for the given species.
    Eigenvalues are sorted |V_ZZ| ≥ |V_YY| ≥ |V_XX| (principal axis convention).

    Args:
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        xx_array (ndarray): ε_xx strain component, shape (n, m).
        xz_array (ndarray): ε_xz strain component, shape (n, m).
        zz_array (ndarray): ε_zz strain component, shape (n, m).
        use_sundfors (bool): Use older Sundfors (1974) parameter set if True.

    Returns:
        eta (ndarray): Biaxiality (V_XX - V_YY) / V_ZZ, shape (n, m).
        V_XX, V_YY, V_ZZ (ndarray): EFG principal components, shape (n, m).
        euler_angles (ndarray): Euler angles to the PAF at each site, shape (n, m, 3).
    """
    params = old_species_dict if use_sundfors else species_dict
    species = params[nuclear_species]

    S11 = species["S11"]
    S12 = -S11 / 2
    S44 = species["S44"]

    n, m = xx_array.shape
    eta = np.zeros((n, m))
    V_ZZ = np.zeros((n, m))
    V_XX = np.zeros((n, m))
    V_YY = np.zeros((n, m))
    euler_angles = np.zeros((n, m, 3))

    for i in range(n):
        for j in range(m):
            xx = xx_array[i, j]
            xz = xz_array[i, j]
            zz = zz_array[i, j]

            V = np.array([
                [S12 * (zz - xx),                  0,                S44 * xz],
                [0,                (S12 + S11) * xx + S12 * zz, S44 * xz],
                [S44 * xz,          S44 * xz,         2 * S12 * xx + S11 * zz],
            ])

            w, v = np.linalg.eig(V)
            abs_w = np.abs(w)

            idx = np.argsort(abs_w)[::-1]  # descending by magnitude
            V_ZZ[i, j] = w[idx[0]]
            V_YY[i, j] = w[idx[1]]
            V_XX[i, j] = w[idx[2]]

            eta[i, j] = (V_XX[i, j] - V_YY[i, j]) / V_ZZ[i, j]

            rot_mat = np.array([v[:, idx[2]], v[:, idx[1]], v[:, idx[0]]]).T
            euler_angles[i, j] = euler_angles_from_rot_mat(rot_mat)

    return eta, V_XX, V_YY, V_ZZ, euler_angles
