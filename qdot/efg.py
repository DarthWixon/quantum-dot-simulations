"""
Electric field gradient (EFG) tensor calculations for nuclear spins in InGaAs quantum dots.

The EFG tensor at each lattice site is derived from the local strain tensor using
the gradient-elastic tensor components S11 and S44 for each nuclear species.
"""

import numpy as np
from qdot.isotopes import species_dict, old_species_dict


def euler_angles_from_rot_mat(rot_mat):
    """
    Extract Euler angles from a rotation matrix.

    Finds angles (alpha, beta, gamma) such that
    R = Rz(gamma) · Ry(beta) · Rx(alpha).

    Based on Slabaugh, "Computing Euler Angles from a Rotation Matrix."
    Note: the original code docstring incorrectly stated Rx·Ry·Rz;
    the element-wise extraction (rot[2,0] = −sin β) corresponds to Rz·Ry·Rx.

    Args:
        rot_mat (ndarray): 3×3 proper rotation matrix (det = +1).

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

            # eta is undefined (NaN) when V_ZZ is zero (no strain = no EFG)
            if V_ZZ[i, j] != 0:
                eta[i, j] = (V_XX[i, j] - V_YY[i, j]) / V_ZZ[i, j]
            else:
                eta[i, j] = np.nan

            rot_mat = np.array([v[:, idx[2]], v[:, idx[1]], v[:, idx[0]]]).T
            # eig eigenvector signs are arbitrary → det can be −1 (improper rotation).
            # Force det = +1 by recomputing the Z column as the right-hand cross product.
            rot_mat[:, 2] = np.cross(rot_mat[:, 0], rot_mat[:, 1])
            euler_angles[i, j] = euler_angles_from_rot_mat(rot_mat)

    return eta, V_XX, V_YY, V_ZZ, euler_angles


def calculate_efg_vectorised(
    nuclear_species: str,
    xx_array: np.ndarray,
    xz_array: np.ndarray,
    zz_array: np.ndarray,
    use_sundfors: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised EFG tensor calculation — identical interface to calculate_efg.

    Replaces the per-site Python loop with batched NumPy operations:

      1. All (n × m) EFG tensors are constructed in one pass via array slicing
         (no loop, pure element-wise arithmetic).
      2. np.linalg.eigh processes the entire (n, m, 3, 3) stack in one BLAS call.
         eigh exploits the matrix symmetry — faster and numerically better than
         the general eig used in calculate_efg for real symmetric matrices.
      3. Eigenvalue sorting uses argsort on the last axis with take_along_axis
         to simultaneously reorder both eigenvalues and eigenvector columns.
      4. Euler angles are extracted with np.where chains instead of per-site
         branches, including the gimbal-lock case (rot[2,0] == ±1).

    The eigenvalues (V_XX, V_YY, V_ZZ) and η are numerically equivalent to
    calculate_efg. The Euler angles describe the same physical rotations but
    may differ in sign from the scalar version because eigh and eig can return
    eigenvectors with different sign conventions.

    Args:
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        xx_array (ndarray): ε_xx strain component, shape (n, m).
        xz_array (ndarray): ε_xz strain component, shape (n, m).
        zz_array (ndarray): ε_zz strain component, shape (n, m).
        use_sundfors (bool): Use Sundfors (1974) parameters if True.

    Returns:
        eta, V_XX, V_YY, V_ZZ (ndarray): shape (n, m).
        euler_angles (ndarray): shape (n, m, 3).
    """
    params = old_species_dict if use_sundfors else species_dict
    species = params[nuclear_species]

    S11 = species["S11"]
    S12 = -S11 / 2
    S44 = species["S44"]

    n, m = xx_array.shape
    xx, xz, zz = xx_array, xz_array, zz_array

    # ------------------------------------------------------------------
    # Step 1 — build all EFG tensors at once.
    #
    # V is real symmetric. The unique elements are:
    #   V[0,0] = S12*(zz - xx)
    #   V[1,1] = (S12+S11)*xx + S12*zz
    #   V[2,2] = 2*S12*xx + S11*zz
    #   V[0,2] = V[2,0] = V[1,2] = V[2,1] = S44*xz
    #   V[0,1] = V[1,0] = 0
    # ------------------------------------------------------------------
    V = np.zeros((n, m, 3, 3))
    V[:, :, 0, 0] = S12 * (zz - xx)
    V[:, :, 1, 1] = (S12 + S11) * xx + S12 * zz
    V[:, :, 2, 2] = 2 * S12 * xx + S11 * zz
    V[:, :, 0, 2] = S44 * xz
    V[:, :, 2, 0] = S44 * xz
    V[:, :, 1, 2] = S44 * xz
    V[:, :, 2, 1] = S44 * xz

    # ------------------------------------------------------------------
    # Step 2 — batched eigendecomposition.
    #
    # eigh input:  (n, m, 3, 3)
    # w output:    (n, m, 3)    eigenvalues, ascending algebraic order
    # v output:    (n, m, 3, 3) v[i,j,:,k] is the k-th eigenvector at (i,j)
    # ------------------------------------------------------------------
    w, v = np.linalg.eigh(V)

    # ------------------------------------------------------------------
    # Step 3 — sort by descending |eigenvalue|: V_ZZ ≥ V_YY ≥ V_XX.
    # ------------------------------------------------------------------
    sort_idx = np.argsort(np.abs(w), axis=-1)[:, :, ::-1]     # (n, m, 3)

    w_sorted  = np.take_along_axis(w, sort_idx, axis=-1)
    V_ZZ = w_sorted[:, :, 0]
    V_YY = w_sorted[:, :, 1]
    V_XX = w_sorted[:, :, 2]

    # Reorder eigenvector columns to match the sorted eigenvalues.
    # Broadcasting sort_idx from (n,m,3) → (n,m,3,3) applies the same
    # column permutation to every row of the eigenvector matrix.
    sort_idx_v = np.broadcast_to(sort_idx[:, :, np.newaxis, :], (n, m, 3, 3))
    v_sorted = np.take_along_axis(v, sort_idx_v, axis=-1)      # (n, m, 3, 3)
    # v_sorted[:,:,:, 0] = V_ZZ eigenvector at every site
    # v_sorted[:,:,:, 1] = V_YY eigenvector at every site
    # v_sorted[:,:,:, 2] = V_XX eigenvector at every site

    # ------------------------------------------------------------------
    # Step 4 — biaxiality η = (V_XX − V_YY) / V_ZZ.
    # Undefined where V_ZZ = 0 (no strain → no EFG).
    # ------------------------------------------------------------------
    eta = np.where(V_ZZ != 0, (V_XX - V_YY) / V_ZZ, np.nan)

    # ------------------------------------------------------------------
    # Step 5 — assemble rotation matrices.
    #
    # Columns: [V_XX eigvec | V_YY eigvec | V_ZZ eigvec]
    # Matches the scalar:  rot = np.array([v[:,idx[2]], v[:,idx[1]], v[:,idx[0]]]).T
    # ------------------------------------------------------------------
    rot = np.stack([
        v_sorted[:, :, :, 2],   # V_XX eigenvector → column 0
        v_sorted[:, :, :, 1],   # V_YY eigenvector → column 1
        v_sorted[:, :, :, 0],   # V_ZZ eigenvector → column 2
    ], axis=-1)                  # (n, m, 3, 3)

    # eigh eigenvector signs are arbitrary → det(rot) can be −1 (improper rotation).
    # Force det = +1 site-wise by recomputing column 2 as the right-hand cross product.
    # np.cross on (n,m,3) inputs returns (n,m,3); no loop needed.
    rot[:, :, :, 2] = np.cross(rot[:, :, :, 0], rot[:, :, :, 1])

    # ------------------------------------------------------------------
    # Step 6 — vectorised Euler angle extraction (X-Y-Z, Slabaugh 2008).
    #
    # Gimbal lock: rot[:,:,2,0] == ±1 → cos(beta) = 0, handled by np.where.
    # np.where evaluates both branches everywhere and selects; safe_cos
    # prevents division-by-zero in the normal-case expressions at gimbal sites.
    # ------------------------------------------------------------------
    r20    = rot[:, :, 2, 0]
    gimbal = np.abs(r20) >= 1.0

    beta_normal  = -np.arcsin(np.clip(r20, -1.0, 1.0))
    cos_beta     = np.cos(beta_normal)
    safe_cos     = np.where(np.abs(cos_beta) > 1e-10, cos_beta, 1.0)

    alpha_normal = np.arctan2(rot[:, :, 2, 1] / safe_cos, rot[:, :, 2, 2] / safe_cos)
    gamma_normal = np.arctan2(rot[:, :, 1, 0] / safe_cos, rot[:, :, 0, 0] / safe_cos)

    # Gimbal lock sub-cases (gamma is fixed to 0 in both).
    alpha_neg = np.arctan2( rot[:, :, 0, 1],  rot[:, :, 0, 2])   # r20 == -1
    alpha_pos = np.arctan2(-rot[:, :, 0, 1], -rot[:, :, 0, 2])   # r20 == +1

    beta  = np.where(gimbal, np.where(r20 < 0, np.pi / 2, -np.pi / 2), beta_normal)
    alpha = np.where(gimbal, np.where(r20 < 0, alpha_neg,  alpha_pos),  alpha_normal)
    gamma = np.where(gimbal, 0.0, gamma_normal)

    euler_angles = np.stack([alpha, beta, gamma], axis=-1)         # (n, m, 3)

    return eta, V_XX, V_YY, V_ZZ, euler_angles
