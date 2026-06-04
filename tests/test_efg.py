import numpy as np
import pytest
from qdot.efg import euler_angles_from_rot_mat, calculate_efg, calculate_efg_vectorised


def test_euler_angles_identity():
    alpha, beta, gamma = euler_angles_from_rot_mat(np.eye(3))
    assert abs(alpha) < 1e-10
    assert abs(beta) < 1e-10
    assert abs(gamma) < 1e-10


def test_calculate_efg_zero_strain_gives_zero_efg():
    n, m = 4, 4
    xx = np.zeros((n, m))
    xz = np.zeros((n, m))
    zz = np.zeros((n, m))
    eta, V_XX, V_YY, V_ZZ, euler_angles = calculate_efg("Ga69", xx, xz, zz)
    np.testing.assert_allclose(V_ZZ, 0.0, atol=1e-25)
    np.testing.assert_allclose(V_XX, 0.0, atol=1e-25)
    np.testing.assert_allclose(V_YY, 0.0, atol=1e-25)


def test_calculate_efg_output_shapes():
    n, m = 3, 5
    xx = np.random.rand(n, m) * 0.01
    xz = np.random.rand(n, m) * 0.01
    zz = np.random.rand(n, m) * 0.01
    eta, V_XX, V_YY, V_ZZ, euler_angles = calculate_efg("In115", xx, xz, zz)
    assert eta.shape == (n, m)
    assert V_ZZ.shape == (n, m)
    assert euler_angles.shape == (n, m, 3)


def test_calculate_efg_biaxiality_definition():
    # eta = (V_XX - V_YY) / V_ZZ by definition
    n, m = 2, 2
    xx = np.array([[0.01, 0.02], [0.03, 0.01]])
    xz = np.array([[0.005, 0.01], [0.002, 0.008]])
    zz = np.array([[0.02, 0.01], [0.015, 0.025]])
    eta, V_XX, V_YY, V_ZZ, _ = calculate_efg("As75", xx, xz, zz)
    np.testing.assert_allclose(eta, (V_XX - V_YY) / V_ZZ, atol=1e-10)


def test_calculate_efg_sundfors_gives_different_result():
    n, m = 2, 2
    xx = np.ones((n, m)) * 0.01
    xz = np.ones((n, m)) * 0.005
    zz = np.ones((n, m)) * 0.02
    _, _, _, V_ZZ_new, _ = calculate_efg("Ga69", xx, xz, zz, use_sundfors=False)
    _, _, _, V_ZZ_old, _ = calculate_efg("Ga69", xx, xz, zz, use_sundfors=True)
    assert not np.allclose(V_ZZ_new, V_ZZ_old)


# ---------------------------------------------------------------------------
# Vectorised implementation
# ---------------------------------------------------------------------------

def _make_strain(shape, seed=0):
    rng = np.random.default_rng(seed)
    xx = rng.uniform(-0.02,  0.02, shape)
    xz = rng.uniform(-0.01,  0.01, shape)
    zz = rng.uniform(-0.02,  0.02, shape)
    return xx, xz, zz




class TestVectorisedEFG:
    def test_output_shapes_match_scalar(self):
        n, m = 5, 7
        xx, xz, zz = _make_strain((n, m))
        eta_v, vxx_v, vyy_v, vzz_v, euler_v = calculate_efg_vectorised("Ga69", xx, xz, zz)
        assert eta_v.shape   == (n, m)
        assert vxx_v.shape   == (n, m)
        assert vyy_v.shape   == (n, m)
        assert vzz_v.shape   == (n, m)
        assert euler_v.shape == (n, m, 3)

    def test_eigenvalues_match_scalar(self):
        """V_ZZ, V_YY, V_XX from the vectorised version must equal the scalar version."""
        xx, xz, zz = _make_strain((20, 25))
        for species in ["Ga69", "Ga71", "As75", "In115"]:
            _, vxx_s, vyy_s, vzz_s, _ = calculate_efg(species, xx, xz, zz)
            _, vxx_v, vyy_v, vzz_v, _ = calculate_efg_vectorised(species, xx, xz, zz)
            np.testing.assert_allclose(vzz_v, vzz_s, rtol=1e-10,
                                       err_msg=f"{species}: V_ZZ mismatch")
            np.testing.assert_allclose(vyy_v, vyy_s, rtol=1e-10,
                                       err_msg=f"{species}: V_YY mismatch")
            np.testing.assert_allclose(vxx_v, vxx_s, rtol=1e-10,
                                       err_msg=f"{species}: V_XX mismatch")

    def test_eta_matches_scalar(self):
        xx, xz, zz = _make_strain((15, 15))
        eta_s, *_ = calculate_efg("As75", xx, xz, zz)
        eta_v, *_ = calculate_efg_vectorised("As75", xx, xz, zz)
        np.testing.assert_allclose(eta_v, eta_s, rtol=1e-10)

    def test_zero_strain_gives_zero_efg(self):
        n, m = 4, 4
        zeros = np.zeros((n, m))
        eta, V_XX, V_YY, V_ZZ, _ = calculate_efg_vectorised("Ga69", zeros, zeros, zeros)
        np.testing.assert_allclose(V_ZZ, 0.0, atol=1e-25)
        np.testing.assert_allclose(V_XX, 0.0, atol=1e-25)
        np.testing.assert_allclose(V_YY, 0.0, atol=1e-25)

    def test_biaxiality_definition(self):
        xx, xz, zz = _make_strain((6, 6))
        eta, V_XX, V_YY, V_ZZ, _ = calculate_efg_vectorised("In115", xx, xz, zz)
        np.testing.assert_allclose(eta, (V_XX - V_YY) / V_ZZ, atol=1e-10)

    def test_sundfors_gives_different_result(self):
        xx, xz, zz = _make_strain((3, 3))
        _, _, _, vzz_new, _ = calculate_efg_vectorised("Ga69", xx, xz, zz, use_sundfors=False)
        _, _, _, vzz_old, _ = calculate_efg_vectorised("Ga69", xx, xz, zz, use_sundfors=True)
        assert not np.allclose(vzz_new, vzz_old)

    def test_euler_angles_self_consistent(self):
        """
        The vectorised Euler extraction must be consistent with the scalar
        euler_angles_from_rot_mat when both operate on the same rotation matrix.

        The test rebuilds the batch rotation matrices from eigh exactly as
        calculate_efg_vectorised does, then applies euler_angles_from_rot_mat
        site by site and compares to the vectorised output.

        Note on det(rot_mat): np.linalg.eig and eigh return eigenvectors with
        arbitrary signs, so the assembled rotation matrix can have det = ±1.
        This is a pre-existing property of the scalar implementation.  The test
        verifies vectorised consistency, not the sign convention.
        """
        from qdot.efg import euler_angles_from_rot_mat
        from qdot.isotopes import species_dict as sd

        n, m = 8, 8
        xx, xz, zz = _make_strain((n, m))

        S11 = sd["Ga69"]["S11"]
        S12 = -S11 / 2
        S44 = sd["Ga69"]["S44"]

        # Rebuild exactly what calculate_efg_vectorised builds internally.
        V = np.zeros((n, m, 3, 3))
        V[:, :, 0, 0] = S12 * (zz - xx)
        V[:, :, 1, 1] = (S12 + S11) * xx + S12 * zz
        V[:, :, 2, 2] = 2 * S12 * xx + S11 * zz
        V[:, :, 0, 2] = S44 * xz
        V[:, :, 2, 0] = S44 * xz
        V[:, :, 1, 2] = S44 * xz
        V[:, :, 2, 1] = S44 * xz

        w, v = np.linalg.eigh(V)
        sort_idx = np.argsort(np.abs(w), axis=-1)[:, :, ::-1]
        sort_idx_v = np.broadcast_to(sort_idx[:, :, np.newaxis, :], (n, m, 3, 3))
        v_sorted = np.take_along_axis(v, sort_idx_v, axis=-1)
        rot = np.stack([v_sorted[:, :, :, 2],
                        v_sorted[:, :, :, 1],
                        v_sorted[:, :, :, 0]], axis=-1)

        _, _, _, _, euler_v = calculate_efg_vectorised("Ga69", xx, xz, zz)

        for i in range(n):
            for j in range(m):
                expected = euler_angles_from_rot_mat(rot[i, j])
                np.testing.assert_allclose(
                    euler_v[i, j], expected, atol=1e-10,
                    err_msg=f"Euler angle mismatch at site ({i},{j})"
                )
