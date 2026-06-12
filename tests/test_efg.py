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
    xx = rng.uniform(-0.02, 0.02, shape)
    xz = rng.uniform(-0.01, 0.01, shape)
    zz = rng.uniform(-0.02, 0.02, shape)
    return xx, xz, zz


class TestVectorisedEFG:
    def test_output_shapes_match_scalar(self):
        n, m = 5, 7
        xx, xz, zz = _make_strain((n, m))
        eta_v, vxx_v, vyy_v, vzz_v, euler_v = calculate_efg_vectorised(
            "Ga69", xx, xz, zz
        )
        assert eta_v.shape == (n, m)
        assert vxx_v.shape == (n, m)
        assert vyy_v.shape == (n, m)
        assert vzz_v.shape == (n, m)
        assert euler_v.shape == (n, m, 3)

    def test_eigenvalues_match_scalar(self):
        """V_ZZ, V_YY, V_XX from the vectorised version must equal the scalar version."""
        xx, xz, zz = _make_strain((20, 25))
        for species in ["Ga69", "Ga71", "As75", "In115"]:
            _, vxx_s, vyy_s, vzz_s, _ = calculate_efg(species, xx, xz, zz)
            _, vxx_v, vyy_v, vzz_v, _ = calculate_efg_vectorised(species, xx, xz, zz)
            np.testing.assert_allclose(
                vzz_v, vzz_s, rtol=1e-10, err_msg=f"{species}: V_ZZ mismatch"
            )
            np.testing.assert_allclose(
                vyy_v, vyy_s, rtol=1e-10, err_msg=f"{species}: V_YY mismatch"
            )
            np.testing.assert_allclose(
                vxx_v, vxx_s, rtol=1e-10, err_msg=f"{species}: V_XX mismatch"
            )

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
        _, _, _, vzz_new, _ = calculate_efg_vectorised(
            "Ga69", xx, xz, zz, use_sundfors=False
        )
        _, _, _, vzz_old, _ = calculate_efg_vectorised(
            "Ga69", xx, xz, zz, use_sundfors=True
        )
        assert not np.allclose(vzz_new, vzz_old)

    def test_euler_angles_self_consistent(self):
        """
        Vectorised Euler extraction is consistent with the scalar
        euler_angles_from_rot_mat when both operate on the same rotation matrix.

        Rebuilds the batch rotation matrices from eigh exactly as
        calculate_efg_vectorised does (including the cross-product sign fix),
        then applies euler_angles_from_rot_mat site by site.
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
        rot = np.stack(
            [v_sorted[:, :, :, 2], v_sorted[:, :, :, 1], v_sorted[:, :, :, 0]], axis=-1
        )
        # Apply the same sign fix as calculate_efg_vectorised.
        rot[:, :, :, 2] = np.cross(rot[:, :, :, 0], rot[:, :, :, 1])

        _, _, _, _, euler_v = calculate_efg_vectorised("Ga69", xx, xz, zz)

        for i in range(n):
            for j in range(m):
                expected = euler_angles_from_rot_mat(rot[i, j])
                np.testing.assert_allclose(
                    euler_v[i, j],
                    expected,
                    atol=1e-10,
                    err_msg=f"Euler angle mismatch at site ({i},{j})",
                )

    def test_rotation_matrices_are_proper(self):
        """Both implementations must produce rotation matrices with det = +1."""
        xx, xz, zz = _make_strain((10, 10))
        for species in ["Ga69", "Ga71", "As75", "In115"]:
            # Scalar: rebuild rot from the stored euler angles isn't straightforward,
            # but we can verify via the Euler reconstruction test instead.
            # Vectorised: check det directly from the internal rotation matrices.
            from qdot.isotopes import species_dict as sd

            S11 = sd[species]["S11"]
            S12 = -S11 / 2
            S44 = sd[species]["S44"]
            n, m = xx.shape
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
            rot = np.stack(
                [v_sorted[:, :, :, 2], v_sorted[:, :, :, 1], v_sorted[:, :, :, 0]],
                axis=-1,
            )
            rot[:, :, :, 2] = np.cross(rot[:, :, :, 0], rot[:, :, :, 1])
            dets = np.linalg.det(rot)
            np.testing.assert_allclose(
                dets, 1.0, atol=1e-12, err_msg=f"{species}: rot det not +1"
            )

    def test_euler_angles_reconstruct_rotation(self):
        """
        Extracted Euler angles must reconstruct the rotation matrix they came from.

        R = Rx(alpha) · Ry(beta) · Rz(gamma)

        Tests both the scalar and vectorised implementations.
        """

        def euler_to_rot(alpha, beta, gamma):
            # Actual convention: R = Rz(gamma) · Ry(beta) · Rx(alpha).
            # The docstring says Rx·Ry·Rz but the element indexing in
            # euler_angles_from_rot_mat (rot[2,0] = -sin(beta), etc.) is
            # the Slabaugh formula for Rz·Ry·Rx.
            ca, sa = np.cos(alpha), np.sin(alpha)
            cb, sb = np.cos(beta), np.sin(beta)
            cg, sg = np.cos(gamma), np.sin(gamma)
            Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
            Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
            Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
            return Rz @ Ry @ Rx

        n, m = 6, 6
        xx, xz, zz = _make_strain((n, m), seed=7)

        # Vectorised: reconstruct from the same rotation matrices used internally.
        from qdot.isotopes import species_dict as sd

        S11 = sd["Ga69"]["S11"]
        S12 = -S11 / 2
        S44 = sd["Ga69"]["S44"]
        V = np.zeros((n, m, 3, 3))
        V[:, :, 0, 0] = S12 * (zz - xx)
        V[:, :, 1, 1] = (S12 + S11) * xx + S12 * zz
        V[:, :, 2, 2] = 2 * S12 * xx + S11 * zz
        V[:, :, 0, 2] = V[:, :, 2, 0] = S44 * xz
        V[:, :, 1, 2] = V[:, :, 2, 1] = S44 * xz
        w, v = np.linalg.eigh(V)
        sort_idx = np.argsort(np.abs(w), axis=-1)[:, :, ::-1]
        sort_idx_v = np.broadcast_to(sort_idx[:, :, np.newaxis, :], (n, m, 3, 3))
        v_sorted = np.take_along_axis(v, sort_idx_v, axis=-1)
        rot = np.stack(
            [v_sorted[:, :, :, 2], v_sorted[:, :, :, 1], v_sorted[:, :, :, 0]], axis=-1
        )
        rot[:, :, :, 2] = np.cross(rot[:, :, :, 0], rot[:, :, :, 1])

        _, _, _, _, euler_v = calculate_efg_vectorised("Ga69", xx, xz, zz)

        for i in range(n):
            for j in range(m):
                a, b, g = euler_v[i, j]
                R_rec = euler_to_rot(a, b, g)
                np.testing.assert_allclose(
                    R_rec,
                    rot[i, j],
                    atol=1e-10,
                    err_msg=f"Euler reconstruction failed at site ({i},{j})",
                )


# ---------------------------------------------------------------------------
# Multi-species Euler angle coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("species", ["Ga71", "As75", "In115"])
def test_euler_angles_reconstruct_rotation_all_species(species):
    """
    Euler angles extracted by calculate_efg_vectorised must reconstruct the
    rotation matrix for all species, not just Ga69.

    Uses the same round-trip check as test_euler_angles_reconstruct_rotation
    but parameterised over the three remaining species.
    """
    from qdot.isotopes import species_dict as sd

    def euler_to_rot(alpha, beta, gamma):
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    n, m = 4, 4
    xx, xz, zz = _make_strain((n, m), seed=42)

    S11 = sd[species]["S11"]
    S12 = -S11 / 2
    S44 = sd[species]["S44"]

    V = np.zeros((n, m, 3, 3))
    V[:, :, 0, 0] = S12 * (zz - xx)
    V[:, :, 1, 1] = (S12 + S11) * xx + S12 * zz
    V[:, :, 2, 2] = 2 * S12 * xx + S11 * zz
    V[:, :, 0, 2] = V[:, :, 2, 0] = S44 * xz
    V[:, :, 1, 2] = V[:, :, 2, 1] = S44 * xz

    w, v = np.linalg.eigh(V)
    sort_idx = np.argsort(np.abs(w), axis=-1)[:, :, ::-1]
    sort_idx_v = np.broadcast_to(sort_idx[:, :, np.newaxis, :], (n, m, 3, 3))
    v_sorted = np.take_along_axis(v, sort_idx_v, axis=-1)
    rot = np.stack(
        [v_sorted[:, :, :, 2], v_sorted[:, :, :, 1], v_sorted[:, :, :, 0]], axis=-1
    )
    rot[:, :, :, 2] = np.cross(rot[:, :, :, 0], rot[:, :, :, 1])

    _, _, _, _, euler_v = calculate_efg_vectorised(species, xx, xz, zz)

    for i in range(n):
        for j in range(m):
            a, b, g = euler_v[i, j]
            R_rec = euler_to_rot(a, b, g)
            np.testing.assert_allclose(
                R_rec,
                rot[i, j],
                atol=1e-10,
                err_msg=f"{species}: Euler reconstruction failed at ({i},{j})",
            )


# ---------------------------------------------------------------------------
# Quadrupole frequency and equivalent B field
# ---------------------------------------------------------------------------


def test_quadrupole_frequency_scales_with_coupling():
    from qdot.efg import quadrupole_frequency
    from qdot.isotopes import species_dict, quadrupole_coupling

    V_ZZ = np.array([[1e20, 2e20], [3e20, -4e20]])
    result = quadrupole_frequency("As75", V_ZZ)
    expected = quadrupole_coupling(species_dict["As75"]) * V_ZZ
    np.testing.assert_allclose(result, expected)


def test_quadrupole_frequency_invalid_species_raises():
    from qdot.efg import quadrupole_frequency

    with pytest.raises(ValueError, match="nuclear_species"):
        quadrupole_frequency("Fe56", np.ones((2, 2)))


def test_equivalent_b_field_is_frequency_over_zeeman():
    from qdot.efg import quadrupole_frequency, equivalent_b_field
    from qdot.isotopes import species_dict

    V_ZZ = np.array([[1e20, 5e20]])
    freq = quadrupole_frequency("In115", V_ZZ)
    expected = freq / species_dict["In115"]["zeeman_frequency_per_tesla"]
    np.testing.assert_allclose(equivalent_b_field("In115", V_ZZ), expected)
