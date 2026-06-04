import numpy as np
import pytest
from qdot.efg import euler_angles_from_rot_mat, calculate_efg


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
