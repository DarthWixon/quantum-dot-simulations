import pathlib

import numpy as np
import pytest

from qdot.io import load_efg, load_strain_data, save_efg


def _write_strain_files(directory: pathlib.Path, xx, xy, yy) -> None:
    """Write three epsilon text files to directory."""
    np.savetxt(directory / "full_epsilon_xx.txt", xx)
    np.savetxt(directory / "full_epsilon_xy.txt", xy)
    np.savetxt(directory / "full_epsilon_yy.txt", yy)


@pytest.fixture
def strain_dir(tmp_path):
    """10×10 synthetic strain arrays written to a temp directory."""
    rng = np.random.default_rng(0)
    xx = rng.uniform(-0.02, 0.02, (10, 10))
    xy = rng.uniform(-0.01, 0.01, (10, 10))
    yy = rng.uniform(-0.02, 0.02, (10, 10))
    _write_strain_files(tmp_path, xx, xy, yy)
    return tmp_path, xx, xy, yy


class TestLoadStrainData:
    def test_output_shapes(self, strain_dir):
        d, *_ = strain_dir
        # crop [left=2, right=8, top=1, bottom=7] → 6 rows × 6 cols
        xx, xz, zz = load_strain_data(d, [2, 8, 1, 7])
        assert xx.shape == (6, 6)
        assert xz.shape == (6, 6)
        assert zz.shape == (6, 6)

    def test_step_size_reduces_shape(self, strain_dir):
        d, *_ = strain_dir
        xx, xz, zz = load_strain_data(d, [0, 10, 0, 10], step_size=2)
        assert xx.shape == (5, 5)

    def test_xx_values_match_source(self, strain_dir):
        d, src_xx, src_xy, src_yy = strain_dir
        xx, xz, zz = load_strain_data(d, [0, 10, 0, 10])
        np.testing.assert_allclose(xx, src_xx)

    def test_xz_is_sign_flipped_xy(self, strain_dir):
        # xz = -full_xy by convention
        d, src_xx, src_xy, src_yy = strain_dir
        xx, xz, zz = load_strain_data(d, [0, 10, 0, 10])
        np.testing.assert_allclose(xz, -src_xy)

    def test_zz_is_sign_flipped_yy(self, strain_dir):
        # zz = -full_yy by convention
        d, src_xx, src_xy, src_yy = strain_dir
        xx, xz, zz = load_strain_data(d, [0, 10, 0, 10])
        np.testing.assert_allclose(zz, -src_yy)

    def test_region_crop_selects_correct_rows_and_cols(self, strain_dir):
        d, src_xx, src_xy, src_yy = strain_dir
        left, right, top, bottom = 1, 5, 2, 6
        xx, xz, zz = load_strain_data(d, [left, right, top, bottom])
        np.testing.assert_allclose(xx, src_xx[top:bottom, left:right])

    def test_accepts_str_path(self, strain_dir):
        d, *_ = strain_dir
        xx, xz, zz = load_strain_data(str(d), [0, 10, 0, 10])
        assert xx.shape == (10, 10)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            load_strain_data(tmp_path, [0, 5, 0, 5])


# ---------------------------------------------------------------------------
# EFG save / load
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(1)
BOUNDS = [10, 20, 5, 10]
SHAPE = (5, 10)  # (bottom-top, right-left)


def _make_efg_arrays(shape=SHAPE):
    n, m = shape
    eta = RNG.uniform(0.0, 1.0, (n, m))
    V_XX = RNG.uniform(-1e14, 1e14, (n, m))
    V_YY = RNG.uniform(-1e14, 1e14, (n, m))
    V_ZZ = RNG.uniform(-1e14, 1e14, (n, m))
    euler_angles = RNG.uniform(0, 2 * np.pi, (n, m, 3))
    return eta, V_XX, V_YY, V_ZZ, euler_angles


class TestSaveEfg:
    def test_creates_npz_file(self, tmp_path):
        save_efg(tmp_path, "Ga69", BOUNDS, 1, *_make_efg_arrays())
        npz_files = list(tmp_path.glob("*.npz"))
        assert len(npz_files) == 1

    def test_filename_contains_species_and_bounds(self, tmp_path):
        save_efg(tmp_path, "In115", BOUNDS, 2, *_make_efg_arrays())
        name = list(tmp_path.glob("*.npz"))[0].name
        assert "In115" in name
        assert str(BOUNDS) in name
        assert "step_size_2" in name

    def test_sundfors_flag_changes_filename(self, tmp_path):
        arrays = _make_efg_arrays()
        save_efg(tmp_path, "Ga69", BOUNDS, 1, *arrays, use_sundfors=False)
        save_efg(tmp_path, "Ga69", BOUNDS, 1, *arrays, use_sundfors=True)
        names = {f.name for f in tmp_path.glob("*.npz")}
        assert len(names) == 2

    def test_mirrored_strain_flag_changes_filename(self, tmp_path):
        arrays = _make_efg_arrays()
        save_efg(tmp_path, "Ga69", BOUNDS, 1, *arrays, real_strain=True)
        save_efg(tmp_path, "Ga69", BOUNDS, 1, *arrays, real_strain=False)
        names = {f.name for f in tmp_path.glob("*.npz")}
        assert len(names) == 2

    def test_skips_silently_if_file_exists(self, tmp_path):
        arrays = _make_efg_arrays()
        save_efg(tmp_path, "Ga69", BOUNDS, 1, *arrays)
        path = list(tmp_path.glob("*.npz"))[0]
        mtime_before = path.stat().st_mtime

        # Second call with different data — file must not be overwritten.
        save_efg(tmp_path, "Ga69", BOUNDS, 1, *_make_efg_arrays())
        assert path.stat().st_mtime == mtime_before


class TestLoadEfg:
    def test_round_trip_values(self, tmp_path):
        eta, V_XX, V_YY, V_ZZ, euler_angles = _make_efg_arrays()
        save_efg(tmp_path, "Ga69", BOUNDS, 1, eta, V_XX, V_YY, V_ZZ, euler_angles)
        eta_l, vxx_l, vyy_l, vzz_l, euler_l = load_efg(tmp_path, "Ga69", BOUNDS, 1)
        np.testing.assert_array_equal(eta_l, eta)
        np.testing.assert_array_equal(vxx_l, V_XX)
        np.testing.assert_array_equal(vyy_l, V_YY)
        np.testing.assert_array_equal(vzz_l, V_ZZ)

    def test_euler_angles_reshaped_to_n_sites_by_3(self, tmp_path):
        # save_efg accepts (n, m, 3); load_efg always returns (n*m, 3)
        eta, V_XX, V_YY, V_ZZ, euler_angles = _make_efg_arrays()
        save_efg(tmp_path, "Ga69", BOUNDS, 1, eta, V_XX, V_YY, V_ZZ, euler_angles)
        _, _, _, _, euler_l = load_efg(tmp_path, "Ga69", BOUNDS, 1)
        n, m = SHAPE
        assert euler_l.shape == (n * m, 3)

    def test_euler_angles_values_preserved(self, tmp_path):
        eta, V_XX, V_YY, V_ZZ, euler_angles = _make_efg_arrays()
        save_efg(tmp_path, "Ga69", BOUNDS, 1, eta, V_XX, V_YY, V_ZZ, euler_angles)
        _, _, _, _, euler_l = load_efg(tmp_path, "Ga69", BOUNDS, 1)
        np.testing.assert_array_equal(euler_l, euler_angles.reshape(-1, 3))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_efg(tmp_path, "Ga69", BOUNDS, 1)

    def test_accepts_str_path(self, tmp_path):
        save_efg(tmp_path, "As75", BOUNDS, 1, *_make_efg_arrays())
        eta, *_ = load_efg(str(tmp_path), "As75", BOUNDS, 1)
        assert eta.shape == SHAPE
