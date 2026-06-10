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


# ---------------------------------------------------------------------------
# Region catalogue and rectangle conversion
# ---------------------------------------------------------------------------


def test_sokolov_regions_contains_dot_region():
    from qdot.io import SOKOLOV_REGIONS, SOKOLOV_DOT_REGION

    assert SOKOLOV_REGIONS["entire_dot"] == SOKOLOV_DOT_REGION
    for bounds in SOKOLOV_REGIONS.values():
        left, right, top, bottom = bounds
        assert left < right and top < bottom


def test_region_from_rectangle():
    from qdot.io import region_from_rectangle

    result = region_from_rectangle([100, 1200, 200, 1000], [50, 30, 200, 100])
    assert result == [150, 350, 230, 330]


# ---------------------------------------------------------------------------
# Mirrored datasets
# ---------------------------------------------------------------------------


class TestMirrorArray:
    def test_left_right_keeps_left_half(self):
        from qdot.io import mirror_array

        data = np.arange(12.0).reshape(3, 4)
        result = mirror_array(data, "left_right")
        np.testing.assert_array_equal(result[:, :2], data[:, :2])
        np.testing.assert_array_equal(result, np.fliplr(result))

    def test_right_left_keeps_right_half(self):
        from qdot.io import mirror_array

        data = np.arange(12.0).reshape(3, 4)
        result = mirror_array(data, "right_left")
        np.testing.assert_array_equal(result[:, 2:], data[:, 2:])
        np.testing.assert_array_equal(result, np.fliplr(result))

    def test_odd_columns_drop_centre(self):
        from qdot.io import mirror_array

        data = np.arange(15.0).reshape(3, 5)
        assert mirror_array(data, "left_right").shape == (3, 4)

    def test_invalid_direction_raises(self):
        from qdot.io import mirror_array

        with pytest.raises(ValueError, match="direction"):
            mirror_array(np.zeros((2, 2)), "top_bottom")


class TestMirroredDatasets:
    def test_strain_round_trip(self, strain_dir):
        from qdot.io import (
            create_mirrored_strain_data,
            load_mirrored_data,
            load_strain_data,
            mirror_array,
        )

        d, *_ = strain_dir
        bounds = [0, 8, 0, 6]
        create_mirrored_strain_data(d, bounds, "left_right")

        xx_m, xz_m, zz_m = load_mirrored_data(d, bounds, mirror_type="left_right")
        xx, xz, zz = load_strain_data(d, bounds)
        np.testing.assert_allclose(xx_m, mirror_array(xx, "left_right"))
        np.testing.assert_allclose(xz_m, mirror_array(xz, "left_right"))
        np.testing.assert_allclose(zz_m, mirror_array(zz, "left_right"))

    def test_strain_skips_existing(self, strain_dir):
        from qdot.io import create_mirrored_strain_data, load_mirrored_data

        d, *_ = strain_dir
        bounds = [0, 8, 0, 6]
        create_mirrored_strain_data(d, bounds)
        first, *_ = load_mirrored_data(d, bounds)

        # Overwrite the source files; without overwrite=True the archive
        # must stay unchanged.
        rng = np.random.default_rng(99)
        _write_strain_files(
            d,
            rng.random((10, 10)),
            rng.random((10, 10)),
            rng.random((10, 10)),
        )
        create_mirrored_strain_data(d, bounds)
        unchanged, *_ = load_mirrored_data(d, bounds)
        np.testing.assert_array_equal(first, unchanged)

        create_mirrored_strain_data(d, bounds, overwrite=True)
        replaced, *_ = load_mirrored_data(d, bounds)
        assert not np.array_equal(first, replaced)

    def test_concentration_round_trip(self, tmp_path):
        from qdot.io import (
            create_mirrored_concentration_data,
            load_mirrored_concentration_data,
            mirror_array,
        )

        rng = np.random.default_rng(3)
        conc = rng.uniform(0, 0.5, (10, 10))
        np.save(tmp_path / "conc_data_to_scale_cubic_interpolation.npy", conc)

        bounds = [0, 8, 0, 6]
        create_mirrored_concentration_data(tmp_path, bounds, "right_left")
        result = load_mirrored_concentration_data(tmp_path, bounds, "right_left")
        np.testing.assert_allclose(result, mirror_array(conc[0:6, 0:8], "right_left"))


# ---------------------------------------------------------------------------
# NMR map archives
# ---------------------------------------------------------------------------


class TestNmrMapArchive:
    _FIELDS = np.linspace(0.1, 2.0, 4)
    _FREQS = np.linspace(1e6, 50e6, 6)
    _LOCS = [(0, 0), (1, 1)]
    _BOUNDS = [0, 2, 0, 2]

    def _save(self, d, data, **kwargs):
        from qdot.io import save_nmr_map

        save_nmr_map(
            d,
            "Ga69",
            "Faraday",
            self._BOUNDS,
            data,
            self._FIELDS,
            self._FREQS,
            self._LOCS,
            **kwargs,
        )

    def test_round_trip(self, tmp_path):
        from qdot.io import load_nmr_map

        data = np.random.default_rng(0).random((4, 6))
        self._save(tmp_path, data)

        loaded, fields, freqs, locs = load_nmr_map(
            tmp_path, "Ga69", "Faraday", 2, self._BOUNDS, self._FIELDS, self._FREQS
        )
        np.testing.assert_allclose(loaded, data)
        np.testing.assert_allclose(fields, self._FIELDS)
        np.testing.assert_allclose(freqs, self._FREQS)
        np.testing.assert_array_equal(locs, self._LOCS)

    def test_skips_existing_unless_overwrite(self, tmp_path):
        from qdot.io import load_nmr_map

        first = np.ones((4, 6))
        second = np.full((4, 6), 2.0)
        self._save(tmp_path, first)
        self._save(tmp_path, second)
        loaded, *_ = load_nmr_map(
            tmp_path, "Ga69", "Faraday", 2, self._BOUNDS, self._FIELDS, self._FREQS
        )
        np.testing.assert_allclose(loaded, first)

        self._save(tmp_path, second, overwrite=True)
        loaded, *_ = load_nmr_map(
            tmp_path, "Ga69", "Faraday", 2, self._BOUNDS, self._FIELDS, self._FREQS
        )
        np.testing.assert_allclose(loaded, second)

    def test_sundfors_variant_has_separate_file(self, tmp_path):
        from qdot.io import load_nmr_map

        self._save(tmp_path, np.ones((4, 6)))
        self._save(tmp_path, np.full((4, 6), 3.0), use_sundfors=True)

        standard, *_ = load_nmr_map(
            tmp_path, "Ga69", "Faraday", 2, self._BOUNDS, self._FIELDS, self._FREQS
        )
        sundfors, *_ = load_nmr_map(
            tmp_path,
            "Ga69",
            "Faraday",
            2,
            self._BOUNDS,
            self._FIELDS,
            self._FREQS,
            use_sundfors=True,
        )
        np.testing.assert_allclose(standard, np.ones((4, 6)))
        np.testing.assert_allclose(sundfors, np.full((4, 6), 3.0))
