import pathlib

import numpy as np
import pytest

from qdot.io import load_strain_data


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
