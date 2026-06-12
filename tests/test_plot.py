"""
Smoke tests for qdot.plot: every function renders synthetic data and saves a
non-empty file. Figure content is not asserted — these catch API breakage,
exceptions, and backend issues only.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from qdot import plot

_FIELDS = np.linspace(0.1, 2.0, 4)
_FREQS = np.linspace(1e6, 50e6, 8)


def _saved(tmp_path, name):
    path = tmp_path / name
    assert path.exists() and path.stat().st_size > 0
    return path


@pytest.fixture
def nmr_map():
    rng = np.random.default_rng(0)
    return rng.uniform(0.1, 1.0, (len(_FIELDS), len(_FREQS)))


class TestNmrMapPlots:
    def test_plot_nmr_map(self, tmp_path, nmr_map):
        plot.plot_nmr_map(
            nmr_map, _FIELDS, _FREQS, title="t", save_path=tmp_path / "m.png"
        )
        _saved(tmp_path, "m.png")

    def test_plot_nmr_map_linear_scale(self, tmp_path, nmr_map):
        plot.plot_nmr_map(
            nmr_map, _FIELDS, _FREQS, log_scale=False, save_path=tmp_path / "m.png"
        )
        _saved(tmp_path, "m.png")

    def test_plot_nmr_map_pair(self, tmp_path, nmr_map):
        plot.plot_nmr_map_pair(
            nmr_map,
            nmr_map * 2,
            ("Faraday", "Voigt"),
            _FIELDS,
            _FREQS,
            save_path=tmp_path / "p.png",
        )
        _saved(tmp_path, "p.png")

    def test_plot_nmr_map_difference(self, tmp_path, nmr_map):
        plot.plot_nmr_map_difference(
            nmr_map, nmr_map * 2, _FIELDS, _FREQS, save_path=tmp_path / "d.png"
        )
        _saved(tmp_path, "d.png")

    def test_plot_nmr_map_layered(self, tmp_path, nmr_map):
        plot.plot_nmr_map_layered(
            {"In115": nmr_map, "Ga69": nmr_map * 2, "As75": nmr_map * 3},
            _FIELDS,
            _FREQS,
            save_path=tmp_path / "l.png",
        )
        _saved(tmp_path, "l.png")


class TestEnergyLevelPlots:
    def test_plot_energy_levels(self, tmp_path):
        sweep = np.linspace(0, 3, 10)
        levels = np.outer([-1.5, -0.5, 0.5, 1.5], sweep) * 1e7
        plot.plot_energy_levels(
            sweep, levels, "Applied B Field (T)", save_path=tmp_path / "el.png"
        )
        _saved(tmp_path, "el.png")

    def test_plot_energy_levels_comparison(self, tmp_path):
        sweep = np.linspace(0, 3, 10)
        levels = np.outer([-1.5, -0.5, 0.5, 1.5], sweep) * 1e7
        plot.plot_energy_levels_comparison(
            sweep, levels, levels * 0.8, save_path=tmp_path / "elc.png"
        )
        _saved(tmp_path, "elc.png")


class TestSiteMapPlots:
    def test_plot_site_map(self, tmp_path):
        plot.plot_site_map(
            np.random.default_rng(0).random((5, 5)),
            colorbar_label="x",
            title="t",
            save_path=tmp_path / "s.png",
        )
        _saved(tmp_path, "s.png")

    def test_plot_biaxiality(self, tmp_path):
        plot.plot_biaxiality(
            np.random.default_rng(0).random((5, 5)), save_path=tmp_path / "b.png"
        )
        _saved(tmp_path, "b.png")

    def test_plot_quadrupole_frequency(self, tmp_path):
        plot.plot_quadrupole_frequency(
            np.random.default_rng(0).random((5, 5)) * 1e6,
            save_path=tmp_path / "q.png",
        )
        _saved(tmp_path, "q.png")

    def test_plot_efg_directions(self, tmp_path):
        rng = np.random.default_rng(0)
        euler = rng.uniform(0, np.pi, (6, 6, 3))
        background = rng.random((6, 6))
        plot.plot_efg_directions(
            euler,
            background,
            background_label="η",
            spacing=2,
            save_path=tmp_path / "e.png",
        )
        _saved(tmp_path, "e.png")

    def test_plot_efg_directions_with_lengths(self, tmp_path):
        rng = np.random.default_rng(0)
        plot.plot_efg_directions(
            rng.uniform(0, np.pi, (6, 6, 3)),
            rng.random((6, 6)),
            arrow_lengths=rng.random((6, 6)),
            spacing=2,
            double_headed=False,
            save_path=tmp_path / "e2.png",
        )
        _saved(tmp_path, "e2.png")


class TestHistogramPlots:
    @pytest.mark.parametrize("fit", [None, "gauss", "maxwell", "gamma"])
    def test_plot_frequency_histogram_fits(self, tmp_path, fit):
        rng = np.random.default_rng(0)
        freqs = rng.normal(0, 0.3, 2000) * 1e6
        plot.plot_frequency_histogram(
            freqs, fit=fit, label="Ga69", save_path=tmp_path / "h.png"
        )
        _saved(tmp_path, "h.png")

    def test_invalid_fit_raises(self, tmp_path):
        with pytest.raises(ValueError, match="fit"):
            plot.plot_frequency_histogram(
                np.random.default_rng(0).normal(0, 0.3, 500) * 1e6,
                fit="lorentz",
                save_path=tmp_path / "h.png",
            )

    def test_plot_frequency_histograms_overlay(self, tmp_path):
        rng = np.random.default_rng(0)
        sets = {s: rng.normal(0, 0.3, 500) * 1e6 for s in ("Ga69", "As75")}
        plot.plot_frequency_histograms(sets, save_path=tmp_path / "ho.png")
        _saved(tmp_path, "ho.png")

    def test_plot_frequency_histograms_grid(self, tmp_path):
        rng = np.random.default_rng(0)
        sets = {
            s: rng.normal(0, 0.3, 800) * 1e6 for s in ("Ga69", "Ga71", "As75", "In115")
        }
        plot.plot_frequency_histograms_grid(
            sets, fit="gamma", save_path=tmp_path / "hg.png"
        )
        _saved(tmp_path, "hg.png")


class TestStrainPlots:
    @pytest.mark.parametrize("layout", ["horizontal", "vertical"])
    def test_plot_measured_strain(self, tmp_path, layout):
        rng = np.random.default_rng(0)
        xx, xz, zz = (rng.uniform(-0.02, 0.02, (8, 8)) for _ in range(3))
        plot.plot_measured_strain(
            xx, xz, zz, layout=layout, save_path=tmp_path / "ms.png"
        )
        _saved(tmp_path, "ms.png")

    def test_invalid_layout_raises(self):
        arr = np.zeros((4, 4))
        with pytest.raises(ValueError, match="layout"):
            plot.plot_measured_strain(arr, arr, arr, layout="diagonal")

    def test_plot_strain_row_cut(self, tmp_path):
        rng = np.random.default_rng(0)
        xx, xz, zz = (rng.uniform(-0.02, 0.02, (8, 8)) for _ in range(3))
        plot.plot_strain_row_cut(xx, xz, zz, row=3, save_path=tmp_path / "rc.png")
        _saved(tmp_path, "rc.png")


class TestCurvePlots:
    def test_plot_polarisation_curve(self, tmp_path):
        deph = np.linspace(1, 0.5, 20)
        plot.plot_polarisation_curve(
            deph, np.cos(deph), reference=0.9, save_path=tmp_path / "pc.png"
        )
        _saved(tmp_path, "pc.png")

    def test_plot_fidelity_curves(self, tmp_path):
        strengths = np.linspace(0, 1, 10)
        curves = {"1 photon": 1 - 0.2 * strengths, "2 photons": 1 - 0.4 * strengths}
        plot.plot_fidelity_curves(strengths, curves, save_path=tmp_path / "f.png")
        _saved(tmp_path, "f.png")
