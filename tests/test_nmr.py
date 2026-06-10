import numpy as np
import pytest
from qdot.io import save_efg
from qdot.nmr import absorption_spectrum, varied_field_spectra

# RF frequencies spanning the Ga69 Zeeman transition at 1 T (~10.22 MHz).
# Dense enough that at least one point lands within the 10 kHz Lorentzian
# linewidth of a transition, giving non-trivial rates.
_RF_FREQS = np.linspace(8e6, 13e6, 20)

_BOUNDS = [0, 2, 0, 2]
_SHAPE = (2, 2)
_SPECIES = "Ga69"
_LOCATION = (0, 0)


@pytest.fixture
def efg_dir(tmp_path):
    """
    2×2 synthetic EFG archive for Ga69.

    V_ZZ is in the range 1e20–5e20 V/m² so the quadrupolar term (~1 MHz) is
    a meaningful fraction of the Zeeman term (~10 MHz at 1 T). This ensures
    the Faraday and Voigt Hamiltonians differ visibly and the NMR spectrum
    has non-trivial structure.
    """
    rng = np.random.default_rng(7)
    eta = rng.uniform(0.3, 0.7, _SHAPE)
    V_XX = rng.uniform(-5e20, 5e20, _SHAPE)
    V_YY = rng.uniform(-5e20, 5e20, _SHAPE)
    V_ZZ = rng.uniform(1e20, 5e20, _SHAPE)
    euler_angles = rng.uniform(0, 2 * np.pi, (*_SHAPE, 3))
    save_efg(tmp_path, _SPECIES, _BOUNDS, 1, eta, V_XX, V_YY, V_ZZ, euler_angles)
    return tmp_path


# ---------------------------------------------------------------------------
# absorption_spectrum
# ---------------------------------------------------------------------------


class TestAbsorptionSpectrum:
    def test_invalid_species_raises(self, efg_dir):
        with pytest.raises(ValueError, match="nuclear_species"):
            absorption_spectrum(
                "Xx99", 1.0, "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
            )

    def test_invalid_geometry_raises(self, efg_dir):
        with pytest.raises(ValueError, match="field_geometry"):
            absorption_spectrum(
                _SPECIES, 1.0, "Diagonal", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
            )

    def test_output_shape(self, efg_dir):
        result = absorption_spectrum(
            _SPECIES, 1.0, "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        assert result.shape == (_RF_FREQS.shape[0],)

    def test_output_is_nonneg(self, efg_dir):
        result = absorption_spectrum(
            _SPECIES, 1.0, "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        assert np.all(result >= 0)

    def test_output_is_finite(self, efg_dir):
        result = absorption_spectrum(
            _SPECIES, 1.0, "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        assert np.all(np.isfinite(result))

    def test_faraday_and_voigt_differ(self, efg_dir):
        # With a ~1 MHz quadrupolar term the two geometries have different
        # eigenvectors and peak positions, so the spectra are not equal.
        faraday = absorption_spectrum(
            _SPECIES, 1.0, "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        voigt = absorption_spectrum(
            _SPECIES, 1.0, "Voigt", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        assert not np.array_equal(faraday, voigt)

    def test_accepts_str_path(self, efg_dir):
        result = absorption_spectrum(
            _SPECIES, 1.0, "Faraday", _RF_FREQS, _LOCATION, str(efg_dir), _BOUNDS
        )
        assert result.shape == (_RF_FREQS.shape[0],)


# ---------------------------------------------------------------------------
# varied_field_spectra
# ---------------------------------------------------------------------------

_FIELDS = [0.5, 1.0, 2.0]


class TestVariedFieldSpectra:
    def test_returns_list_of_correct_length(self, efg_dir):
        result = varied_field_spectra(
            _SPECIES, _FIELDS, "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        assert len(result) == len(_FIELDS)

    def test_each_entry_has_required_keys(self, efg_dir):
        result = varied_field_spectra(
            _SPECIES, _FIELDS, "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        for entry in result:
            assert {"applied_field", "rf_freq_list", "data"} <= entry.keys()

    def test_data_shape_per_entry(self, efg_dir):
        result = varied_field_spectra(
            _SPECIES, _FIELDS, "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        for entry in result:
            assert entry["data"].shape == (_RF_FREQS.shape[0],)

    def test_applied_field_values_match_input(self, efg_dir):
        result = varied_field_spectra(
            _SPECIES, _FIELDS, "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        for entry, B in zip(result, _FIELDS):
            assert entry["applied_field"] == B

    def test_different_fields_give_different_spectra(self, efg_dir):
        # 1.0 T has its Zeeman transition within _RF_FREQS (8-13 MHz);
        # 2.0 T peaks at ~20 MHz, outside the range → near-zero rates.
        result = varied_field_spectra(
            _SPECIES, [1.0, 2.0], "Faraday", _RF_FREQS, _LOCATION, efg_dir, _BOUNDS
        )
        assert not np.array_equal(result[0]["data"], result[1]["data"])


# ---------------------------------------------------------------------------
# Multi-site engine: absorption_map / summed_absorption_spectrum
# ---------------------------------------------------------------------------

_LOCS = [(0, 0), (0, 1), (1, 1)]
_SMALL_FIELDS = np.array([0.5, 1.0])


class TestAbsorptionMap:
    def test_shape(self, efg_dir):
        from qdot.nmr import absorption_map

        result = absorption_map(
            _SPECIES,
            _SMALL_FIELDS,
            "Faraday",
            _RF_FREQS,
            _LOCS,
            efg_dir,
            _BOUNDS,
            processes=2,
        )
        assert result.shape == (len(_SMALL_FIELDS), len(_RF_FREQS))

    def test_matches_sum_of_single_site_spectra(self, efg_dir):
        from qdot.nmr import summed_absorption_spectrum

        summed = summed_absorption_spectrum(
            _SPECIES, 1.0, "Faraday", _RF_FREQS, _LOCS, efg_dir, _BOUNDS, processes=2
        )
        expected = np.zeros(len(_RF_FREQS))
        for loc in _LOCS:
            expected += absorption_spectrum(
                _SPECIES, 1.0, "Faraday", _RF_FREQS, loc, efg_dir, _BOUNDS
            )
        np.testing.assert_allclose(summed, expected, rtol=1e-12)

    def test_empty_locations_gives_zeros(self, efg_dir):
        from qdot.nmr import absorption_map

        result = absorption_map(
            _SPECIES, _SMALL_FIELDS, "Faraday", _RF_FREQS, [], efg_dir, _BOUNDS
        )
        assert np.all(result == 0)

    def test_invalid_species_raises(self, efg_dir):
        from qdot.nmr import absorption_map

        with pytest.raises(ValueError, match="nuclear_species"):
            absorption_map(
                "Fe56", _SMALL_FIELDS, "Faraday", _RF_FREQS, _LOCS, efg_dir, _BOUNDS
            )

    def test_nonnegative(self, efg_dir):
        from qdot.nmr import absorption_map

        result = absorption_map(
            _SPECIES,
            _SMALL_FIELDS,
            "Faraday",
            _RF_FREQS,
            _LOCS,
            efg_dir,
            _BOUNDS,
            processes=2,
        )
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# Combined spectrum and experimental simulation
# ---------------------------------------------------------------------------


@pytest.fixture
def all_species_dir(tmp_path):
    """EFG archives for all four species plus a concentration file."""
    rng = np.random.default_rng(11)
    for species in ("Ga69", "Ga71", "As75", "In115"):
        eta = rng.uniform(0.3, 0.7, _SHAPE)
        V_XX = rng.uniform(-5e20, 5e20, _SHAPE)
        V_YY = rng.uniform(-5e20, 5e20, _SHAPE)
        V_ZZ = rng.uniform(1e20, 5e20, _SHAPE)
        euler = rng.uniform(0, 2 * np.pi, (*_SHAPE, 3))
        save_efg(tmp_path, species, _BOUNDS, 1, eta, V_XX, V_YY, V_ZZ, euler)
    # Concentration file covering the bounds; mean ~0.25.
    conc = rng.uniform(0.2, 0.3, (4, 4))
    np.save(tmp_path / "conc_data_to_scale_cubic_interpolation.npy", conc)
    return tmp_path


class TestCombinedSpectrum:
    def test_normalised_to_one(self, all_species_dir):
        from qdot.nmr import combined_spectrum

        result = combined_spectrum(
            1.0, "Faraday", _RF_FREQS, _LOCS, all_species_dir, _BOUNDS, processes=2
        )
        assert result.shape == (len(_RF_FREQS),)
        assert np.isclose(np.max(result), 1.0)
        assert np.all(result >= 0)


class TestExperimentalNmrSimulation:
    def test_returns_three_species_maps(self, all_species_dir):
        from qdot.nmr import experimental_nmr_simulation

        maps = experimental_nmr_simulation(
            "Faraday",
            _SMALL_FIELDS,
            _RF_FREQS,
            n_locations=4,
            data_dir=all_species_dir,
            region_bounds=_BOUNDS,
            rng=5,
            processes=2,
        )
        assert set(maps) == {"In115", "Ga69", "As75"}
        for data in maps.values():
            assert data.shape == (len(_SMALL_FIELDS), len(_RF_FREQS))
            assert np.all(data >= 0)

    def test_too_few_locations_raises(self, all_species_dir):
        from qdot.nmr import experimental_nmr_simulation

        with pytest.raises(ValueError, match="too small"):
            experimental_nmr_simulation(
                "Faraday",
                _SMALL_FIELDS,
                _RF_FREQS,
                n_locations=1,
                data_dir=all_species_dir,
                region_bounds=_BOUNDS,
            )


# ---------------------------------------------------------------------------
# RF pulse selectivity
# ---------------------------------------------------------------------------


class TestLorentzianPulse:
    def test_peak_at_centre(self):
        from qdot.nmr import lorentzian_pulse

        freqs = np.linspace(0, 100, 1001)
        pulse = lorentzian_pulse(freqs, centre=40.0, width=5.0)
        assert freqs[np.argmax(pulse)] == pytest.approx(40.0, abs=0.1)

    def test_unit_area(self):
        from qdot.nmr import lorentzian_pulse
        from scipy.integrate import simpson

        # Wide grid so the Cauchy tails are mostly captured.
        freqs = np.linspace(-1000, 1000, 200001)
        pulse = lorentzian_pulse(freqs, centre=0.0, width=2.0)
        assert simpson(pulse, x=freqs) == pytest.approx(1.0, abs=1e-2)


class TestPulseCaptureFraction:
    def test_wide_pulse_captures_everything(self):
        from qdot.nmr import pulse_capture_fraction

        freqs = np.linspace(0, 100, 2001)
        spectrum = np.exp(-((freqs - 50) ** 2) / 4)
        frac = pulse_capture_fraction(spectrum, freqs, 50.0, 1e6)
        assert frac == pytest.approx(1.0, abs=1e-3)

    def test_offresonant_narrow_pulse_captures_little(self):
        from qdot.nmr import pulse_capture_fraction

        freqs = np.linspace(0, 100, 2001)
        spectrum = np.exp(-((freqs - 50) ** 2) / 4)
        frac = pulse_capture_fraction(spectrum, freqs, 5.0, 0.5)
        assert frac < 0.05

    def test_window_restricts_integral(self):
        from qdot.nmr import pulse_capture_fraction

        freqs = np.linspace(0, 100, 2001)
        # Two identical peaks; pulse on the first one.
        spectrum = np.exp(-((freqs - 30) ** 2) / 4) + np.exp(-((freqs - 70) ** 2) / 4)
        in_band = pulse_capture_fraction(
            spectrum, freqs, 30.0, 1e6, window=(20.0, 40.0)
        )
        overall = pulse_capture_fraction(spectrum, freqs, 30.0, 20.0)
        assert in_band == pytest.approx(1.0, abs=1e-3)
        assert overall < in_band
