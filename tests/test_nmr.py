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
