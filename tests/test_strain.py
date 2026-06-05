"""
Tests for qdot.strain.

Physical invariants verified:
- Pure GaAs has zero strain: all Ga-As bonds have equal natural length (1.0),
  so the uniform grid is already the minimum-energy state.
- Strain tensor is symmetric (ε_ij = ε_ji): enforced by the 0.5*(F + Fᵀ)
  formula, but worth pinning explicitly.
- Identical strained and unstrained coords → zero strain.
- A single In atom introduces strain (In-As natural length is 1.06 vs 1.00).
- More In substitutions produce a larger peak strain than a single atom.
"""

import math
import numpy as np
import pytest

from qdot.strain import (
    block_in_positions,
    gaas_lattice,
    gaas_lattice_with_indium,
    gaas_lattice_with_indium_centre,
    run_strain_simulation,
    strain_tensor,
    unstrained_positions,
)

# ---------------------------------------------------------------------------
# Lattice generators
# ---------------------------------------------------------------------------


class TestLatticeGenerators:
    def test_gaas_lattice_shapes(self):
        large, small = gaas_lattice(5, 7)
        assert small.shape == (5, 7)
        assert large.shape == (7, 9)

    def test_gaas_lattice_only_ga_and_as(self):
        _, small = gaas_lattice(6, 6)
        assert set(small.flatten()).issubset({0, 1})

    def test_gaas_lattice_alternates(self):
        """Every pair of adjacent atoms should be different species."""
        _, small = gaas_lattice(6, 6)
        assert np.all(
            small[:-1, :] != small[1:, :]
        ), "vertical neighbours not alternating"
        assert np.all(
            small[:, :-1] != small[:, 1:]
        ), "horizontal neighbours not alternating"

    def test_indium_centre_exactly_one_in(self):
        _, small = gaas_lattice_with_indium_centre(7, 7)
        assert np.sum(small == 2) == 1

    def test_indium_centre_at_centre(self):
        n = 7
        _, small = gaas_lattice_with_indium_centre(n, n)
        r, c = np.argwhere(small == 2)[0]
        # must be within one site of the geometric centre
        assert abs(r - n // 2) <= 1
        assert abs(c - n // 2) <= 1

    def test_indium_centre_replaces_ga_not_as(self):
        """In can only replace Ga (species 0), not As (species 1)."""
        n = 7
        _, gaas = gaas_lattice(n, n)
        _, with_in = gaas_lattice_with_indium_centre(n, n)
        r, c = np.argwhere(with_in == 2)[0]
        assert gaas[r, c] == 0, "In was placed on an As site"

    def test_block_in_positions_count(self):
        pos = block_in_positions([2, 2], [5, 5])
        assert len(pos) == 9  # 3×3

    def test_block_in_positions_range(self):
        pos = block_in_positions([1, 3], [4, 6])
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        assert min(xs) == 1 and max(xs) == 3
        assert min(ys) == 3 and max(ys) == 5


# ---------------------------------------------------------------------------
# Unstrained positions
# ---------------------------------------------------------------------------


class TestUnstrainedPositions:
    def test_shape(self):
        coords = unstrained_positions(5, 7)
        assert coords.shape == (5, 7, 2)

    def test_uniform_row_spacing(self):
        coords = unstrained_positions(6, 6)
        dy = np.diff(coords[:, 0, 1])
        assert np.allclose(dy, dy[0], rtol=1e-10)

    def test_uniform_col_spacing(self):
        coords = unstrained_positions(6, 6)
        dx = np.diff(coords[0, :, 0])
        assert np.allclose(dx, dx[0], rtol=1e-10)


# ---------------------------------------------------------------------------
# Strain tensor
# ---------------------------------------------------------------------------


class TestStrainTensor:
    def test_output_shape(self):
        coords = unstrained_positions(5, 7)
        tensors = strain_tensor(coords, coords)
        assert tensors.shape == (5, 7, 2, 2)

    def test_identical_coords_zero_strain(self):
        """No displacement → no strain. Tests the deformation gradient formula directly."""
        coords = unstrained_positions(6, 6)
        tensors = strain_tensor(coords, coords)
        assert np.allclose(tensors, 0, atol=1e-10)

    def test_symmetry_pure_gaas(self):
        """ε_01 == ε_10 for pure GaAs."""
        _, unstrained, strained, *_ = run_strain_simulation(5, 5, lattice_type=0)
        tensors = strain_tensor(strained, unstrained)
        assert np.allclose(tensors[:, :, 0, 1], tensors[:, :, 1, 0], atol=1e-10)

    def test_symmetry_with_indium(self):
        """ε_01 == ε_10 holds for strained lattice too."""
        _, unstrained, strained, *_ = run_strain_simulation(7, 7, lattice_type=1)
        tensors = strain_tensor(strained, unstrained)
        assert np.allclose(tensors[:, :, 0, 1], tensors[:, :, 1, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------


class TestRunStrainSimulation:
    def test_return_shapes(self):
        n = 7
        species, unstrained, strained, t, _, _ = run_strain_simulation(
            n, n, lattice_type=0
        )
        assert species.shape == (n, n)
        assert unstrained.shape == (n, n, 2)
        assert strained.shape == (n, n, 2)
        assert isinstance(t, float)

    def test_pure_gaas_zero_strain(self):
        """
        Pure GaAs: all Ga-As bonds share the same natural length (1.0) and the
        uniform grid already minimises the energy. The optimizer should leave
        positions unchanged, giving zero strain everywhere.
        """
        _, unstrained, strained, *_ = run_strain_simulation(5, 5, lattice_type=0)
        tensors = strain_tensor(strained, unstrained)
        assert np.allclose(tensors, 0, atol=1e-6)

    def test_pure_gaas_row_col_error_zero(self):
        *_, avg_row_diff, avg_col_diff = run_strain_simulation(5, 5, lattice_type=0)
        assert avg_row_diff == pytest.approx(0.0, abs=1e-3)
        assert avg_col_diff == pytest.approx(0.0, abs=1e-3)

    def test_pure_gaas_strained_equals_unstrained(self):
        _, unstrained, strained, *_ = run_strain_simulation(5, 5, lattice_type=0)
        assert np.allclose(strained, unstrained, atol=1e-6)

    def test_single_in_distorts_lattice(self):
        """In-As natural length is 1.06, so the In site must push neighbours out."""
        _, unstrained, strained, *_ = run_strain_simulation(7, 7, lattice_type=1)
        assert not np.allclose(strained, unstrained, atol=1e-4)

    def test_single_in_nonzero_strain(self):
        _, unstrained, strained, *_ = run_strain_simulation(7, 7, lattice_type=1)
        tensors = strain_tensor(strained, unstrained)
        assert np.max(np.abs(tensors)) > 1e-4

    def test_in_block_more_strain_than_single_in(self):
        """
        A 3×3 In block contains more Indium than a single atom and should
        produce a larger peak strain.
        """
        _, u1, s1, *_ = run_strain_simulation(9, 9, lattice_type=1)
        peak_single = np.max(np.abs(strain_tensor(s1, u1)))

        in_pos = block_in_positions([3, 3], [6, 6])
        _, u2, s2, *_ = run_strain_simulation(9, 9, lattice_type=2, in_positions=in_pos)
        peak_block = np.max(np.abs(strain_tensor(s2, u2)))

        assert peak_block > peak_single

    def test_in_block_species_contains_indium(self):
        in_pos = block_in_positions([2, 2], [5, 5])
        species, *_ = run_strain_simulation(9, 9, lattice_type=2, in_positions=in_pos)
        assert np.any(species == 2), "No Indium found in species array"
