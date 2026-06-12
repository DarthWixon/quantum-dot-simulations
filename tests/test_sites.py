import numpy as np

from qdot.sites import all_locations, random_locations, find_best_locations


class TestAllLocations:
    def test_count_and_contents(self):
        locs = all_locations((2, 3))
        assert len(locs) == 6
        assert locs == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


class TestRandomLocations:
    def test_count(self):
        assert len(random_locations(10, (4, 5), rng=0)) == 10

    def test_within_bounds(self):
        locs = random_locations(200, (4, 5), rng=0)
        rows = [r for r, _ in locs]
        cols = [c for _, c in locs]
        assert min(rows) >= 0 and max(rows) < 4
        assert min(cols) >= 0 and max(cols) < 5

    def test_seed_reproducible(self):
        assert random_locations(20, (10, 10), rng=42) == random_locations(
            20, (10, 10), rng=42
        )


class TestFindBestLocations:
    def test_all_seeds_snap_to_global_peak(self):
        # One strong peak; with a window covering the whole array every seed
        # must snap to it.
        xx = np.zeros((10, 10))
        xx[5, 5] = 1.0
        xz = np.zeros((10, 10))
        zz = np.zeros((10, 10))

        locs = find_best_locations(8, xx, xz, zz, search_range=10, rng=1)
        assert all(loc == (5, 5) for loc in locs)

    def test_boundary_peak_found_with_clipped_window(self):
        # Peak in the corner: window clipping must not shift the result.
        xx = np.zeros((6, 6))
        xx[0, 0] = 1.0

        locs = find_best_locations(5, xx, xx, xx, search_range=20, rng=2)
        assert all(loc == (0, 0) for loc in locs)

    def test_metric_is_component_sum(self):
        # Peaks in different components at different sites; the sum decides.
        xx = np.zeros((8, 8))
        xz = np.zeros((8, 8))
        zz = np.zeros((8, 8))
        xx[2, 2] = 0.4
        xz[2, 2] = 0.4  # combined 0.8 at (2, 2)
        zz[6, 6] = 0.7  # only 0.7 at (6, 6)

        locs = find_best_locations(5, xx, xz, zz, search_range=10, rng=3)
        assert all(loc == (2, 2) for loc in locs)
