import os
import sys
import numpy as np

# Make angkor package importable when pytest runs from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "angkor"))

from cmfd import CMFD


class StubSolver:
    """
    Minimal solver stub to exercise CMFD multi-group routines.
    Provides geometry sizes, spacing, and a _get_xs() method.
    """

    def __init__(self):
        self.nx = 4
        self.ny = 4
        self.dx = 1.0
        self.dy = 1.0
        self.G = 2  # two energy groups

    def _get_xs(self, i, j):
        # Uniform cross sections sufficient for algebraic checks
        return {
            "D": [1.0, 0.5],
            "sigma_a": [0.1, 0.2],
            "nu_sigma_f": [0.2, 0.1],
            "sigma_s": [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            "chi": [1.0, 0.0],
        }


def test_rebalance_flux_mg_applies_coarse_ratio():
    solver = StubSolver()
    cmfd = CMFD(solver, rf=2, n_groups=solver.G)

    # Fine flux: group 0 > group 1 so scaling is detectable
    flux_fine = np.array(
        [
            np.ones((4, 4)),
            0.5 * np.ones((4, 4)),
        ]
    )

    flux_c_old = cmfd.homogenize_flux_mg(flux_fine)
    # Perturb coarse flux to force rescaling.
    # Current CMFD design preserves amplitude correction (no per-group renorm).
    factors = np.array([2.0, 0.4])[:, None, None]
    flux_c_new = flux_c_old * factors

    flux_new = cmfd.rebalance_flux_mg(
        flux_fine, flux_c_old, flux_c_new
    )

    # Rebalance should apply ratio directly (subject to clip).
    assert np.allclose(flux_new[0].sum(), flux_fine[0].sum() * 2.0, rtol=1e-12)
    assert np.allclose(flux_new[1].sum(), flux_fine[1].sum() * 0.4, rtol=1e-12)
    assert np.all(flux_new > 0.0)

def test_accelerate_mg_returns_coarse_k():
    """After F4/F5 the coarse eigenvalue is exact and must be returned."""
    solver = StubSolver()
    cmfd = CMFD(solver, rf=2, n_groups=solver.G)
    flux = np.ones((solver.G, solver.ny, solver.nx))
    flux_out, k_out = cmfd.accelerate_mg(flux, 1.250, n_cycles=1)
    assert k_out != 1.250          # no longer the pass-through
    assert k_out > 0
