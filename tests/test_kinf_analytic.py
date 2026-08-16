import pytest
from pathlib import Path
from angkor.input_reader import InputReader
from angkor.solver_mg import SolverMG

INPUT = str(Path(__file__).resolve().parents[1] / "input" / "kinf_test.yaml")
# BC = {"left": "reflective", "right": "reflective",
#       "top": "reflective", "bottom": "reflective"}


def test_kinf_analytic():
    """Infinite medium, zero leakage. k_inf follows from arithmetic:
        phi2/phi1 = sigma_s12 / sigma_a2
        k_inf = (nsf1 + nsf2 * phi2/phi1) / (sigma_a1 + sigma_s12)
    Guards removal, scattering, fission and reflective BC assembly.
    """
    reader = InputReader(INPUT)
    reader.read()
    solver = SolverMG(reader.engine, reader.materials, reader.solver,
                      n_groups=2, boundary=reader.boundary)
    k, _ = solver.solve()

    ratio = 0.0210 / 0.0820
    k_analytic = (0.0060 + 0.1350 * ratio) / (0.0095 + 0.0210)
    assert abs(k - k_analytic) * 1e5 < 1.0      # within 1 pcm