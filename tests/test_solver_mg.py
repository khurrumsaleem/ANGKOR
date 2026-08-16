import pytest
from pathlib import Path
from angkor.input_reader import InputReader
from angkor.solver_mg import SolverMG

INPUT = str(Path(__file__).resolve().parents[1] / "input" / "pwr_2d.yaml")


@pytest.fixture(scope="module")
def solver():
    """Assembled solver, built once and shared by every test here."""
    reader = InputReader(INPUT)
    reader.read()
    s = SolverMG(reader.engine, reader.materials, reader.solver,
                 n_groups=2, boundary=reader.boundary)
    s._build_matrices()
    return s


def test_face_coupling_harmonic():
    """Harmonic mean; equal D unchanged; zero transmission into a black cell."""
    assert SolverMG._face_coupling(1.40, 1.13, 1.0) == pytest.approx(1.2505929, abs=1e-6)
    assert SolverMG._face_coupling(0.37, 0.16, 1.0) == pytest.approx(0.2233962, abs=1e-6)
    assert SolverMG._face_coupling(1.40, 1.40, 1.0) == pytest.approx(1.40)
    assert SolverMG._face_coupling(1.40, 0.00, 1.0) == 0.0


def test_A_is_symmetric(solver):
    """Diffusion operator is self-adjoint. Guards F1."""
    assert abs(solver.A - solver.A.T).max() < 1e-14


def test_interior_diagonal(solver):
    """Interior fuel cell, group 1: sigma_a + sigma_s12 + 4D/h^2."""
    row = solver._cell_index(0, 50, 50)
    assert solver.A[row, row] == pytest.approx(0.0095 + 0.0210 + 4*1.40, abs=1e-9)