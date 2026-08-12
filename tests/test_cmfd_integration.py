from pathlib import Path
import sys
import numpy as np

# Make imports robust for pytest from repo root
ROOT = Path(__file__).resolve().parents[1]
ANGKOR_DIR = ROOT / "angkor"
for p in (ROOT, ANGKOR_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

try:
    from angkor.input_reader import InputReader
    from angkor.solver_mg import SolverMG
    from angkor.cmfd import CMFD
except ModuleNotFoundError:
    # Fallback for direct-module import style
    from input_reader import InputReader
    from solver_mg import SolverMG
    from cmfd import CMFD


INPUT_FILE = ROOT / "input" / "pwr_2d.yaml"


def _eigen_residual(solver, k_eff, flux):
    """
    Relative residual of A*phi = S*phi + (1/k)*F*phi
    """
    phi = flux.reshape(solver.G * solver.N)
    rhs = solver.S.dot(phi) + solver.F.dot(phi) / k_eff
    lhs = solver.A.dot(phi)
    num = np.linalg.norm(lhs - rhs)
    den = np.linalg.norm(rhs) + 1e-20
    return num / den


def _run_case(use_cmfd):
    reader = InputReader(str(INPUT_FILE))
    reader.read()

    solver = SolverMG(
        reader.engine,
        reader.materials,
        reader.solver,
        n_groups=2,
        boundary=reader.boundary,
    )

    cmfd = CMFD(solver, rf=10) if use_cmfd else None
    k_eff, flux = solver.solve(cmfd=cmfd, cmfd_interval=5)
    res = _eigen_residual(solver, k_eff, flux)

    return {
        "k": k_eff,
        "flux": flux,
        "iters": solver.iterations,
        "residual": res,
    }


def test_mg_cmfd_matches_baseline_and_is_well_converged():
    plain = _run_case(use_cmfd=False)
    cmfd = _run_case(use_cmfd=True)

    diff_pcm = abs(cmfd["k"] - plain["k"]) * 1.0e5

    # Physics consistency: CMFD should not bias k significantly.
    assert diff_pcm < 5.0, (
        f"CMFD k-bias too large: {diff_pcm:.2f} pcm "
        f"(k_plain={plain['k']:.8f}, k_cmfd={cmfd['k']:.8f})"
    )

    # Numerical consistency: both solutions should satisfy the eigen equation.
    assert plain["residual"] < 1e-5, (
        f"Baseline residual too large: {plain['residual']:.3e}"
    )
    assert cmfd["residual"] < 1e-4, (
        f"CMFD residual too large: {cmfd['residual']:.3e}"
    )

    # CMFD should not be slower than baseline.
    assert cmfd["iters"] <= plain["iters"], (
        f"CMFD did not accelerate: cmfd={cmfd['iters']} vs plain={plain['iters']}"
    )

    # Physical sanity: flux should remain positive.
    assert np.min(cmfd["flux"]) > 0.0, "CMFD produced non-positive flux."
