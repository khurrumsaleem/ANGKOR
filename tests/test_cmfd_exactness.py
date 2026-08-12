import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'angkor'))
import numpy as np

from angkor.input_reader import InputReader
from angkor.solver_mg   import SolverMG
from angkor.cmfd        import CMFD

reader = InputReader("input/pwr_2d.yaml")
reader.read()

from pathlib import Path
INPUT = str(Path(__file__).resolve().parents[1] / "input" / "pwr_2d.yaml")

def test_cmfd_exactness():
    """Coarse operator built from a converged fine solution must
    reproduce that solution exactly. Guards F4.
    """
    reader = InputReader(INPUT)
    reader.read()
    reader.solver.convergence = 1e-10
    solver = SolverMG(reader.engine, reader.materials, reader.solver,
                      n_groups=2, boundary=reader.boundary)
    k_fine, flux = solver.solve()

    cmfd   = CMFD(solver, rf=10)
    flux_c = cmfd.homogenize_flux_mg(flux)
    de, dn = cmfd.compute_dhat_mg(flux, flux_c)
    Eb     = cmfd.compute_boundary_coupling(flux, flux_c)
    cmfd.build_coarse_matrices_mg(de, dn, Eb)
    k_c, flux_c_new = cmfd.solve_coarse_mg(k_fine, phi_init=flux_c)

    assert abs(k_c - k_fine) * 1e5 < 0.01          # pcm
    a = flux_c / flux_c.max()
    b = flux_c_new / flux_c_new.max()
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-8

# ── RUN 1: No CMFD ────────────────────────────────────────────────────
print("\n" + "="*50)
print("  RUN 1: No CMFD")
print("="*50)
s1 = SolverMG(reader.engine, reader.materials, reader.solver, n_groups=2, boundary=reader.boundary)
t0 = time.time()
k_plain, _ = s1.solve(cmfd=None)
t1 = time.time()
print(f"  Time: {t1-t0:.1f}s   Iterations: {s1.iterations}")

# ── RUN 2: With MG CMFD ───────────────────────────────────────────────
print("\n" + "="*50)
print("  RUN 2: With MG CMFD")
print("="*50)
s2 = SolverMG(reader.engine, reader.materials, reader.solver, n_groups=2, boundary=reader.boundary)
cmfd_mg = CMFD(s2, rf=10)
t0 = time.time()
k_cmfd, _ = s2.solve(cmfd=cmfd_mg, cmfd_interval=5)
t1 = time.time()
print(f"  Time: {t1-t0:.1f}s   Iterations: {s2.iterations}")

# ── COMPARISON ────────────────────────────────────────────────────────
diff = abs(k_cmfd - k_plain) * 1e5
speedup = s1.iterations / s2.iterations

print("\n" + "="*50)
print("  COMPARISON")
print("="*50)
print(f"  k without CMFD : {k_plain:.6f}  ({s1.iterations} iters)")
print(f"  k with MG CMFD : {k_cmfd:.6f}  ({s2.iterations} iters)")
print(f"  Difference     : {diff:.1f} pcm  (target: < 5 pcm)")
print(f"  Iter speedup   : {speedup:.2f}x  (target: > 2x)")

if diff < 5 and s2.iterations < 30:
    print("\n   STATUS: PASS — MG CMFD working correctly")
elif diff < 5:
    print("\n   STATUS: PARTIAL — k agrees but speedup below target")
    print("     Check cmfd_interval and gating thresholds in solver_mg.py")
else:
    print(f"\n  STATUS: FAIL — {diff:.0f} pcm bias")
    print("     Check BC flags in build_coarse_matrices_mg:")
    print("     west_vac and south_vac should match solver.bc")
    # Run diagnostic
    print("\n  Running CMFD matrix diagnostic...")
    from angkor.cmfd import CMFD as CMFDa
    s3      = SolverMG(reader.engine, reader.materials, reader.solver, n_groups=2)
    cmfd_d  = CMFDa(s3, rf=10)
    # Need a flux to build matrices — use flat flux
    import numpy as np
    flat = np.ones((2, s3.ny, s3.nx))
    flat_c = cmfd_d.homogenize_flux_mg(flat)
    dhat_e, dhat_n = cmfd_d.compute_dhat_mg(flat, flat_c)
    cmfd_d._flux_fine = flat
    cmfd_d.build_coarse_matrices_mg(dhat_e, dhat_n)
    cmfd_d.diagnostic()
    print("  solver.bc =", s3.bc)
