import time
import numpy as np
from angkor.input_reader import InputReader
from angkor.solver_mg import SolverMG
from angkor.cmfd import CMFD

INPUT = "input/pwr_2d.yaml"
K_REF = 1.17978975          # verified baseline


def run(rf=None, interval=1, tol=1e-10, max_iter=400):
    """One case. rf=None means no CMFD."""
    reader = InputReader(INPUT)
    reader.read()
    reader.solver.convergence   = tol
    reader.solver.max_iterations = max_iter

    solver = SolverMG(reader.engine, reader.materials, reader.solver,
                      n_groups=2, boundary=reader.boundary)
    cmfd = CMFD(solver, rf=rf) if rf else None

    t0 = time.time()
    k, flux = solver.solve(cmfd=cmfd, cmfd_interval=interval)
    dt = time.time() - t0
    return k, solver.iterations, dt


print(f"{'rf':>4} {'cell':>6} {'int':>4} {'iters':>6} {'time':>7} {'k':>12} {'pcm':>9}")
print("-" * 56)

k, it, dt = run(rf=None)
print(f"{'--':>4} {'--':>6} {'--':>4} {it:6d} {dt:6.1f}s {k:12.8f} {(k-K_REF)*1e5:+9.2f}")

for rf in (2, 4, 5, 10, 20):
    for interval in (1, 2, 5):
        try:
            k, it, dt = run(rf=rf, interval=interval)
            flag = "" if abs(k - K_REF) * 1e5 < 1.0 else "  <-- WRONG"
            print(f"{rf:4d} {rf*1.0:5.0f}cm {interval:4d} {it:6d} {dt:6.1f}s "
                  f"{k:12.8f} {(k-K_REF)*1e5:+9.2f}{flag}")
        except Exception as e:
            print(f"{rf:4d} {rf*1.0:5.0f}cm {interval:4d}   FAILED: {type(e).__name__}")