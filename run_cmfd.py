import time
from angkor.input_reader import InputReader
from angkor.solver_mg import SolverMG
from angkor.cmfd import CMFD

reader = InputReader("input/pwr_2d.yaml")
reader.read()
reader.solver.convergence = 1e-10          # same question as the baseline

solver = SolverMG(reader.engine, reader.materials, reader.solver,
                  n_groups=2, boundary=reader.boundary)
cmfd = CMFD(solver, rf=4)

t0 = time.time()
k, flux = solver.solve(cmfd=cmfd, cmfd_interval=2)
t1 = time.time()

print(f"\n  k          = {k:.8f}")
print(f"  iterations = {solver.iterations}")
print(f"  time       = {t1-t0:.1f} s")