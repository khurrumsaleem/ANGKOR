from angkor.input_reader import InputReader
from angkor.solver_mg import SolverMG

reader = InputReader("input/pwr_2d.yaml")
reader.read()

solver = SolverMG(reader.engine, reader.materials, reader.solver,
                  n_groups=2, boundary=reader.boundary)
solver._build_matrices()

asym  = abs(solver.A - solver.A.T).max()
scale = abs(solver.A).max()
print(f"max |A - A.T|      = {asym:.6e}")
print(f"max |A|            = {scale:.6e}")
print(f"relative asymmetry = {asym/scale:.6e}")

import numpy as np
d = abs(solver.A - solver.A.T).tocoo()
mask = d.data > 1e-12
print(f"asymmetric entries: {mask.sum()}")

rows = d.row[mask]
g    = rows // solver.N
n    = rows %  solver.N
print(f"  group 1: {(g==0).sum()},  group 2: {(g==1).sum()}")
print(f"  distinct |D jumps|: {np.unique(np.round(d.data[mask], 6))}")

if mask.sum():
    j, i = n // solver.nx, n % solver.nx
    print(f"  i range: {i.min()}..{i.max()},  j range: {j.min()}..{j.max()}")
else:
    print("  matrix is symmetric")