import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__)), "angkor"))

from input_reader import InputReader
from solver_mg import SolverMG

reader = InputReader("input/pwr_2d.yaml")
reader.read()

solver = SolverMG(reader.engine, reader.materials, reader.solver, 
                  n_groups=2, boundary = reader.boundary)
solver._build_matrices()

asym  = abs(solver.A - solver.A.T).max()
scale = abs(solver.A).max()

print(f"max |A-A.T|         ={asym:.6e}")
print(f"max |A|             ={scale:.6e}")
print(f"relative asymetry   ={asym/scale:.6e} ")