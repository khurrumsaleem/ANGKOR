from angkor.input_reader import InputReader
from angkor.solver_mg import SolverMG

reader = InputReader("input/kinf_test.yaml")
reader.read()

# BC = {"left": "reflective", "right": "reflective",
#       "top": "reflective", "bottom": "reflective"}

solver = SolverMG(reader.engine, reader.materials, reader.solver,
                  n_groups=2, boundary=reader.boundary)
k, flux = solver.solve()

K_ANALYTIC = 1.330268
print(f"\n  k computed = {k:.8f}")
print(f"  k analytic = {K_ANALYTIC:.8f}")
print(f"  difference = {(k - K_ANALYTIC)*1e5:+.3f} pcm")