import numpy as np
from angkor.input_reader import InputReader
from angkor.solver_mg import SolverMG
from angkor.cmfd import CMFD


reader = InputReader("input/pwr_2d.yaml")
reader.read()
reader.solver.convergence = 1e-10
solver = SolverMG(reader.engine, reader.materials, reader.solver,
                  n_groups=2, boundary=reader.boundary)

k_fine, flux = solver.solve()                 # converged, no CMFD

cmfd = CMFD(solver, rf=10)
flux_c         = cmfd.homogenize_flux_mg(flux)
dhat_e, dhat_n = cmfd.compute_dhat_mg(flux, flux_c)
Eb = cmfd.compute_boundary_coupling(flux, flux_c)
cmfd.build_coarse_matrices_mg(dhat_e, dhat_n, Eb)
k_c, flux_c_new = cmfd.solve_coarse_mg(k_fine, phi_init=flux_c)

print(f"\n  k_fine   = {k_fine:.8f}")
print(f"  k_coarse = {k_c:.8f}")
print(f"  delta    = {(k_c - k_fine)*1e5:+.5f} pcm")

a = flux_c     / flux_c.max()
b = flux_c_new / flux_c_new.max()
print(f"  shape L2 error  = {np.linalg.norm(a-b)/np.linalg.norm(a):.3e}")
print(f"  max local error = {np.abs(a-b).max():.3e}")
for g in range(2):
    print(f"  group {g+1} row 5: coarse-in  {np.round(a[g,5],4)}")
    print(f"           row 5: coarse-out {np.round(b[g,5],4)}")