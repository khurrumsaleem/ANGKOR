# solver_mg.py
# ANGKOR - Multi-group 2D neutron diffusion solver

import numpy as np
from scipy import sparse
from scipy.sparse import linalg


class SolverMG:
    """G-group 2D neutron diffusion solver."""

    def __init__(self, engine, materials, settings, n_groups, boundary=None):
        self.engine    = engine
        self.materials = materials
        self.settings  = settings
        self.G         = n_groups
        self.nx        = len(self.engine.x_centers)
        self.ny        = len(self.engine.y_centers)
        self.N         = self.nx * self.ny
        self.dx        = self.engine.domain_x / self.nx
        self.dy        = self.engine.domain_y / self.ny
        self.buckling  = getattr(settings, "buckling", 0.0)
        self.bc        = boundary or {
            "left": "vacuum", "right": "vacuum",
            "top":  "vacuum", "bottom": "vacuum",
        }
        self.k_eff      = None
        self.flux       = None
        self.iterations = 0

        print(f"    SolverMG Initialized---")
        print(f"    Groups  : {self.G}")
        print(f"    Mesh    : {self.nx} x {self.ny} = {self.N} cells")

    def _get_xs(self, i, j):
        """Return cross sections at cell (i,j) in unified array format.

        Supports two material formats from the input file:
          - 2-group format : keys D1, D2, sigma_a1, sigma_a2, sigma_s12,
                             nu_sigma_f1, nu_sigma_f2
          - Multi-group format: keys groups, D (list), sigma_a (list),
                             nu_sigma_f (list), sigma_s (list-of-lists), chi (list)
        Always returns a dict with array keys:
            D, sigma_a, nu_sigma_f, chi  -> shape (G,)
            sigma_s                      -> shape (G, G)
        Plus the flat 2-group scalar keys for backward-compat with cmfd.py
        2-group helpers (D1, D2, etc.) when G == 2.
        """
        mat_name = self.engine.material_map[j][i]
        mat      = self.materials[mat_name]

        if "groups" in mat:
            # ── Multi-group array format ──────────────────────────────
            G          = mat["groups"]
            D          = np.asarray(mat["D"],          dtype=float)
            sigma_a    = np.asarray(mat["sigma_a"],    dtype=float)
            nu_sigma_f = np.asarray(mat["nu_sigma_f"], dtype=float)
            chi        = np.asarray(mat["chi"],        dtype=float)
            sigma_s    = np.asarray(mat["sigma_s"],    dtype=float).reshape(G, G)
        else:
            # ── Legacy 2-group scalar format ─────────────────────────
            D1      = float(mat["D1"]);           D2      = float(mat["D2"])
            siga1   = float(mat["sigma_a1"]);     siga2   = float(mat["sigma_a2"])
            sigs12  = float(mat["sigma_s12"])
            nusigf1 = float(mat["nu_sigma_f1"]);  nusigf2 = float(mat["nu_sigma_f2"])

            D          = np.array([D1,      D2     ])
            sigma_a    = np.array([siga1,   siga2  ])
            nu_sigma_f = np.array([nusigf1, nusigf2])
            chi        = np.array([1.0,     0.0    ])
            # sigma_s[from_group, to_group]
            sigma_s    = np.array([[0.0, sigs12],
                                   [0.0, 0.0   ]])

        xs = {
            "D":          D,
            "sigma_a":    sigma_a,
            "nu_sigma_f": nu_sigma_f,
            "chi":        chi,
            "sigma_s":    sigma_s,
        }

        # Provide flat 2-group helpers for code that still uses them
        # (e.g., cmfd.compute_dhat which reads xs["D1"] etc.)
        if len(D) == 2:
            xs["D1"]          = D[0];          xs["D2"]          = D[1]
            xs["sigma_a1"]    = sigma_a[0];    xs["sigma_a2"]    = sigma_a[1]
            xs["sigma_s12"]   = sigma_s[0, 1]
            xs["nu_sigma_f1"] = nu_sigma_f[0]; xs["nu_sigma_f2"] = nu_sigma_f[1]

        return xs

    def _cell_index(self, g, i, j):
        return g*self.N + j*self.nx + i

    def _build_D_field(self):
        """Diffusion coefficient over the whole mesh, shape (G, ny, nx).

        Built once so the assembly loop can see a neighbour's D without
        calling _get_xs again.
        """
        D = np.zeros((self.G, self.ny, self.nx))
        for j in range(self.ny):
            for i in range(self.nx):
                D[:, j, i] = self._get_xs(i, j)["D"]
        return D

    @staticmethod
    def _face_coupling(D_a, D_b, h):
        """Coupling coefficient for the face between two cells, units 1/cm^2.

            2*D_a*D_b / ((D_a + D_b) * h**2)

        Must match the D_harm used in cmfd.compute_dhat_mg exactly.
        """
        s = D_a + D_b
        if s <= 0.0:
            return 0.0
        return 2.0 * D_a * D_b / (s * h * h)

    def _build_matrices(self):
        Dfield = self._build_D_field()
        size = self.G * self.N
        rows_A, cols_A, vals_A = [], [], []
        rows_F, cols_F, vals_F = [], [], []
        rows_S, cols_S, vals_S = [], [], []

        def add_A(r,c,v): rows_A.append(r); cols_A.append(c); vals_A.append(v)
        def add_F(r,c,v): rows_F.append(r); cols_F.append(c); vals_F.append(v)
        def add_S(r,c,v): rows_S.append(r); cols_S.append(c); vals_S.append(v)

        for j in range(self.ny):
            for i in range(self.nx):
                xs     = self._get_xs(i, j)
                D      = xs["D"];       siga   = xs["sigma_a"]
                nusigf = xs["nu_sigma_f"]; sigs = xs["sigma_s"]
                chi    = xs["chi"]
                siga_eff = [siga[g] + D[g]*self.buckling for g in range(self.G)]

                for g in range(self.G):
                    row = self._cell_index(g, i, j)
                    Dx_bnd  = D[g] / self.dx**2
                    Dy_bnd  = D[g] / self.dy**2
                    sigma_r = siga_eff[g] + sum(
                        sigs[g,g2] for g2 in range(self.G) if g2 != g)
                    C = sigma_r

                    if i > 0:
                        c = self._face_coupling(Dfield[g,j,i], Dfield[g,j,i-1], self.dx)
                        add_A(row, self._cell_index(g,i-1,j), -c); C += c
                    elif self.bc.get("left","vacuum") == "vacuum":
                        C += Dx_bnd
                    
                    if i < self.nx-1:
                        c = self._face_coupling(Dfield[g,j,i], Dfield[g,j,i+1], self.dx)
                        add_A(row, self._cell_index(g,i+1,j), -c); C += c
                    elif self.bc.get("right","vacuum") == "vacuum":
                        C += Dx_bnd
                    
                    if j > 0:
                        c = self._face_coupling(Dfield[g,j,i], Dfield[g,j-1,i], self.dy)
                        add_A(row, self._cell_index(g,i,j-1), -c); C += c
                    elif self.bc.get("bottom","vacuum") == "vacuum":
                        C += Dy_bnd
                    
                    if j < self.ny-1:
                        c = self._face_coupling(Dfield[g,j,i], Dfield[g,j+1,i], self.dy)
                        add_A(row, self._cell_index(g,i,j+1), -c); C += c
                    elif self.bc.get("top","vacuum") == "vacuum":
                        C += Dy_bnd

                    add_A(row, row, C)

                    for g2 in range(self.G):
                        if g2 != g:
                            add_S(row, self._cell_index(g2,i,j), sigs[g2,g])
                    for g2 in range(self.G):
                        add_F(row, self._cell_index(g2,i,j), chi[g]*nusigf[g2])

        self.A = sparse.csr_matrix((vals_A,(rows_A,cols_A)), shape=(size,size))
        self.F = sparse.csr_matrix((vals_F,(rows_F,cols_F)), shape=(size,size))
        self.S = sparse.csr_matrix((vals_S,(rows_S,cols_S)), shape=(size,size))
        print(f"  Matrix A: {self.A.shape}, {self.A.nnz} non-zeros")
        print(f"  Matrix F: {self.F.shape}, {self.F.nnz} non-zeros")
        print(f"  Matrix S: {self.S.shape}, {self.S.nnz} non-zeros")
        print(f"   Diagonal average leakage term: {np.mean(self.A.diagonal())}")

    def solve(self, cmfd=None, cmfd_interval=5):
        """
        Power iteration with optional MG CMFD acceleration.

        cmfd_interval : apply CMFD every this many iterations (default 5).
                        Ignored when cmfd is None.

        KEY DESIGN RULE — convergence is checked BEFORE CMFD is applied.
        This guarantees that, at the moment we break, self.k_eff and
        self.flux are both produced by the same power-iteration step and
        are therefore mutually consistent.  Previously, CMFD ran first and
        then convergence was tested against the pre-CMFD k_error, which
        left k and phi describing different states (+65 pcm bias).
        """
        if cmfd_interval < 1:
            raise ValueError("cmfd_interval must be >= 1")

        N        = self.N
        G        = self.G
        max_iter = getattr(self.settings, "max_iterations", 1000)
        tol      = self.settings.convergence

        print(f"\n  Starting power iteration (G={G} groups)...")
        print(f"  Building matrices...")
        self._build_matrices()

        phi = np.ones(G * N)
        k   = 1.0
        cmfd_active = cmfd is not None

        for iteration in range(max_iter):

            # ── Standard power iteration ───────────────────────────────
            fission_source = self.F.dot(phi)
            scatter_source = self.S.dot(phi)
            rhs     = scatter_source + fission_source / k
            phi_new = linalg.spsolve(self.A, rhs)

            fission_new = self.F.dot(phi_new)
            k_new     = k * (np.sum(fission_new) / np.sum(fission_source))
            k_error   = abs(k_new - k)
            phi_error = np.max(
                np.abs(phi_new - phi) / (np.abs(phi_new) + 1e-12)
            )

            phi = phi_new / phi_new.max()
            k   = k_new

            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"  Iter {iteration+1:4d}: k={k:.6f}  dk={k_error:.2e}  "
                      f"dphi={phi_error:.2e}")

            # ── Convergence check — BEFORE CMFD ───────────────────────
            # THE KEY BIAS FIX:
            # Check convergence BEFORE applying CMFD, not after.
            # This guarantees that self.k_eff and self.flux are produced
            # by the SAME power-iteration step and are mutually consistent.
            #
            # When CMFD is active, require only k_error < tol.
            # CMFD continuously nudges phi each iteration; requiring
            # phi_error < tol would prevent convergence since phi_error
            # also measures the CMFD perturbation itself.
            if cmfd_active:
                converged = k_error < tol
            else:
                converged = (k_error < tol) and (phi_error < tol)

            if converged:
                print(f"\n  CONVERGED at iteration {iteration + 1}!")
                break

            # ── MG CMFD acceleration ───────────────────────────────────
            # Apply every cmfd_interval iterations once k is within 0.5%
            # of its final value (so d-hat is computed from a reasonably
            # converged flux, keeping corrections small and stable).
            # alpha=0.1: each CMFD step nudges phi by only 10%; the powercmf
            # iteration dominates and CMFD gradually steers the flux shape.
            if cmfd_active and (iteration % cmfd_interval) == 0:
                phi_before   = phi.copy()
                flux_g       = phi.reshape(G, self.ny, self.nx)
                flux_g, k    = cmfd.accelerate_mg(flux_g, k, n_cycles=1)
                phi_new_cmfd = flux_g.reshape(G * N)

                # Safety guard: reject if CMFD produced non-physical values
                if phi_new_cmfd.min() < 0 or phi_new_cmfd.max() > 1e6:
                    print(f"  [CMFD] rejected at iter {iteration+1} "
                          f"(non-physical flux)")
                    phi_new_cmfd = phi_before

                phi = phi_new_cmfd / phi_new_cmfd.max()
            
        self.iterations = iteration + 1
        self.k_eff = k
        self.flux  = np.zeros((G, self.ny, self.nx))
        for g in range(G):
            self.flux[g] = phi[g*N:(g+1)*N].reshape(self.ny, self.nx)
    
        print(f"\n  {'='*40}")
        print(f"  k-eff = {self.k_eff:.6f}")
        print(f"  {'='*40}")
        print(f"\n  Total iterations: {iteration + 1}")
        return self.k_eff, self.flux

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from input_reader import InputReader
    filepath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "input", "pwr_2d.yaml")
    reader = InputReader(filepath)
    reader.read()
    solver = SolverMG(reader.engine, reader.materials,
                      reader.solver, n_groups=2)
    k_eff, flux = solver.solve()
    print(f"\n  Result: k-eff = {k_eff:.6f}")
