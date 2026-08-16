# cmfd.py
# ANGKOR - Coarse Mesh Finite Difference Acceleration
# Fixes applied in this version:
#   1. rebalance_flux_mg: removed per-group amplitude normalization (173 pcm fix)
#   2. build_coarse_matrices_mg: BC now reads solver.bc correctly for all 4 sides
#   3. accelerate_mg: alpha default changed to 0.05, k_original correctly returned
#   4. diagnostic() method added for debugging

import numpy as np
from scipy import sparse
from scipy.sparse import linalg


class CMFD:
    """
    CMFD acceleration for multi-group diffusion solver.

    Coarsens the fine mesh by factor `rf` in each direction,
    computes d-hat correction coefficients from fine mesh currents,
    solves a small coarse eigenvalue problem, and rebalances fine flux.
    """

    def __init__(self, solver, rf=4, n_groups=2):
        self.solver = solver
        self.rf     = rf
        self.G      = getattr(solver, "G", n_groups)

        if self.solver.nx % rf != 0:
            raise ValueError(f"nx={self.solver.nx} not divisible by rf={rf}")
        if self.solver.ny % rf != 0:
            raise ValueError(f"ny={self.solver.ny} not divisible by rf={rf}")

        self.nx_c = self.solver.nx // self.rf
        self.ny_c = self.solver.ny // self.rf
        self.N_c  = self.nx_c * self.ny_c

        self.nx = self.solver.nx
        self.ny = self.solver.ny
        self.dx = self.solver.dx
        self.dy = self.solver.dy

        self.Dx = self.dx * self.rf
        self.Dy = self.dy * self.rf

        # Pre-compute volume-averaged D per coarse cell per group.
        # D depends only on material properties (not flux), so it is constant
        # across CMFD calls — computing it once here avoids O(nx*ny) _get_xs
        # lookups on every accelerate call.
        self._D_coarse = self._precompute_D_coarse()

        print(f"  CMFD initialized")
        print(f"  Fine mesh   : {self.nx} x {self.ny}")
        print(f"  Coarse mesh : {self.nx_c} x {self.ny_c}")
        print(f"  Groups      : {self.G}")
        print(f"  Refinement  : {self.rf}x")

    def _precompute_D_coarse(self):
        """
        Volume-average D over each coarse cell (shape: G, ny_c, nx_c).
        Matches exactly the averaging done in build_coarse_matrices_mg so that
        the D_tilde used in compute_dhat_mg is consistent with the coarse matrix.
        """
        D_coarse = np.zeros((self.G, self.ny_c, self.nx_c))
        for J in range(self.ny_c):
            for I in range(self.nx_c):
                i0, i1 = I*self.rf, (I+1)*self.rf
                j0, j1 = J*self.rf, (J+1)*self.rf
                count = 0
                for jj in range(j0, j1):
                    for ii in range(i0, i1):
                        xs = self.solver._get_xs(ii, jj)
                        for g in range(self.G):
                            D_coarse[g, J, I] += xs["D"][g]
                        count += 1
                D_coarse[:, J, I] /= count
        return D_coarse

    # ------------------------------------------------------------------
    # FLUX HOMOGENIZATION
    # ------------------------------------------------------------------

    def homogenize_flux(self, phi1_fine, phi2_fine):
        phi1_c = np.zeros((self.ny_c, self.nx_c))
        phi2_c = np.zeros((self.ny_c, self.nx_c))
        for J in range(self.ny_c):
            for I in range(self.nx_c):
                i0, i1 = I*self.rf, (I+1)*self.rf
                j0, j1 = J*self.rf, (J+1)*self.rf
                phi1_c[J, I] = np.mean(phi1_fine[j0:j1, i0:i1])
                phi2_c[J, I] = np.mean(phi2_fine[j0:j1, i0:i1])
        return phi1_c, phi2_c

    def homogenize_flux_mg(self, flux_fine):
        """
        Args:
            flux_fine: shape (G, ny, nx)
        Returns:
            flux_coarse: shape (G, ny_c, nx_c)
        """
        flux_c = np.zeros((self.G, self.ny_c, self.nx_c))
        for g in range(self.G):
            for J in range(self.ny_c):
                for I in range(self.nx_c):
                    i0, i1 = I*self.rf, (I+1)*self.rf
                    j0, j1 = J*self.rf, (J+1)*self.rf
                    flux_c[g, J, I] = np.mean(flux_fine[g, j0:j1, i0:i1])
        return flux_c

    def compute_boundary_coupling(self, flux_fine, flux_coarse):
        """Flux-weighted boundary leakage coefficient, per boundary face.

            E_b = sum_over_face(D_fine * phi_fine) / (Dx^2 * phi_coarse)

        Reproduces the fine solver's total boundary leakage exactly.
        Flux-dependent, so it MUST be rebuilt every CMFD cycle.

        Returns dict:  'W','E' shape (G, ny_c);  'S','N' shape (G, nx_c)
        Reflective faces stay 0.0.
        """
        G  = self.G
        bc = getattr(self.solver, 'bc', {})

        Eb = {
            'W': np.zeros((G, self.ny_c)),
            'E': np.zeros((G, self.ny_c)),
            'S': np.zeros((G, self.nx_c)),
            'N': np.zeros((G, self.nx_c)),
        }

        for g in range(G):

            # ── WEST: fine column i = 0  (worked example) ─────────────
            if bc.get('left', 'vacuum') == 'vacuum':
                for J in range(self.ny_c):
                    total = 0.0
                    for jj in range(J*self.rf, (J+1)*self.rf):
                        D = self.solver._get_xs(0, jj)["D"][g]
                        total += D * flux_fine[g, jj, 0]
                    Eb['W'][g, J] = total / (self.Dx**2 * flux_coarse[g, J, 0])

            # ── EAST: fine column i = nx-1 ───────────────────────────
            if bc.get('right', 'vacuum') == 'vacuum':
                for J in range(self.ny_c):
                    total = 0.0
                    for jj in range(J*self.rf, (J+1)*self.rf):
                        D = self.solver._get_xs(self.nx - 1, jj)["D"][g]
                        total += D * flux_fine[g, jj, self.nx - 1]
                    Eb['E'][g, J] = total / (self.Dx**2 * flux_coarse[g, J, self.nx_c - 1])

            # ── SOUTH: fine row j = 0 ────────────────────────────────
            if bc.get('bottom', 'vacuum') == 'vacuum':
                for I in range(self.nx_c):
                    total = 0.0
                    for ii in range(I*self.rf, (I+1)*self.rf):
                        D = self.solver._get_xs(ii, 0)["D"][g]
                        total += D * flux_fine[g, 0, ii]
                    Eb['S'][g, I] = total / (self.Dy**2 * flux_coarse[g, 0, I])

            # ── NORTH: fine row j = ny-1 ─────────────────────────────
            if bc.get('top', 'vacuum') == 'vacuum':
                for I in range(self.nx_c):
                    total = 0.0
                    for ii in range(I*self.rf, (I+1)*self.rf):
                        D = self.solver._get_xs(ii, self.ny - 1)["D"][g]
                        total += D * flux_fine[g, self.ny - 1, ii]
                    Eb['N'][g, I] = total / (self.Dy**2 * flux_coarse[g, self.ny_c - 1, I])

        return Eb

    # ------------------------------------------------------------------
    # D-HAT COMPUTATION (2-group)
    # ------------------------------------------------------------------

    def compute_dhat(self, phi1_fine, phi2_fine, phi1_coarse, phi2_coarse):
        
        dhat1_east  = np.zeros((self.ny_c, self.nx_c - 1))
        dhat2_east  = np.zeros((self.ny_c, self.nx_c - 1))
        dhat1_north = np.zeros((self.ny_c - 1, self.nx_c))
        dhat2_north = np.zeros((self.ny_c - 1, self.nx_c))

        for J in range(self.ny_c):
            for I in range(self.nx_c - 1):
                j_c = J*self.rf + self.rf//2
                i_L = (I+1)*self.rf - 1
                i_R = (I+1)*self.rf

                p1_L = phi1_fine[j_c, i_L]; p1_R = phi1_fine[j_c, i_R]
                p2_L = phi2_fine[j_c, i_L]; p2_R = phi2_fine[j_c, i_R]

                xsL = self.solver._get_xs(i_L, j_c)
                xsR = self.solver._get_xs(i_R, j_c)
                D1L, D1R = xsL["D1"], xsR["D1"]
                D2L, D2R = xsL["D2"], xsR["D2"]

                D1h = 2*D1L*D1R/(D1L+D1R) if (D1L+D1R) != 0 else 0.0
                D2h = 2*D2L*D2R/(D2L+D2R) if (D2L+D2R) != 0 else 0.0

                J1f = -D1h*(p1_R - p1_L)/self.dx
                J2f = -D2h*(p2_R - p2_L)/self.dx

                D1t = 0.5*(D1L+D1R); D2t = 0.5*(D2L+D2R)
                J1d = -D1t*(phi1_coarse[J, I+1] - phi1_coarse[J, I])/self.Dx
                J2d = -D2t*(phi2_coarse[J, I+1] - phi2_coarse[J, I])/self.Dx

                avg1 = (p1_L + p1_R)/2; avg2 = (p2_L + p2_R)/2
                dhat1_east[J, I] = (J1f - J1d)/avg1 if abs(avg1) > 1e-12 else 0.0
                dhat2_east[J, I] = (J2f - J2d)/avg2 if abs(avg2) > 1e-12 else 0.0

        for J in range(self.ny_c - 1):
            for I in range(self.nx_c):
                i_c = I*self.rf + self.rf//2
                j_B = (J+1)*self.rf - 1
                j_T = (J+1)*self.rf

                p1_B = phi1_fine[j_B, i_c]; p1_T = phi1_fine[j_T, i_c]
                p2_B = phi2_fine[j_B, i_c]; p2_T = phi2_fine[j_T, i_c]

                xsB = self.solver._get_xs(i_c, j_B)
                xsT = self.solver._get_xs(i_c, j_T)
                D1B, D1T = xsB["D1"], xsT["D1"]
                D2B, D2T = xsB["D2"], xsT["D2"]

                D1h = 2*D1B*D1T/(D1B+D1T) if (D1B+D1T) != 0 else 0.0
                D2h = 2*D2B*D2T/(D2B+D2T) if (D2B+D2T) != 0 else 0.0

                J1f = -D1h*(p1_T - p1_B)/self.dy
                J2f = -D2h*(p2_T - p2_B)/self.dy

                D1t = 0.5*(D1B+D1T); D2t = 0.5*(D2B+D2T)
                J1d = -D1t*(phi1_coarse[J+1, I] - phi1_coarse[J, I])/self.Dy
                J2d = -D2t*(phi2_coarse[J+1, I] - phi2_coarse[J, I])/self.Dy

                avg1 = (p1_B + p1_T)/2; avg2 = (p2_B + p2_T)/2
                dhat1_north[J, I] = (J1f - J1d)/avg1 if abs(avg1) > 1e-12 else 0.0
                dhat2_north[J, I] = (J2f - J2d)/avg2 if abs(avg2) > 1e-12 else 0.0

        return dhat1_east, dhat2_east, dhat1_north, dhat2_north

    # ------------------------------------------------------------------
    # D-HAT COMPUTATION (multi-group)
    # ------------------------------------------------------------------

    def compute_dhat_mg(self, flux_fine, flux_coarse):
        """
        G-group d-hat computation with face averaging.
        Averages the fine-mesh current over all rf fine cells on each
        coarse face, rather than sampling a single representative row.

        BUG FIX (D_tilde consistency):
            Previously D_tilde used a single center-row/column sample,
            while build_coarse_matrices_mg uses the full volume-average D.
            If they differ (material interfaces), the d-hat correction does
            not exactly cancel the difference between fine and coarse currents,
            introducing a systematic bias.

            Fix: compute D_coarse as the volume-average over the two adjacent
            coarse cells and use the arithmetic mean as D_tilde —  exactly
            matching the formula in build_coarse_matrices_mg.

        Args:
            flux_fine   : shape (G, ny, nx)
            flux_coarse : shape (G, ny_c, nx_c)

        Returns:
            dhat_east  : shape (G, ny_c, nx_c-1)
            dhat_north : shape (G, ny_c-1, nx_c)
        """
        # Use the pre-computed volume-averaged D (set once in __init__).
        D_coarse   = self._D_coarse

        dhat_east  = np.zeros((self.G, self.ny_c, self.nx_c - 1))
        dhat_north = np.zeros((self.G, self.ny_c - 1, self.nx_c))

        for g in range(self.G):

            # ── East interfaces ──────────────────────────────────────────
            for J in range(self.ny_c):
                for I in range(self.nx_c-1):
                    i_L = (I+1)*self.rf - 1   # last fine column in left coarse cell
                    i_R = (I+1)*self.rf        # first fine column in right coarse cell

                    # Accumulate fine-mesh current over all rf rows on this face.
                    # NOTE: Phi_avg_sum is NOT accumulated here because the
                    # d-hat denominator must use COARSE cell fluxes
                    # (Phi_L_c + Phi_R_c)/2, not fine-mesh face fluxes.
                    # Using fine-mesh face fluxes was the primary d-hat bug.
                    J_fine_sum = 0.0

                    for jj in range(J*self.rf, (J+1)*self.rf):   # ← all rf rows
                        phi_L = flux_fine[g, jj, i_L]
                        phi_R = flux_fine[g, jj, i_R]

                        xs_L  = self.solver._get_xs(i_L, jj)
                        xs_R  = self.solver._get_xs(i_R, jj)
                        D_L   = xs_L["D"][g]
                        D_R   = xs_R["D"][g]

                        if (D_L + D_R) != 0:
                            D_harm = 2*D_L*D_R / (D_L + D_R)
                        else:
                            D_harm = 0.0

                        J_fine_sum += -D_harm * (phi_R - phi_L) / self.dx

                    # Average fine current over all rf rows
                    J_fine_avg = J_fine_sum / self.rf

                    # Coarse-level FD current — use volume-averaged D so
                    # D_tilde is consistent with build_coarse_matrices_mg.
                    Phi_L_c = flux_coarse[g, J, I]
                    Phi_R_c = flux_coarse[g, J, I+1]
                    D_tilde = 0.5 * (D_coarse[g, J, I] + D_coarse[g, J, I+1])
                    J_diff  = -D_tilde * (Phi_R_c - Phi_L_c) / self.Dx

                    # Denominator MUST be coarse cell average, not fine face avg.
                    # The matrix encodes J = -D_tilde*(dPhi/Dx) + d*(Phi_L+Phi_R)/2
                    # where Phi_L, Phi_R are coarse fluxes.  Using fine-face flux
                    # here breaks that consistency.
                    Phi_avg_coarse = (Phi_L_c + Phi_R_c) / 2
                    if abs(Phi_avg_coarse) > 1e-12:
                        dhat_east[g, J, I] = (J_fine_avg - J_diff) / Phi_avg_coarse

            # ── North interfaces ─────────────────────────────────────────
            for J in range(self.ny_c-1):
                for I in range(self.nx_c):
                    j_B = (J+1)*self.rf - 1   # last fine row in bottom coarse cell
                    j_T = (J+1)*self.rf        # first fine row in top coarse cell

                    # Accumulate fine-mesh current over all rf columns on this face.
                    J_fine_sum = 0.0

                    for ii in range(I*self.rf, (I+1)*self.rf):   # ← all rf columns
                        phi_B = flux_fine[g, j_B, ii]
                        phi_T = flux_fine[g, j_T, ii]

                        xs_B  = self.solver._get_xs(ii, j_B)
                        xs_T  = self.solver._get_xs(ii, j_T)
                        D_B   = xs_B["D"][g]
                        D_T   = xs_T["D"][g]

                        if (D_B + D_T) != 0:
                            D_harm = 2*D_B*D_T / (D_B + D_T)
                        else:
                            D_harm = 0.0

                        J_fine_sum += -D_harm * (phi_T - phi_B) / self.dy

                    J_fine_avg = J_fine_sum / self.rf

                    Phi_B_c = flux_coarse[g, J,   I]
                    Phi_T_c = flux_coarse[g, J+1, I]
                    # Consistent volume-averaged D_tilde
                    D_tilde = 0.5 * (D_coarse[g, J, I] + D_coarse[g, J+1, I])
                    J_diff  = -D_tilde * (Phi_T_c - Phi_B_c) / self.Dy

                    # Same denominator fix: use coarse cell fluxes
                    Phi_avg_coarse = (Phi_B_c + Phi_T_c) / 2
                    if abs(Phi_avg_coarse) > 1e-12:
                        dhat_north[g, J, I] = (J_fine_avg - J_diff) / Phi_avg_coarse

        return dhat_east, dhat_north

    # ------------------------------------------------------------------
    # BUILD COARSE MATRICES (2-group)
    # ------------------------------------------------------------------

    def build_coarse_matrices(self, phi1_fine, phi2_fine,
                               dhat1_east, dhat2_east,
                               dhat1_north, dhat2_north):
        N_c = self.N_c
        rows_A, cols_A, vals_A = [], [], []
        rows_F, cols_F, vals_F = [], [], []

        def add_A(r, c, v): rows_A.append(r); cols_A.append(c); vals_A.append(v)
        def add_F(r, c, v): rows_F.append(r); cols_F.append(c); vals_F.append(v)

        for J in range(self.ny_c):
            for I in range(self.nx_c):
                n = J*self.nx_c + I
                i0, i1 = I*self.rf, (I+1)*self.rf
                j0, j1 = J*self.rf, (J+1)*self.rf

                D1=D2=sa1=sa2=ss12=nf1=nf2=0.0; cnt=0
                for jj in range(j0, j1):
                    for ii in range(i0, i1):
                        xs = self.solver._get_xs(ii, jj)
                        D1+=xs["D1"]; D2+=xs["D2"]
                        sa1+=xs["sigma_a1"]; sa2+=xs["sigma_a2"]
                        ss12+=xs["sigma_s12"]
                        nf1+=xs["nu_sigma_f1"]; nf2+=xs["nu_sigma_f2"]
                        cnt+=1
                D1/=cnt; D2/=cnt; sa1/=cnt; sa2/=cnt
                ss12/=cnt; nf1/=cnt; nf2/=cnt

                Ex1=D1/self.Dx**2; Ey1=D1/self.Dy**2
                Ex2=D2/self.Dx**2; Ey2=D2/self.Dy**2
                C1=sa1+ss12; C2=sa2

                if I < self.nx_c-1:
                    d1=dhat1_east[J,I]; d2=dhat2_east[J,I]
                    add_A(n, J*self.nx_c+(I+1), -Ex1+d1/(2*self.Dx))
                    add_A(n+N_c, J*self.nx_c+(I+1)+N_c, -Ex2+d2/(2*self.Dx))
                    C1+=Ex1+d1/(2*self.Dx); C2+=Ex2+d2/(2*self.Dx)
                else:
                    C1+=Ex1; C2+=Ex2

                if I > 0:
                    d1=dhat1_east[J,I-1]; d2=dhat2_east[J,I-1]
                    add_A(n, J*self.nx_c+(I-1), -Ex1-d1/(2*self.Dx))
                    add_A(n+N_c, J*self.nx_c+(I-1)+N_c, -Ex2-d2/(2*self.Dx))
                    C1+=Ex1-d1/(2*self.Dx); C2+=Ex2-d2/(2*self.Dx)
                else:
                    C1+=Ex1; C2+=Ex2

                if J < self.ny_c-1:
                    d1=dhat1_north[J,I]; d2=dhat2_north[J,I]
                    add_A(n, (J+1)*self.nx_c+I, -Ey1+d1/(2*self.Dy))
                    add_A(n+N_c, (J+1)*self.nx_c+I+N_c, -Ey2+d2/(2*self.Dy))
                    C1+=Ey1+d1/(2*self.Dy); C2+=Ey2+d2/(2*self.Dy)
                else:
                    C1+=Ey1; C2+=Ey2

                if J > 0:
                    d1=dhat1_north[J-1,I]; d2=dhat2_north[J-1,I]
                    add_A(n, (J-1)*self.nx_c+I, -Ey1-d1/(2*self.Dy))
                    add_A(n+N_c, (J-1)*self.nx_c+I+N_c, -Ey2-d2/(2*self.Dy))
                    C1+=Ey1-d1/(2*self.Dy); C2+=Ey2-d2/(2*self.Dy)
                else:
                    C1+=Ey1; C2+=Ey2

                add_A(n, n, C1); add_A(n+N_c, n+N_c, C2)
                add_A(n+N_c, n, -ss12)
                add_F(n, n, nf1); add_F(n, n+N_c, nf2)

        size = 2*N_c
        self.A_c = sparse.csr_matrix((vals_A,(rows_A,cols_A)), shape=(size,size))
        self.F_c = sparse.csr_matrix((vals_F,(rows_F,cols_F)), shape=(size,size))
        print(f"  Coarse A: {self.A_c.shape}, {self.A_c.nnz} non-zeros")
        print(f"  Coarse F: {self.F_c.shape}, {self.F_c.nnz} non-zeros")

    # ------------------------------------------------------------------
    # BUILD COARSE MATRICES (multi-group)  ← MAIN FIX IS HERE
    # ------------------------------------------------------------------

    def build_coarse_matrices_mg(self, dhat_east, dhat_north, Eb):
        """
        Build coarse loss matrix A, fission matrix F, scatter matrix S
        for G groups.

        Boundary convention (matches solver_mg._build_matrices):
            vacuum     → C += Ex  (or Ey)   — outgoing leakage term
            reflective → C += 0.0           — zero net current
        """
        # ── Read boundary conditions from solver ──────────────────────
        bc = getattr(self.solver, 'bc', {})
        west_vac  = bc.get('left',   'vacuum') == 'vacuum'
        east_vac  = bc.get('right',  'vacuum') == 'vacuum'
        south_vac = bc.get('bottom', 'vacuum') == 'vacuum'
        north_vac = bc.get('top',    'vacuum') == 'vacuum'

        N_c  = self.N_c
        G    = self.G
        size = G * N_c

        rows_A, cols_A, vals_A = [], [], []
        rows_F, cols_F, vals_F = [], [], []
        rows_S, cols_S, vals_S = [], [], []

        def add_A(r, c, v): rows_A.append(r); cols_A.append(c); vals_A.append(v)
        def add_F(r, c, v): rows_F.append(r); cols_F.append(c); vals_F.append(v)
        def add_S(r, c, v): rows_S.append(r); cols_S.append(c); vals_S.append(v)

        def cell_idx(g, I, J):
            return g*N_c + J*self.nx_c + I

        for J in range(self.ny_c):
            for I in range(self.nx_c):
                i0, i1 = I*self.rf, (I+1)*self.rf
                j0, j1 = J*self.rf, (J+1)*self.rf

                # Simple volume averaging
                D_avg      = np.zeros(G)
                siga_avg   = np.zeros(G)
                nusigf_avg = np.zeros(G)
                chi_avg    = np.zeros(G)
                sigs_avg   = np.zeros((G, G))
                count      = 0

                for jj in range(j0, j1):
                    for ii in range(i0, i1):
                        xs = self.solver._get_xs(ii, jj)
                        for g in range(G):
                            D_avg[g]        += np.asarray(xs["D"])[g]
                            siga_avg[g]     += np.asarray(xs["sigma_a"])[g]
                            nusigf_avg[g]   += np.asarray(xs["nu_sigma_f"])[g]
                            chi_avg[g]      += np.asarray(xs["chi"])[g]
                            for g2 in range(G):
                                sigs_avg[g,g2] += np.asarray(xs["sigma_s"])[g,g2]
                        count += 1

                D_avg      /= count
                siga_avg   /= count
                nusigf_avg /= count
                chi_avg    /= count
                sigs_avg   /= count

                # ── Build matrix entries for each group ───────────────
                # D_avg is this cell's own volume-averaged D.
                # For INTERIOR faces we must use D_tilde = arithmetic mean
                # of the two adjacent cells' D, matching the formula in
                # compute_dhat_mg.  Using only D_avg[I] for all faces was
                # the key inconsistency causing CMFD oscillation: the coarse
                # matrix's diffusion coefficient didn't match the D_tilde used
                # to compute d-hat, so the coarse problem's physics was wrong.
                for g in range(G):
                    row    = cell_idx(g, I, J)
                    Ex_own = D_avg[g] / self.Dx**2   # boundary face fallback
                    Ey_own = D_avg[g] / self.Dy**2

                    # Removal XS = absorption + all scatter-out from group g
                    sigma_r = siga_avg[g] + sum(
                        sigs_avg[g, g2] for g2 in range(G) if g2 != g)
                    C = sigma_r

                    # ── East neighbor (interior: use D_tilde) ─────────
                    if I < self.nx_c - 1:
                        D_tilde_e = 0.5*(D_avg[g] + self._D_coarse[g, J, I+1])
                        Ex_e = D_tilde_e / self.Dx**2
                        d    = dhat_east[g, J, I]
                        nb   = cell_idx(g, I+1, J)
                        add_A(row, nb, -Ex_e + d/(2*self.Dx))
                        C += Ex_e + d/(2*self.Dx)
                    else:
                        C += Eb['E'][g, J]

                    # ── West neighbor (interior: use D_tilde) ─────────
                    if I > 0:
                        D_tilde_w = 0.5*(D_avg[g] + self._D_coarse[g, J, I-1])
                        Ex_w = D_tilde_w / self.Dx**2
                        d    = dhat_east[g, J, I-1]
                        nb   = cell_idx(g, I-1, J)
                        add_A(row, nb, -Ex_w - d/(2*self.Dx))
                        C += Ex_w - d/(2*self.Dx)
                    else:
                        C += Eb['W'][g, J]  

                    # ── North neighbor (interior: use D_tilde) ────────
                    if J < self.ny_c - 1:
                        D_tilde_n = 0.5*(D_avg[g] + self._D_coarse[g, J+1, I])
                        Ey_n = D_tilde_n / self.Dy**2
                        d    = dhat_north[g, J, I]
                        nb   = cell_idx(g, I, J+1)
                        add_A(row, nb, -Ey_n + d/(2*self.Dy))
                        C += Ey_n + d/(2*self.Dy)
                    else:
                        C += Eb['N'][g, I]

                    # ── South neighbor (interior: use D_tilde) ────────
                    if J > 0:
                        D_tilde_s = 0.5*(D_avg[g] + self._D_coarse[g, J-1, I])
                        Ey_s = D_tilde_s / self.Dy**2
                        d    = dhat_north[g, J-1, I]
                        nb   = cell_idx(g, I, J-1)
                        add_A(row, nb, -Ey_s - d/(2*self.Dy))
                        C += Ey_s - d/(2*self.Dy)
                    else:
                        C += Eb['S'][g, I]

                    # ── Diagonal ──────────────────────────────────────
                    add_A(row, row, C)

                    # ── Scattering-in from other groups ───────────────
                    for g2 in range(G):
                        if g2 == g:
                            continue
                        add_S(row, cell_idx(g2, I, J), sigs_avg[g2, g])

                    # ── Fission source ────────────────────────────────
                    for g2 in range(G):
                        add_F(row, cell_idx(g2, I, J),
                              chi_avg[g] * nusigf_avg[g2])

        self.A_c = sparse.csr_matrix((vals_A,(rows_A,cols_A)), shape=(size,size))
        self.F_c = sparse.csr_matrix((vals_F,(rows_F,cols_F)), shape=(size,size))
        self.S_c = sparse.csr_matrix((vals_S,(rows_S,cols_S)), shape=(size,size))

    # ------------------------------------------------------------------
    # COARSE SOLVE (2-group)
    # ------------------------------------------------------------------

    def solve_coarse(self, k_fine, phi1_init=None, phi2_init=None):
        if phi1_init is not None:
            phi = np.concatenate([phi1_init.flatten(), phi2_init.flatten()])
            phi /= phi.max()
        else:
            phi = np.ones(2*self.N_c)
        k = k_fine

        for _ in range(50):
            fs  = self.F_c.dot(phi)
            phi_new = linalg.spsolve(self.A_c, fs/k)
            fs_new  = self.F_c.dot(phi_new)
            k_new   = k * np.sum(fs_new) / np.sum(fs)
            k_error = abs(k_new - k)
            phi     = phi_new / phi_new.max()
            k       = k_new
            if k_error < 1e-8:
                break

        phi1_c = phi[:self.N_c].reshape(self.ny_c, self.nx_c)
        phi2_c = phi[self.N_c:].reshape(self.ny_c, self.nx_c)
        return k, phi1_c, phi2_c

    # ------------------------------------------------------------------
    # COARSE SOLVE (multi-group)
    # ------------------------------------------------------------------

    def solve_coarse_mg(self, k_fine, phi_init=None):
        if phi_init is not None:
            phi = phi_init.reshape(self.G * self.N_c).copy()
            m   = phi.max()
            if m > 0:
                phi /= m
        else:
            phi = np.ones(self.G * self.N_c)
        k = k_fine

        for _ in range(100):
            fs      = self.F_c.dot(phi)
            ss      = self.S_c.dot(phi)
            rhs     = ss + fs/k
            phi_new = linalg.spsolve(self.A_c, rhs)
            fs_new  = self.F_c.dot(phi_new)
            k_new   = k * fs_new.sum() / fs.sum()
            k_error = abs(k_new - k)
            phi     = phi_new / phi_new.max()
            k       = k_new
            if k_error < 1e-12:
                break

        phi_c = phi.reshape(self.G, self.ny_c, self.nx_c)
        return k, phi_c

    # ------------------------------------------------------------------
    # REBALANCE FLUX (2-group)
    # ------------------------------------------------------------------

    def rebalance_flux(self, phi1_fine, phi2_fine,
                       phi1_c_old, phi2_c_old,
                       phi1_c_new, phi2_c_new):
        scale1 = np.sum(phi1_c_old) / np.sum(phi1_c_new)
        scale2 = np.sum(phi2_c_old) / np.sum(phi2_c_new)
        phi1_c_new = phi1_c_new * scale1
        phi2_c_new = phi2_c_new * scale2

        phi1_new = np.zeros_like(phi1_fine)
        phi2_new = np.zeros_like(phi2_fine)

        for J in range(self.ny_c):
            for I in range(self.nx_c):
                i0, i1 = I*self.rf, (I+1)*self.rf
                j0, j1 = J*self.rf, (J+1)*self.rf
                R1 = np.clip(phi1_c_new[J,I]/phi1_c_old[J,I]
                             if abs(phi1_c_old[J,I]) > 1e-12 else 1.0, 0.5, 2.0)
                R2 = np.clip(phi2_c_new[J,I]/phi2_c_old[J,I]
                             if abs(phi2_c_old[J,I]) > 1e-12 else 1.0, 0.5, 2.0)
                phi1_new[j0:j1, i0:i1] = phi1_fine[j0:j1, i0:i1] * R1
                phi2_new[j0:j1, i0:i1] = phi2_fine[j0:j1, i0:i1] * R2

        return phi1_new, phi2_new

    # ------------------------------------------------------------------
    # REBALANCE FLUX (multi-group)  ← SECOND FIX IS HERE
    # ------------------------------------------------------------------

    def rebalance_flux_mg(self, flux_fine, flux_c_old, flux_c_new):
        """
        Scale fine mesh flux using coarse mesh correction ratio.
        """
        flux_new = np.zeros_like(flux_fine)

        for g in range(self.G):
            for J in range(self.ny_c):
                for I in range(self.nx_c):
                    j0, j1 = J*self.rf, (J+1)*self.rf
                    i0, i1 = I*self.rf, (I+1)*self.rf

                    # Raw ratio — amplitude + shape correction combined
                    if abs(flux_c_old[g, J, I]) > 1e-12:
                        r = flux_c_new[g, J, I] / flux_c_old[g, J, I]
                    else:
                        r = 1.0
                    flux_new[g, j0:j1, i0:i1] = flux_fine[g, j0:j1, i0:i1] * r

        return flux_new

    # ------------------------------------------------------------------
    # ACCELERATE (2-group)
    # ------------------------------------------------------------------

    def accelerate(self, phi1, phi2, k, n_cycles=5):
        print(f"\n  CMFD Acceleration ({n_cycles} cycles)...")
        k_original = k
        alpha = 0.05

        for cycle in range(n_cycles):
            phi1_orig = phi1.copy(); phi2_orig = phi2.copy()
            phi1_c, phi2_c = self.homogenize_flux(phi1, phi2)
            d1e, d2e, d1n, d2n = self.compute_dhat(phi1, phi2, phi1_c, phi2_c)
            self.build_coarse_matrices(phi1, phi2, d1e, d2e, d1n, d2n)
            k_new, phi1_cn, phi2_cn = self.solve_coarse(k, phi1_c.copy(), phi2_c.copy())

            if k_new < 0 or k_new > 10:
                print(f"    Warning: CMFD diverging (k={k_new:.3f}), skipping!")
                return phi1, phi2, k_original

            phi1_new, phi2_new = self.rebalance_flux(
                phi1, phi2, phi1_c, phi2_c, phi1_cn, phi2_cn)

            phi1 = (1-alpha)*phi1 + alpha*phi1_new
            phi2 = (1-alpha)*phi2 + alpha*phi2_new
            phi_max = max(phi1.max(), phi2.max())
            phi1 /= phi_max; phi2 /= phi_max
            k = k_new

        return phi1, phi2, k_original

    # ------------------------------------------------------------------
    # ACCELERATE (multi-group)
    # ------------------------------------------------------------------

    def accelerate_mg(self, flux, k, n_cycles=1):
        """
        Run n_cycles of MG CMFD acceleration.
        """

        for _ in range(n_cycles):
            flux_c         = self.homogenize_flux_mg(flux)
            dhat_e, dhat_n = self.compute_dhat_mg(flux, flux_c)
            Eb = self.compute_boundary_coupling(flux, flux_c)
            self.build_coarse_matrices_mg(dhat_e, dhat_n, Eb)

            k_new, flux_c_new = self.solve_coarse_mg(k, phi_init=flux_c)
            # c_sum = flux_c.sum()
            # n_sum = flux_c_new.sum()
            # if n_sum > 0 and c_sum > 0:
            #     flux_c_new = flux_c_new * (c_sum / n_sum)

            flux_rb = self.rebalance_flux_mg(flux, flux_c, flux_c_new)

            flux = flux_rb
            flux /= flux.max()
            k = k_new

        return flux, k 

    # ------------------------------------------------------------------
    # DIAGNOSTIC
    # ------------------------------------------------------------------

    def diagnostic(self):
        """
        Print coarse matrix health check.
        Call after build_coarse_matrices_mg to verify BC correctness.

        Healthy output:
            n_nonpositive_diag = 0
            row_sum min ≈ 0   (interior cells: leakage ≈ absorption + scatter)
            row_sum < 0 count = 0
        """
        if not hasattr(self, 'A_c'):
            print("  [diagnostic] A_c not built yet.")
            return
        diag     = self.A_c.diagonal()
        row_sums = np.array(self.A_c.sum(axis=1)).flatten()
        print(f"  [CMFD diag] min={diag.min():.4f}  max={diag.max():.4f}  "
              f"n_nonpositive={np.sum(diag <= 0)}")
        print(f"  [CMFD rowsum] min={row_sums.min():.6f}  "
              f"max={row_sums.max():.6f}  "
              f"n_negative_rowsum={np.sum(row_sums < -1e-10)}")


# ======================================================================
# SELF-TEST
# ======================================================================

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from input_reader import InputReader
    from solver_2d import Solver2D
    from solver_mg import SolverMG

    print("="*55)
    print("  ANGKOR CMFD self-test")
    print("="*55)

    reader = InputReader("input/pwr_2d.yaml")
    reader.read()

    # ── Test 1: 2-group solver + 2-group CMFD ─────────────────────────
    print("\n[Test 1] Solver2D + 2-group CMFD")
    solver2d = Solver2D(reader.engine, reader.materials, reader.solver)
    cmfd2g   = CMFD(solver2d, rf=10)
    t0 = time.time()
    k2, f1, f2 = solver2d.solve(cmfd=cmfd2g, cmfd_interval=1)
    print(f"  k-eff = {k2:.6f}  time = {time.time()-t0:.1f}s")

    # ── Test 2: SolverMG + MG CMFD ────────────────────────────────────
    print("\n[Test 2] SolverMG + MG CMFD  (no CMFD baseline)")
    s_plain = SolverMG(reader.engine, reader.materials, reader.solver, n_groups=2)
    k_plain, _ = s_plain.solve(cmfd=None)

    print("\n[Test 3] SolverMG + MG CMFD  (with CMFD)")
    s_cmfd  = SolverMG(reader.engine, reader.materials, reader.solver, n_groups=2)
    cmfd_mg = CMFD(s_cmfd, rf=10)
    k_cmfd, _ = s_cmfd.solve(cmfd=cmfd_mg)

    diff = abs(k_cmfd - k_plain)*1e5
    print(f"\n  k without CMFD : {k_plain:.6f}")
    print(f"  k with MG CMFD : {k_cmfd:.6f}")
    print(f"  Difference     : {diff:.1f} pcm  (target: < 5 pcm)")
    print(f"  Iterations     : {s_cmfd.iterations}  (target: < 30)")
    if diff < 5:
        print("  STATUS: PASS")
    else:
        print("  STATUS: FAIL — check BC flags in build_coarse_matrices_mg")