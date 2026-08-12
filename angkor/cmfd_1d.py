import numpy as np 


def build_fd_matrices(N, h, D, sigma_a, nu_sigma_f):
    """
    Build the lost matrix A and fission matrix M
    for 1-group 1D finite difference diffusion.

    Args:
        N (int): numer of find mesh cells
        h (float): cell width (cm)
        D (array shape (N,)): diffusion coefficient per cel
        sigma_a (array shape (N,)): absorption xs per cell (cm^-1)
        nu_sigma_f (array shape (N,)): nu*fission xs per cell (cm^-1)
    
    boundary conditions: zero flux (phi=0) at both ends.
    Returns
    A : (N,N) ndarray: lost matrix
    M : (N,N) ndarray: fission matrix
    """
    
    A = np.zeros((N,N))
    M = np.zeros((N,N))
    
    for i in range(N):
        if i == 0:
            D_left = D[0]
        else:
            D_left = 2*D[i-1]*D[i] / (D[i-1] + D[i])

        if i== N-1: D_right = D[N-1]
        else:       D_right = 2*D[i]*D[i+1] / (D[i] + D[i+1])
        
        # Diagonal of A 
        A[i,i] = D_left/h + D_right/h+sigma_a[i]*h
        
        # off diagonal of A 
        if i>0:     A[i,i-1] = -D_left/h 
        if i< N-1:  A[i,i+1] = -D_right/h 
        
        # fission matrix 
        M[i,i] = nu_sigma_f[i]*h 
    return A, M


# Homogeneous slab, should give a symmetric tridiagonal A
N, h = 5, 1.0
D         = np.ones(N) * 0.333
sigma_a   = np.ones(N) * 0.02
nu_sigma_f = np.ones(N) * 0.025

A, M = build_fd_matrices(N, h, D, sigma_a, nu_sigma_f)
print(np.round(A, 4))
print(np.round(M, 4))
    