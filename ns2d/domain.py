"""
Domain setup, grid construction, and initial conditions.

This module provides utilities for setting up the computational domain,
constructing wavenumber grids, and generating initial conditions for
2D Navier-Stokes simulations.
"""

import numpy as np
import dedalus.public as d3
from mpi4py import MPI


def build_domain(Nx, Ny, Lx, Ly, dealias, dtype, mesh=None):
    """
    Build a 2D periodic Fourier domain for Dedalus.

    Args:
        Nx (int): Number of grid points in x direction
        Ny (int): Number of grid points in y direction
        Lx (float): Domain length in x direction
        Ly (float): Domain length in y direction
        dealias (float): Dealiasing factor (typically 1.5 or higher)
        dtype: NumPy data type (e.g., np.float64, np.float32)
        mesh (tuple, optional): MPI process mesh shape (procs_x, procs_y).
            If None, uses 1D decomposition.

    Returns:
        tuple: (coords, dist, xbasis, ybasis, x, y)
            - coords: Dedalus coordinate system
            - dist: Dedalus distributor (handles MPI parallelization)
            - xbasis: Fourier basis in x direction
            - ybasis: Fourier basis in y direction
            - x: Local x grid points
            - y: Local y grid points
    """
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype, comm=MPI.COMM_WORLD, mesh=mesh)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
    x, y = dist.local_grids(xbasis, ybasis)
    return coords, dist, xbasis, ybasis, x, y


def wavenumbers(Nx, Ny, Lx, Ly):
    """
    Compute wavenumber grids for 2D FFT (rfft2 layout).

    Args:
        Nx (int): Number of grid points in x
        Ny (int): Number of grid points in y
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        tuple: (kx, ky, KX, KY, K2, K)
            - kx: 1D array of x wavenumbers (Nx,)
            - ky: 1D array of y wavenumbers (Ny//2+1,) [rfft layout]
            - KX: 2D meshgrid of x wavenumbers (Nx, Ny//2+1)
            - KY: 2D meshgrid of y wavenumbers (Nx, Ny//2+1)
            - K2: Squared magnitude |k|² (Nx, Ny//2+1)
            - K: Magnitude |k| (Nx, Ny//2+1)
    """
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)       # (Nx,)
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)      # (Ny//2+1,)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K = np.sqrt(K2)
    return kx, ky, KX, KY, K2, K


def initial_condition(rng, K2, Ny, alpha=49.0, power=2.5, scale=7.0**1.5):
    """
    Generate a random initial vorticity field in spectral space.

    The initial condition is drawn from a complex Gaussian distribution with
    variance Var[w_hat(k)] = scale * (|k|² + alpha)^(-power). This produces
    smooth, energetic initial conditions suitable for turbulence simulations.

    Args:
        rng: NumPy random generator instance
        K2 (ndarray): Squared wavenumber magnitude array (Nx, Ny//2+1)
        Ny (int): Number of grid points in y direction
        alpha (float): Spectral roll-off parameter (default: 49.0)
        power (float): Power-law exponent for spectrum (default: 2.5)
        scale (float): Overall amplitude scaling (default: 7.0**1.5)

    Returns:
        ndarray: Complex vorticity field in spectral space (Nx, Ny//2+1)
            with appropriate rfft reality conditions enforced.
    """
    # Variance as a function of wavenumber
    var_k = scale * np.power(K2 + alpha, -power, where=(K2 + alpha) > 0)
    var_k = np.maximum(var_k, 0.0)

    # Draw complex Gaussian samples
    xi_real = rng.standard_normal(var_k.shape)
    xi_imag = rng.standard_normal(var_k.shape)
    w_hat = (xi_real + 1j * xi_imag) * np.sqrt(var_k / 2.0)

    # Enforce zero mean (k=0 mode)
    w_hat[0, 0] = 0.0

    # Enforce Nyquist reality condition for even Ny
    if Ny % 2 == 0:
        w_hat[:, -1] = np.real(w_hat[:, -1]) + 0j

    return w_hat


def vorticity_to_velocity(w_hat, KX, KY, K2, Nx, Ny):
    """
    Convert vorticity field to velocity field via streamfunction.

    For 2D incompressible flow:
        ω = ∇²ψ  =>  ψ = -ω / |k|²
        u = ∇⊥ψ  =>  û = i k⊥ ψ̂

    Args:
        w_hat (ndarray): Vorticity in spectral space (Nx, Ny//2+1)
        KX (ndarray): x wavenumber meshgrid
        KY (ndarray): y wavenumber meshgrid
        K2 (ndarray): Squared wavenumber magnitude
        Nx (int): Grid size in x
        Ny (int): Grid size in y

    Returns:
        tuple: (ux_grid, uy_grid, psi_grid)
            - ux_grid: x-velocity in physical space (Nx, Ny)
            - uy_grid: y-velocity in physical space (Nx, Ny)
            - psi_grid: streamfunction in physical space (Nx, Ny)
    """
    # Compute streamfunction: ψ̂ = -ω̂ / k²
    with np.errstate(divide='ignore', invalid='ignore'):
        psi_hat = -w_hat / np.where(K2 == 0.0, 1.0, K2)
    psi_hat[0, 0] = 0.0

    # Compute velocity: û = ik_y ψ̂,  v̂ = -ik_x ψ̂
    uxh = 1j * KY * psi_hat
    uyh = -1j * KX * psi_hat

    # Transform to physical space
    psi_grid = np.fft.irfft2(psi_hat, s=(Nx, Ny))
    ux_grid = np.fft.irfft2(uxh, s=(Nx, Ny))
    uy_grid = np.fft.irfft2(uyh, s=(Nx, Ny))

    return ux_grid, uy_grid, psi_grid
