"""
Field-level utilities for NS2D post-processing.

This module provides spectral transforms between velocity, vorticity,
and streamfunction, for use on snapshot data written by the solver.

All functions operate on full grid-space arrays (NumPy ndarrays) and
are intended for offline analysis, not for in-situ use in the solver.
"""

import numpy as np
import dedalus.public as d3
from mpi4py import MPI


# Cache for offline LBVP solvers keyed by (Nx, Ny, Lx, Ly, dealias, dtype)
_PSI_LBVP_CACHE = {}


def velocity_to_vorticity(ux_grid, uy_grid, Lx, Ly):
    """
    Compute vorticity from velocity using FFTs (offline utility).

    For 2D incompressible flow:
        ω = ∂v/∂x - ∂u/∂y  =>  ω̂ = i k_x v̂ - i k_y û

    Args:
        ux_grid (ndarray): x-velocity in physical space (Nx, Ny)
        uy_grid (ndarray): y-velocity in physical space (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        ndarray: Vorticity in physical space (Nx, Ny)
    """
    Nx, Ny = ux_grid.shape

    # Wavenumber grids for rfft2 layout
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    # Transform to spectral space
    ux_hat = np.fft.rfft2(ux_grid)
    uy_hat = np.fft.rfft2(uy_grid)

    # Compute vorticity: ω̂ = i k_x v̂ - i k_y û
    omega_hat = 1j * KX * uy_hat - 1j * KY * ux_hat

    # Transform back to physical space
    omega_grid = np.fft.irfft2(omega_hat, s=(Nx, Ny))
    return omega_grid


def streamfunction_to_velocity(psi_grid, Lx, Ly):
    """
    Convert streamfunction to velocity using FFTs (offline utility).

    For 2D incompressible flow:
        u = ∂ψ/∂y  =>  û = i k_y ψ̂
        v = -∂ψ/∂x =>  v̂ = -i k_x ψ̂

    Args:
        psi_grid (ndarray): Streamfunction in physical space (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        tuple: (ux_grid, uy_grid) velocity components in physical space
    """
    Nx, Ny = psi_grid.shape

    # Wavenumber grids for rfft2 layout
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    # Transform to spectral space
    psi_hat = np.fft.rfft2(psi_grid)

    # Compute velocity: û = i k_y ψ̂,  v̂ = -i k_x ψ̂
    ux_hat = 1j * KY * psi_hat
    uy_hat = -1j * KX * psi_hat

    # Transform back to physical space
    ux_grid = np.fft.irfft2(ux_hat, s=(Nx, Ny))
    uy_grid = np.fft.irfft2(uy_hat, s=(Nx, Ny))
    return ux_grid, uy_grid


def vorticity_to_streamfunction(omega_grid, Lx, Ly, dealias=1.0, dtype=np.float64):
    """
    Compute streamfunction from vorticity using a Dedalus LBVP (offline).

    Solves the Poisson problem:
        -∇²ψ = ω,    with  ∫ψ dA = 0

    on a 2D RealFourier domain using Dedalus. This mirrors the formulation
    used in the in-situ solver, but operates entirely on snapshot data.

    Args:
        omega_grid (ndarray): Vorticity in physical space (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y
        dealias (float): Dealiasing factor for RealFourier bases (default: 1.0)
        dtype: NumPy dtype for Dedalus fields (default: np.float64)

    Returns:
        ndarray: Streamfunction in physical space (Nx, Ny)
    """
    Nx, Ny = omega_grid.shape

    # Build or retrieve cached single-rank Dedalus LBVP for this grid/domain.
    key = (Nx, Ny, float(Lx), float(Ly), float(dealias), np.dtype(dtype).str)
    cache_entry = _PSI_LBVP_CACHE.get(key)

    if cache_entry is None:
        coords = d3.CartesianCoordinates("x", "y")
        dist = d3.Distributor(coords, dtype=dtype, comm=MPI.COMM_SELF)
        xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx), dealias=dealias)
        ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0, Ly), dealias=dealias)

        psi = dist.Field(name="psi", bases=(xbasis, ybasis))
        tau_psi = dist.Field(name="tau_psi")
        omega = dist.Field(name="omega", bases=(xbasis, ybasis))

        problem = d3.LBVP(
            [psi, tau_psi],
            namespace={"psi": psi, "tau_psi": tau_psi, "omega": omega, "d3": d3},
        )
        problem.add_equation("-lap(psi) + tau_psi = omega")
        problem.add_equation("integ(psi) = 0")
        solver = problem.build_solver()

        cache_entry = (psi, tau_psi, omega, solver)
        _PSI_LBVP_CACHE[key] = cache_entry

    psi, tau_psi, omega, solver = cache_entry

    # Assign vorticity data on the full grid
    omega.change_scales(1)
    omega["g"] = omega_grid.astype(dtype, copy=False)

    # Solve LBVP: -lap(psi) + tau_psi = omega,  integ(psi) = 0
    solver.solve()

    psi.change_scales(1)
    psi_grid = np.array(psi["g"], copy=True)
    return psi_grid


