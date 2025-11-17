"""
Standalone spectral analysis functions for 2D turbulence.

This module provides NumPy-based implementations of spectral diagnostics
that do not require Dedalus or mpi4py. These functions work with velocity
fields in physical space (regular NumPy arrays).

Functions:
- compute_spectra: Compute energy and enstrophy spectra E(k), Z(k)
- compute_energy_flux: Compute spectral energy transfer and flux
- compute_enstrophy_flux: Compute spectral enstrophy transfer and flux

All functions use shell-averaging in Fourier space for isotropic analysis.
"""

import numpy as np


def compute_spectra(ux_grid, uy_grid, Lx, Ly):
    """
    Compute isotropic 1D energy and enstrophy spectra from 2D velocity field.

    Uses shell-averaging in Fourier space to compute spectra as a function
    of wavenumber magnitude |k|.

    Args:
        ux_grid (ndarray): x-velocity in physical space (Nx, Ny)
        uy_grid (ndarray): y-velocity in physical space (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        tuple: (k_bins, E_k, Z_k)
            - k_bins: Physical wavenumber bins (rad/length)
            - E_k: Energy spectrum E(k) = 0.5 <|û|²>_shell
            - Z_k: Enstrophy spectrum Z(k) = <|ω̂|²>_shell

    Notes:
        - Assumes Lx ≈ Ly for isotropic shell averaging
        - Accounts for rfft symmetry factors
        - Shell index n corresponds to physical wavenumber n*k0 where k0=2π/L
    """
    Nx, Ny = ux_grid.shape
    N = Nx * Ny
    assert abs(Lx - Ly) < 1e-12, "Isotropic shell binning requires Lx ≈ Ly"
    k0 = 2 * np.pi / Lx

    # Transform to spectral space
    uxh = np.fft.rfft2(ux_grid)
    uyh = np.fft.rfft2(uy_grid)

    # Energy per mode (normalised)
    area = Lx * Ly
    E_mode = 0.5 * (np.abs(uxh)**2 + np.abs(uyh)**2) / (N * N)

    # Vorticity in spectral space
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    omegah = 1j * (KX * uyh - KY * uxh)
    Z_mode = (np.abs(omegah)**2) / (N * N)

    # rfft symmetry weight: double ky>0 interior modes
    weight = 2.0 * np.ones_like(E_mode)
    weight[:, 0] = 1.0  # ky=0 is not doubled
    if Ny % 2 == 0:
        weight[:, -1] = 1.0  # Nyquist is real-valued

    E_mode *= weight * area
    Z_mode *= weight * area

    # Shell indices (integer radius in index space)
    ix = np.fft.fftfreq(Nx, d=1.0 / Nx)
    iy = np.arange(0, Ny // 2 + 1)
    IX, IY = np.meshgrid(ix, iy, indexing='ij')
    shell_idx = np.floor(np.sqrt(IX**2 + IY**2)).astype(int)

    # Bin into shells
    mmax = shell_idx.max()
    Ek = np.bincount(shell_idx.ravel(), weights=E_mode.ravel(), minlength=mmax + 1)
    Zk = np.bincount(shell_idx.ravel(), weights=Z_mode.ravel(), minlength=mmax + 1)

    k_bins = np.arange(mmax + 1) * k0
    return k_bins, Ek, Zk


def compute_energy_flux(ux_grid, uy_grid, Lx, Ly):
    """
    Compute spectral energy flux using pseudo-spectral method.

    The energy transfer T(k) measures how energy is redistributed among
    wavenumbers by the nonlinear advective term. The cumulative flux
    Π(k) = -∑_{k'≤k} T(k') represents the energy cascade rate.

    Args:
        ux_grid (ndarray): x-velocity in physical space (Nx, Ny)
        uy_grid (ndarray): y-velocity in physical space (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        tuple: (k_bins, T_shell, Pi_shell)
            - k_bins: Physical wavenumber bins
            - T_shell: Shell-summed energy transfer T(k)
            - Pi_shell: Cumulative energy flux Π(k) = -cumsum(T)
                       (positive for forward/downscale cascade)

    Notes:
        - Uses 2/3 rectangular dealiasing filter
        - T(k) computed from Re[û* · N̂] where N = (u·∇)u
        - Π(k) > 0 indicates downscale (forward) energy cascade
    """
    Nx, Ny = ux_grid.shape
    assert abs(Lx - Ly) < 1e-12, "Isotropic shell binning requires Lx ≈ Ly"
    N = Nx * Ny
    k0 = 2 * np.pi / Lx

    # Wavenumber grids
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)[:, None]
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)[None, :]

    # Transform to spectral space
    uxh = np.fft.rfft2(ux_grid)
    uyh = np.fft.rfft2(uy_grid)

    # 2/3 rectangular dealiasing filter
    kx_max = (Nx // 3) * (2 * np.pi / Lx)
    ky_max = (Ny // 3) * (2 * np.pi / Ly)
    dealias_rect = (np.abs(kx) <= kx_max) & (np.abs(ky) <= ky_max)

    # Apply filter
    uxh_f = uxh * dealias_rect
    uyh_f = uyh * dealias_rect

    # Compute derivatives in spectral space
    dudx_h = 1j * kx * uxh_f
    dudy_h = 1j * ky * uxh_f
    dvdx_h = 1j * kx * uyh_f
    dvdy_h = 1j * ky * uyh_f

    # Transform to physical space
    dudx = np.fft.irfft2(dudx_h, s=(Nx, Ny))
    dudy = np.fft.irfft2(dudy_h, s=(Nx, Ny))
    dvdx = np.fft.irfft2(dvdx_h, s=(Nx, Ny))
    dvdy = np.fft.irfft2(dvdy_h, s=(Nx, Ny))

    ux_f = np.fft.irfft2(uxh_f, s=(Nx, Ny))
    uy_f = np.fft.irfft2(uyh_f, s=(Nx, Ny))

    # Nonlinear term: N = (u·∇)u
    Nx_grid = ux_f * dudx + uy_f * dudy
    Ny_grid = ux_f * dvdx + uy_f * dvdy

    # Back to spectral space
    Nxh = np.fft.rfft2(Nx_grid)
    Nyh = np.fft.rfft2(Ny_grid)

    # rfft symmetry weights
    weight = np.ones_like(uxh, dtype=np.float64)
    if weight.shape[1] > 1:
        weight[:, 1:-1] = 2.0
    weight[:, 0] = 1.0
    if Ny % 2 == 0:
        weight[:, -1] = 1.0

    # Energy transfer per mode: T = Re[û* · N̂]
    T_mode = np.real(np.conj(uxh) * Nxh + np.conj(uyh) * Nyh) / (N * N)
    area = Lx * Ly
    T_mode *= weight * area

    # Shell binning
    ix = np.fft.fftfreq(Nx, d=1.0 / Nx)[:, None]
    iy = np.arange(0, Ny // 2 + 1)[None, :]
    shell_idx = np.floor(np.sqrt(ix * ix + iy * iy)).astype(int)

    mmax = int(shell_idx.max())
    T_shell = np.bincount(shell_idx.ravel(), weights=T_mode.ravel(), minlength=mmax + 1)

    k_bins = np.arange(mmax + 1) * k0
    Pi_shell = -np.cumsum(T_shell)  # Cumulative flux

    return k_bins, T_shell, Pi_shell


def compute_enstrophy_flux(ux_grid, uy_grid, Lx, Ly):
    """
    Compute spectral enstrophy flux using pseudo-spectral method.

    The enstrophy transfer TΩ(k) measures how enstrophy is redistributed by
    the nonlinear term (u·∇)ω. The cumulative enstrophy flux ΠΩ(k) represents
    the enstrophy cascade rate.

    Args:
        ux_grid (ndarray): x-velocity in physical space (Nx, Ny)
        uy_grid (ndarray): y-velocity in physical space (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        tuple: (k_bins, T_shell, Pi_shell)
            - k_bins: Physical wavenumber bins
            - T_shell: Shell-summed enstrophy transfer TΩ(k)
            - Pi_shell: Cumulative enstrophy flux ΠΩ(k) = -cumsum(TΩ)
                       (positive for downscale/direct enstrophy cascade)

    Notes:
        - Uses 2/3 rectangular dealiasing filter
        - TΩ(k) computed from Re[ω̂* · N̂ω] where Nω = (u·∇)ω
        - In 2D turbulence, ΠΩ > 0 indicates downscale enstrophy cascade
    """
    Nx, Ny = ux_grid.shape
    assert abs(Lx - Ly) < 1e-12, "Isotropic shell binning requires Lx ≈ Ly"
    N = Nx * Ny
    k0 = 2 * np.pi / Lx

    # Spectral wavenumbers (rfft packing in y)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)[:, None]
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)[None, :]

    # FFT of velocity
    uxh = np.fft.rfft2(ux_grid)
    uyh = np.fft.rfft2(uy_grid)

    # 2/3 rectangular de-alias filter
    kx_max = (Nx // 3) * (2 * np.pi / Lx)
    ky_max = (Ny // 3) * (2 * np.pi / Ly)
    dealias_rect = (np.abs(kx) <= kx_max) & (np.abs(ky) <= ky_max)

    uxh_f = uxh * dealias_rect
    uyh_f = uyh * dealias_rect

    # Vorticity from filtered velocity: ω̂ = i(kx û_y - ky û_x)
    omegah = 1j * (kx * uyh_f - ky * uxh_f)

    # Gradients of ω (spectral) & inverse FFT to grid
    domegadx_h = 1j * kx * omegah
    domegady_h = 1j * ky * omegah

    ux_f = np.fft.irfft2(uxh_f, s=(Nx, Ny))
    uy_f = np.fft.irfft2(uyh_f, s=(Nx, Ny))
    dωdx = np.fft.irfft2(domegadx_h, s=(Nx, Ny))
    dωdy = np.fft.irfft2(domegady_h, s=(Nx, Ny))

    # Nonlinear term N_ω = (u·∇)ω in grid space, back to spectral
    Nomega_grid = ux_f * dωdx + uy_f * dωdy
    Nomegah = np.fft.rfft2(Nomega_grid)

    # rfft symmetry weight
    weight = np.ones_like(uxh, dtype=np.float64)
    if weight.shape[1] > 1:
        weight[:, 1:-1] = 2.0
    weight[:, 0] = 1.0
    if Ny % 2 == 0:
        weight[:, -1] = 1.0

    # Per-mode enstrophy transfer: TΩ = Re[ω̂* · N̂ω]
    T_mode = np.real(np.conj(omegah) * Nomegah) / (N * N)
    area = Lx * Ly
    T_mode *= weight * area

    # Shell binning (integer radius in index space)
    ix = np.fft.fftfreq(Nx, d=1.0 / Nx)[:, None]
    iy = np.arange(0, Ny // 2 + 1)[None, :]
    shell_idx = np.floor(np.sqrt(ix * ix + iy * iy)).astype(int)
    mmax = int(shell_idx.max())

    T_shell = np.bincount(shell_idx.ravel(), weights=T_mode.ravel(), minlength=mmax + 1)
    k_bins = np.arange(mmax + 1) * k0
    Pi_shell = -np.cumsum(T_shell)  # Cumulative enstrophy flux

    return k_bins, T_shell, Pi_shell
