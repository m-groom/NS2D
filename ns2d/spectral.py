"""
Spectral analysis utilities for 2D turbulence.

This module provides functions to compute:
- Isotropic energy and enstrophy spectra
- Spectral energy flux (energy transfer and cascade)
- Spectral enstrophy flux (enstrophy transfer and cascade)

All functions use shell-averaging in Fourier space for isotropic analysis.
"""

import numpy as np
import dedalus.public as d3
from mpi4py import MPI


def _resolve_bases(u, xbasis, ybasis):
    """Infer Fourier bases from a Dedalus field when not supplied."""

    if xbasis is not None and ybasis is not None:
        return xbasis, ybasis

    domain = getattr(u, 'domain', None)
    bases = getattr(domain, 'bases', None) if domain is not None else None
    if bases is None or len(bases) < 2:
        raise ValueError("xbasis and ybasis must be provided for coefficient diagnostics.")

    xb = xbasis or bases[0]
    yb = ybasis or bases[1]
    return xb, yb


def _resolve_comm(dist, comm):
    """Choose the communicator used for reductions."""

    if comm is not None:
        return comm
    if hasattr(dist, 'comm_cart') and dist.comm_cart is not None:
        return dist.comm_cart
    return dist.comm


def _resolve_context(u, dist=None, xbasis=None, ybasis=None, comm=None):
    """Resolve distributor, bases, and communicator from a field."""

    resolved_dist = dist or getattr(u, 'dist', None)
    if resolved_dist is None:
        raise ValueError("Dedalus distributor is required for coefficient diagnostics.")

    xb, yb = _resolve_bases(u, xbasis, ybasis)
    resolved_comm = _resolve_comm(resolved_dist, comm)
    return resolved_dist, xb, yb, resolved_comm


def _prepare_shell_metadata(u, dist, xbasis, ybasis, Lx, Ly):
    """Compute Parseval weights and shell indices for RealFourier layouts."""

    if abs(Lx - Ly) > 1e-12:
        raise ValueError("Isotropic shell binning requires Lx ≈ Ly.")

    u.change_scales(1)
    u.require_coeff_space()
    coeff_layout = dist.coeff_layout
    x_slice, y_slice = coeff_layout.slices(u.domain, scales=1)

    step_x = x_slice.step or 1
    step_y = y_slice.step or 1
    x_indices = np.arange(x_slice.start, x_slice.stop, step_x, dtype=np.int64)
    y_indices = np.arange(y_slice.start, y_slice.stop, step_y, dtype=np.int64)
    nx_index = x_indices // 2
    ny_index = y_indices // 2

    wx = np.where(nx_index == 0, 1.0, 0.5)
    wy = np.where(ny_index == 0, 1.0, 0.5)
    weight = wx[:, None] * wy[None, :]

    NX, NY = np.meshgrid(nx_index, ny_index, indexing='ij')
    k0 = 2 * np.pi / Lx
    shell_idx = np.rint(np.sqrt(NX**2 + NY**2)).astype(np.int64)

    return weight, shell_idx, k0


def _shell_bincount(comm, shell_idx, values, mmax=None):
    """Sum *values* over isotropic shells with MPI reduction."""

    if shell_idx.size == 0:
        local_max = 0
    else:
        local_max = int(np.max(shell_idx))

    if mmax is None:
        if comm is not None:
            mmax = comm.allreduce(local_max, op=MPI.MAX)
        else:
            mmax = local_max

    local_bins = np.bincount(shell_idx.ravel(), weights=values.ravel(), minlength=mmax + 1)
    if comm is not None:
        global_bins = np.empty_like(local_bins)
        comm.Allreduce(local_bins, global_bins, op=MPI.SUM)
    else:
        global_bins = local_bins

    return global_bins, mmax


def _return_if_root(comm, data):
    """Return *data* on rank 0 and None elsewhere (if MPI present)."""

    if comm is None or comm.rank == 0:
        return data
    return None


def _dealias_scale(xbasis, ybasis):
    """Determine the grid scaling used for dealiasing products."""

    def _scale(val):
        if isinstance(val, (tuple, list)):
            return max(float(v) for v in val)
        return float(val)

    sx = getattr(xbasis, 'dealias', 1.0) or 1.0
    sy = getattr(ybasis, 'dealias', 1.0) or 1.0
    return max(1.0, _scale(sx), _scale(sy))


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


def compute_spectra_from_coeffs(u, dist=None, xbasis=None, ybasis=None, Lx=None, Ly=None, comm=None):
    """
    Compute spectra directly from distributed RealFourier coefficients.

    Args:
        u: Dedalus vector field (RealFourier bases)
        dist: Dedalus distributor (defaults to u.dist)
        xbasis: x-direction basis (defaults to u.domain.bases[0])
        ybasis: y-direction basis (defaults to u.domain.bases[1])
        Lx (float): Domain length in x
        Ly (float): Domain length in y
        comm: Optional MPI communicator override

    Returns:
        tuple or None: (k_bins, E_k, Z_k) on rank 0, None otherwise.
    """

    if Lx is None or Ly is None:
        raise ValueError("Lx and Ly must be provided for coefficient spectra.")

    dist, xb, yb, resolved_comm = _resolve_context(u, dist, xbasis, ybasis, comm)
    weight, shell_idx, k0 = _prepare_shell_metadata(u, dist, xb, yb, Lx, Ly)
    area = Lx * Ly

    u.change_scales(1)
    u.require_coeff_space()
    ux_c = np.asarray(u['c'][0])
    uy_c = np.asarray(u['c'][1])
    coeff_sq = np.abs(ux_c)**2 + np.abs(uy_c)**2
    energy_density = 0.5 * coeff_sq * weight * area

    u.require_grid_space()
    omega = (-d3.div(d3.skew(u))).evaluate()
    omega.change_scales(1)
    omega.require_coeff_space()
    omega_coeff = np.asarray(omega['c'])
    enstrophy_density = np.abs(omega_coeff)**2 * weight * area

    Ek, mmax = _shell_bincount(resolved_comm, shell_idx, energy_density)
    Zk, _ = _shell_bincount(resolved_comm, shell_idx, enstrophy_density, mmax=mmax)
    k_bins = np.arange(mmax + 1, dtype=np.float64) * k0

    u.require_grid_space()
    return _return_if_root(resolved_comm, (k_bins, Ek, Zk))


def compute_energy_flux_from_coeffs(u, dist=None, xbasis=None, ybasis=None, Lx=None, Ly=None, comm=None):
    """Compute spectral energy flux using Dedalus coefficient data."""

    if Lx is None or Ly is None:
        raise ValueError("Lx and Ly must be provided for coefficient flux diagnostics.")

    dist, xb, yb, resolved_comm = _resolve_context(u, dist, xbasis, ybasis, comm)
    weight, shell_idx, k0 = _prepare_shell_metadata(u, dist, xb, yb, Lx, Ly)
    area = Lx * Ly
    scale = _dealias_scale(xb, yb)

    u.change_scales(1)
    u.require_coeff_space()
    ux_orig = np.array(u['c'][0], copy=True)
    uy_orig = np.array(u['c'][1], copy=True)

    if scale > 1.0:
        u.change_scales(scale)
    u.require_grid_space()
    adv = (u @ d3.grad(u)).evaluate()
    adv.change_scales(scale)
    adv.require_grid_space()

    u.change_scales(1)
    adv.change_scales(1)
    u.require_coeff_space()
    adv.require_coeff_space()

    u['c'][0] = ux_orig
    u['c'][1] = uy_orig

    ux_c = ux_orig
    uy_c = uy_orig
    advx_c = np.asarray(adv['c'][0])
    advy_c = np.asarray(adv['c'][1])
    transfer = np.real(np.conj(ux_c) * advx_c + np.conj(uy_c) * advy_c) * weight * area

    T_shell, mmax = _shell_bincount(resolved_comm, shell_idx, transfer)
    Pi_shell = -np.cumsum(T_shell)
    k_bins = np.arange(mmax + 1, dtype=np.float64) * k0

    u.require_grid_space()
    return _return_if_root(resolved_comm, (k_bins, T_shell, Pi_shell))


def compute_enstrophy_flux_from_coeffs(u, dist=None, xbasis=None, ybasis=None, Lx=None, Ly=None, comm=None):
    """Compute spectral enstrophy flux directly from Dedalus fields."""

    if Lx is None or Ly is None:
        raise ValueError("Lx and Ly must be provided for coefficient flux diagnostics.")

    dist, xb, yb, resolved_comm = _resolve_context(u, dist, xbasis, ybasis, comm)
    weight, shell_idx, k0 = _prepare_shell_metadata(u, dist, xb, yb, Lx, Ly)
    area = Lx * Ly
    scale = _dealias_scale(xb, yb)

    u.change_scales(1)
    u.require_coeff_space()
    ux_orig = np.array(u['c'][0], copy=True)
    uy_orig = np.array(u['c'][1], copy=True)
    omega = (-d3.div(d3.skew(u))).evaluate()
    omega.change_scales(1)
    omega.require_coeff_space()
    omega_orig = np.array(omega['c'], copy=True)

    if scale > 1.0:
        u.change_scales(scale)
    u.require_grid_space()
    omega.change_scales(scale)
    omega.require_grid_space()

    adv_scalar = (u @ d3.grad(omega)).evaluate()
    adv_scalar.change_scales(scale)
    adv_scalar.require_grid_space()

    u.change_scales(1)
    omega.change_scales(1)
    adv_scalar.change_scales(1)
    u.require_coeff_space()
    omega.require_coeff_space()
    adv_scalar.require_coeff_space()

    u['c'][0] = ux_orig
    u['c'][1] = uy_orig
    omega['c'] = omega_orig

    omega_c = omega_orig
    adv_c = np.asarray(adv_scalar['c'])
    transfer = np.real(np.conj(omega_c) * adv_c) * weight * area

    T_shell, mmax = _shell_bincount(resolved_comm, shell_idx, transfer)
    Pi_shell = -np.cumsum(T_shell)
    k_bins = np.arange(mmax + 1, dtype=np.float64) * k0

    u.require_grid_space()
    return _return_if_root(resolved_comm, (k_bins, T_shell, Pi_shell))
