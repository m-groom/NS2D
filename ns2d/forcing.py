"""
Stochastic forcing implementations for 2D turbulence simulations.

This module provides band-limited stochastic forcing with two types:
1. White noise (δ-correlated in time)
2. Ornstein-Uhlenbeck (OU) process (finite correlation time)

Both forcing types enforce incompressibility (divergence-free) and support
constant-power rescaling to maintain a target energy injection rate.
"""

import numpy as np
from mpi4py import MPI

from . import spectral

def build_forcing_mask(K, kmin, kmax):
    """
    Create a boolean mask for band-limited forcing in spectral space.

    Args:
        K (ndarray): Wavenumber magnitude array (Nx, Ny//2+1)
        kmin (float): Minimum forced wavenumber (physical units)
        kmax (float): Maximum forced wavenumber (physical units)

    Returns:
        ndarray: Boolean mask, True for wavenumbers in [kmin, kmax],
            with k=0 excluded to preserve zero mean flow.
    """
    mask = (K >= kmin) & (K <= kmax)
    mask[0, 0] = False  # Exclude k=0 to maintain zero mean
    return mask


def project_div_free(kx, ky, fx, fy):
    """
    Project a vector field onto the divergence-free (incompressible) subspace.

    Removes the compressible component: f_div = (k̂·f̂) k̂

    Args:
        kx (ndarray): x-wavenumbers (broadcastable to forcing shape)
        ky (ndarray): y-wavenumbers (broadcastable to forcing shape)
        fx (ndarray): x-component of forcing in spectral space
        fy (ndarray): y-component of forcing in spectral space

    Returns:
        tuple: (fxp, fyp) - divergence-free projections of (fx, fy)
    """
    k2 = kx * kx + ky * ky
    with np.errstate(divide='ignore', invalid='ignore'):
        dot = kx * fx + ky * fy
        corr = np.where(k2 == 0.0, 0.0, dot / k2)
        fxp = fx - kx * corr
        fyp = fy - ky * corr
    return fxp, fyp


def stochastic_forcing(Nx, Ny, KX, KY, K, mask, rng, sigma_base, stype="white", tau=0.5):
    """
    Create a stochastic forcing generator for 2D turbulence simulations.

    Returns a closure that generates divergence-free, band-limited random
    forcing in physical space. The forcing is normalized so that its RMS
    amplitude in grid space is approximately sigma_base, independent of
    grid resolution and forcing band width.

    Args:
        Nx (int): Grid size in x direction
        Ny (int): Grid size in y direction
        KX (ndarray): x-wavenumber meshgrid (Nx, Ny//2+1)
        KY (ndarray): y-wavenumber meshgrid (Nx, Ny//2+1)
        K (ndarray): Wavenumber magnitude (Nx, Ny//2+1)
        mask (ndarray): Boolean mask selecting forced wavenumbers
        rng: NumPy random generator
        sigma_base (float): Target RMS forcing amplitude in grid space
        stype (str): "white" for white-in-time or "ou" for Ornstein-Uhlenbeck
        tau (float): Correlation time for OU forcing (ignored for white)

    Returns:
        callable: update(dt) function that returns (fx_grid, fy_grid),
            the forcing field in physical space for a timestep dt.

    Notes:
        - The returned forcing fields are in GRID space, ready for use
        - Constant-power rescaling should be applied separately
        - rfft reality conditions are enforced automatically
        - Forcing is resolution-independent up to the Nyquist limit
    """
    # State variables for OU process
    ax = np.zeros_like(K, dtype=np.complex128)
    ay = np.zeros_like(K, dtype=np.complex128)

    N = Nx * Ny

    # rfft symmetry weights: double interior ky>0 modes
    weight = np.ones_like(K, dtype=np.float64)
    if weight.shape[1] > 1:
        weight[:, 1:-1] = 2.0
    weight[:, 0] = 1.0
    if Ny % 2 == 0:
        weight[:, -1] = 1.0  # Nyquist is real-valued

    # Effective number of forced modes (accounting for rfft symmetry)
    eff_mask = mask.astype(bool)
    M_eff = float(np.sum(weight[eff_mask]))

    # Spectral amplitude to achieve ~sigma_base RMS in grid space
    A = (sigma_base * N) / np.sqrt(max(M_eff, 1.0))

    def enforce_rfft_reality(arr):
        """Enforce Nyquist reality for even Ny."""
        if Ny % 2 == 0:
            arr[:, -1] = np.real(arr[:, -1]) + 0j

    def update(dt):
        """
        Generate forcing for a timestep dt.

        Args:
            dt (float): Timestep size

        Returns:
            tuple: (fx_grid, fy_grid) - forcing in physical space
        """
        nonlocal ax, ay

        if stype == "white":
            # White-in-time: f(t) = ξ(t) / √dt
            s = A / np.sqrt(max(dt, 1e-12))
            gx = (rng.standard_normal(K.shape) + 1j * rng.standard_normal(K.shape)) * (s / np.sqrt(2.0))
            gy = (rng.standard_normal(K.shape) + 1j * rng.standard_normal(K.shape)) * (s / np.sqrt(2.0))

            # Apply mask and symmetry weights
            gx[~eff_mask] = 0.0
            gy[~eff_mask] = 0.0
            gx *= np.sqrt(weight)
            gy *= np.sqrt(weight)

            # Project to divergence-free
            fxh, fyh = project_div_free(KX, KY, gx, gy)

        else:  # "ou"
            # Ornstein-Uhlenbeck: da = -a/τ dt + dW
            # Exact update: a(t+dt) = e^(-dt/τ) a(t) + ξ √(1 - e^(-2dt/τ))
            if tau <= 0:
                raise ValueError("tau_ou must be > 0 for OU forcing")

            e = np.exp(-dt / tau)
            s = A * np.sqrt(max(0.0, 1.0 - e * e))

            gx = (rng.standard_normal(K.shape) + 1j * rng.standard_normal(K.shape)) * (s / np.sqrt(2.0))
            gy = (rng.standard_normal(K.shape) + 1j * rng.standard_normal(K.shape)) * (s / np.sqrt(2.0))
            gx *= np.sqrt(weight)
            gy *= np.sqrt(weight)

            # Update OU state
            ax = e * ax + gx
            ay = e * ay + gy

            # Apply mask
            ax[~eff_mask] = 0.0
            ay[~eff_mask] = 0.0

            # Enforce reality
            enforce_rfft_reality(ax)
            enforce_rfft_reality(ay)

            # Project to divergence-free
            fxh, fyh = project_div_free(KX, KY, ax, ay)

        # Enforce reality conditions
        enforce_rfft_reality(fxh)
        enforce_rfft_reality(fyh)

        # Transform to physical space
        fx_grid = np.fft.irfft2(fxh, s=(Nx, Ny))
        fy_grid = np.fft.irfft2(fyh, s=(Nx, Ny))

        return fx_grid, fy_grid

    return update


def distributed_stochastic_forcing(dist, coords, xbasis, ybasis, KX, KY, mask,
                                   sigma_base, seed=0, stype="white", tau=0.5):
    """
    Distributed stochastic forcing that mirrors `stochastic_forcing` statistics.
    """

    if stype not in ("white", "ou"):
        raise ValueError(f"Unsupported forcing type '{stype}'")
    if stype == "ou" and tau <= 0:
        raise ValueError("tau_ou must be > 0 for OU forcing")

    # Global grid sizes from bases
    Nx = getattr(xbasis, 'size', None) or getattr(xbasis, 'N')
    Ny = getattr(ybasis, 'size', None) or getattr(ybasis, 'N')
    if Nx is None or Ny is None:
        raise ValueError("Unable to determine basis sizes for distributed forcing")

    # Deduce physical domain lengths from the rFFT wavenumbers KX, KY.
    # For KX built via domain.wavenumbers: kx[1] = 2π / Lx, ky[1] = 2π / Ly.
    dkx = float(np.abs(KX[1, 0])) if Nx > 1 else 0.0
    dky = float(np.abs(KY[0, 1])) if Ny > 1 else 0.0
    Lx = 2 * np.pi / dkx if dkx > 0 else 1.0
    Ly = 2 * np.pi / dky if dky > 0 else 1.0

    # Compute the physical wavenumber band [kmin, kmax] implied by the mask.
    K_mag = np.sqrt(KX * KX + KY * KY)
    if not np.any(mask):
        raise ValueError("Forcing mask is empty; no modes to force.")
    kmin_phys = float(K_mag[mask].min())
    kmax_phys = float(K_mag[mask].max())

    # Allocate a coefficient-space vector field for the forcing.
    forcing_field = dist.VectorField(coords, bases=(xbasis, ybasis), name="forcing")
    forcing_field.require_coeff_space()

    # Local RealFourier mode numbers (integers) for each axis.
    kx_modes = dist.local_modes(xbasis)
    ky_modes = dist.local_modes(ybasis)

    # Broadcast mode numbers to match the local coefficient array shape.
    fx_c = forcing_field['c'][0]
    fy_c = forcing_field['c'][1]
    kx_modes_2d = np.broadcast_to(kx_modes, fx_c.shape)
    ky_modes_2d = np.broadcast_to(ky_modes, fy_c.shape)

    # Physical wavenumbers for each local coefficient (for divergence-free projection).
    kx_phys_local = (2.0 * np.pi / Lx) * kx_modes_2d
    ky_phys_local = (2.0 * np.pi / Ly) * ky_modes_2d

    # Choose shell indices whose physical wavenumbers lie in [kmin_phys, kmax_phys].
    weight_local, shell_idx_local, k0 = spectral._prepare_shell_metadata(
        forcing_field, dist, xbasis, ybasis, Lx, Ly
    )
    # Shell m corresponds to |k| ≈ m * k0.
    m_min = int(np.ceil(kmin_phys / k0))
    m_max = int(np.floor(kmax_phys / k0))
    if m_max < m_min:
        raise ValueError(
            f"No RealFourier shells fall in requested band [{kmin_phys}, {kmax_phys}]"
        )

    mask_local = (shell_idx_local >= m_min) & (shell_idx_local <= m_max)
    # Exclude the (0,0) mode to preserve zero-mean forcing.
    mask_local &= ~((kx_modes_2d == 0) & (ky_modes_2d == 0))

    # Effective number of forced modes in RealFourier space
    M_eff_local = float(np.sum(weight_local[mask_local]))
    comm = dist.comm
    M_eff = comm.allreduce(M_eff_local, op=MPI.SUM)

    # For RealFourier coefficients a_k with variance s^2 on forced modes:
    #   <f^2> ≈ s^2 * sum(weight) = s^2 * M_eff
    # so choose s such that <f^2> ≈ sigma_base^2.
    norm = np.sqrt(2.0) * sigma_base / np.sqrt(max(M_eff, 1.0))

    # State for OU process (in coefficient space).
    state_x = np.zeros_like(fx_c)
    state_y = np.zeros_like(fy_c)

    # Step counter for time-decorrelated seeds in white/OU forcing.
    step_counter = np.array([0], dtype=np.int64)

    def update(dt):
        """Generate one timestep of distributed stochastic forcing."""
        step_counter[0] += 1
        dt_safe = max(dt, 1e-12)

        # Work in coefficient space for the forcing field.
        forcing_field.require_coeff_space()
        fx_c = forcing_field['c'][0]
        fy_c = forcing_field['c'][1]
        fx_c.fill(0.0)
        fy_c.fill(0.0)

        if stype == "white":
            # White-in-time forcing: f ~ ξ / sqrt(dt), where ξ ~ N(0, norm^2).
            scale = norm / np.sqrt(dt_safe)

            # Fill with standard-normal noise in coefficient space using
            # Dedalus's layout-independent RNG utilities.
            forcing_field.fill_random(
                layout='c',
                seed=int(seed) + int(step_counter[0]),
                distribution='standard_normal',
            )

            # Apply scale and mask in RealFourier coefficient space.
            fx_c *= scale
            fy_c *= scale
            fx_c[~mask_local] = 0.0
            fy_c[~mask_local] = 0.0

            # Project to divergence-free using physical wavenumbers.
            tmp_x, tmp_y = project_div_free(kx_phys_local, ky_phys_local, fx_c, fy_c)
            fx_c[...] = tmp_x
            fy_c[...] = tmp_y

        else:
            # Ornstein-Uhlenbeck forcing in coefficient space.
            e = np.exp(-dt_safe / tau)
            s_ou = norm * np.sqrt(max(0.0, 1.0 - e * e))

            # Generate standard-normal noise for the OU increment.
            forcing_field.fill_random(
                layout='c',
                seed=int(seed) + 10_000_000 + int(step_counter[0]),
                distribution='standard_normal',
            )
            eta_x = forcing_field['c'][0]
            eta_y = forcing_field['c'][1]

            # Update OU state only on forced modes.
            state_x[:] = e * state_x
            state_y[:] = e * state_y
            state_x[mask_local] += s_ou * eta_x[mask_local]
            state_y[mask_local] += s_ou * eta_y[mask_local]
            state_x[~mask_local] = 0.0
            state_y[~mask_local] = 0.0

            # Project OU state to divergence-free and use as forcing.
            tmp_x, tmp_y = project_div_free(kx_phys_local, ky_phys_local, state_x, state_y)
            fx_c[...] = tmp_x
            fy_c[...] = tmp_y

        # Return VectorField with updated forcing
        forcing_field.require_grid_space()
        return forcing_field

    return update


def constant_power_rescale(fx_loc, fy_loc, ux_loc, uy_loc, comm, Nx, Ny,
                           eps_target, eps_floor=1e-12, eps_clip=10.0,
                           scale_state=None, eps_smooth=0.0):
    """
    Rescale forcing to achieve constant energy injection rate.

    Computes the instantaneous injection rate ε_inst = <u·f> (domain average)
    and rescales the forcing by (eps_target / ε_inst) to maintain constant
    power input.

    Args:
        fx_loc (ndarray): Local x-forcing on this MPI rank
        fy_loc (ndarray): Local y-forcing on this MPI rank
        ux_loc (ndarray): Local x-velocity on this MPI rank
        uy_loc (ndarray): Local y-velocity on this MPI rank
        comm: MPI communicator
        Nx (int): Global grid size in x
        Ny (int): Global grid size in y
        eps_target (float): Target injection rate
        eps_floor (float): Floor to prevent division by very small values
        eps_clip (float): Maximum allowed rescale factor
        scale_state (ndarray or None): [scale] for exponential smoothing
        eps_smooth (float): Smoothing factor (0 = no smoothing)

    Returns:
        tuple: (fx_rescaled, fy_rescaled) - rescaled forcing fields
    """
    # Compute local contribution to <u·f>
    dot_local = np.sum(ux_loc * fx_loc + uy_loc * fy_loc, dtype=np.float64)
    dot_global = comm.allreduce(dot_local, op=MPI.SUM)
    eps_inst = dot_global / (Nx * Ny)

    # Compute rescale factor
    denom = max(abs(eps_inst), eps_floor)
    scale_raw = eps_target / denom

    # Optional exponential smoothing
    if eps_smooth > 0.0 and scale_state is not None:
        gamma = float(eps_smooth)
        scale = (1.0 - gamma) * scale_state[0] + gamma * scale_raw
        scale_state[0] = scale
    else:
        scale = scale_raw

    # Clip to prevent extreme values
    scale = max(0.0, min(scale, eps_clip))

    # Apply scaling
    fx_rescaled = fx_loc * scale
    fy_rescaled = fy_loc * scale

    return fx_rescaled, fy_rescaled
