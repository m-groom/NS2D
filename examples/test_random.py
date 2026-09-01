import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import matplotlib.pyplot as plt

# Import parent directory
import sys
sys.path.append('..')

from ns2d import forcing, domain, spectral, utils

# -----------------------------
# Domain and bases
# -----------------------------
Nx, Ny = 256, 256
Lx, Ly = 2*np.pi, 2*np.pi

coords = d3.CartesianCoordinates('x', 'y')
dist   = d3.Distributor(coords, dtype=np.float64, comm=MPI.COMM_WORLD)

# Simulation bases: RealFourier 
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=1.5)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=1.5)

u = dist.VectorField(coords, bases=(xbasis, ybasis), name='u')  # 2 components (ux, uy)

x, y = dist.local_grids(xbasis, ybasis)

# -----------------------------
# Compare forcing generators
# -----------------------------

seed = 1234
sigma = 0.02

kx, ky, KX, KY, K2, K = domain.wavenumbers(Nx, Ny, Lx, Ly)

# Forcing mask: match the production solver setup with kmin=8, kmax=12
kmin = 8.0
kmax = 12.0
mask_full = forcing.build_forcing_mask(K, kmin, kmax)

rng = np.random.default_rng(seed)
legacy = forcing.stochastic_forcing(Nx, Ny, KX, KY, K, mask_full, rng, sigma, stype="ou")
fx_old, fy_old = legacy(dt=1.0)

# -----------------------------
# Diagnostics for legacy forcing
# -----------------------------
energy_old = np.sum(np.abs(fx_old)**2 + np.abs(fy_old)**2)
if dist.comm.rank == 0:
    print(f"Energy of the forcing (old): {energy_old:.3e}")
    rms_old = np.sqrt(energy_old / (Nx * Ny))
    max_fx_old = np.max(np.abs(fx_old))
    max_fy_old = np.max(np.abs(fy_old))
    print(f"RMS forcing amplitude (old): {rms_old:.3e}")
    print(f"Max |fx|, |fy| (old): {max_fx_old:.3e}, {max_fy_old:.3e}")

dist_forcing = forcing.distributed_stochastic_forcing(
    dist,
    coords,
    xbasis,
    ybasis,
    KX,
    KY,
    mask_full,
    sigma_base=sigma,
    seed=seed,
    stype="ou",
)

forcing_field_new = dist_forcing(dt=1.0)
forcing_field_new.require_grid_space()
fx_new = forcing_field_new['g'][0]
fy_new = forcing_field_new['g'][1]

energy_new_local = np.sum(np.abs(fx_new)**2 + np.abs(fy_new)**2)
energy_new = dist.comm.allreduce(energy_new_local, op=MPI.SUM)

# Global max amplitudes for the distributed forcing
max_fx_new_local = np.max(np.abs(fx_new))
max_fy_new_local = np.max(np.abs(fy_new))
max_fx_new = dist.comm.allreduce(max_fx_new_local, op=MPI.MAX)
max_fy_new = dist.comm.allreduce(max_fy_new_local, op=MPI.MAX)

# Build Dedalus vector fields for forcing (used for both plots and spectra)
f_old = dist.VectorField(coords, bases=(xbasis, ybasis), name="forcing_old")
f_new = dist.VectorField(coords, bases=(xbasis, ybasis), name="forcing_new")

f_old.change_scales(1)
f_new.change_scales(1)

# On all ranks: assign local slices.
# Legacy forcing is available as a full grid on each rank, so slice it.
# Distributed forcing is already local on each rank.
f_slices = utils.local_slices(f_old)
f_old['g'][0] = fx_old[f_slices]
f_old['g'][1] = fy_old[f_slices]
f_new['g'][0] = fx_new
f_new['g'][1] = fy_new

# Gather full-grid fields to rank 0 for plotting
forcing_old_global = utils.gather_field_to_rank0(f_old, dist.comm, (2, Nx, Ny))
forcing_new_global = utils.gather_field_to_rank0(f_new, dist.comm, (2, Nx, Ny))

if dist.comm.rank == 0:
    print(f"Energy of the forcing (new): {energy_new:.3e}")
    rms_new = np.sqrt(energy_new / (Nx * Ny))
    print(f"RMS forcing amplitude (new): {rms_new:.3e}")
    print(f"Max |fx|, |fy| (new): {max_fx_new:.3e}, {max_fy_new:.3e}")

    # -----------------------------
    # Plot forcing fields in physical space (single-rank diagnostic)
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Use gathered full-grid fields if available; fall back to local arrays.
    if forcing_old_global is not None:
        fx_old_plot = forcing_old_global[0]
        fy_old_plot = forcing_old_global[1]
    else:
        fx_old_plot = fx_old
        fy_old_plot = fy_old

    if forcing_new_global is not None:
        fx_new_plot = forcing_new_global[0]
        fy_new_plot = forcing_new_global[1]
    else:
        fx_new_plot = fx_new
        fy_new_plot = fy_new

    im = axes[0, 0].imshow(
        fx_old_plot,
        origin="lower",
        extent=(0, Lx, 0, Ly),
        aspect="equal",
    )
    axes[0, 0].set_title("fx_old")
    fig.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(
        fy_old_plot,
        origin="lower",
        extent=(0, Lx, 0, Ly),
        aspect="equal",
    )
    axes[0, 1].set_title("fy_old")
    fig.colorbar(im, ax=axes[0, 1])

    im = axes[1, 0].imshow(
        fx_new_plot,
        origin="lower",
        extent=(0, Lx, 0, Ly),
        aspect="equal",
    )
    axes[1, 0].set_title("fx_new")
    fig.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(
        fy_new_plot,
        origin="lower",
        extent=(0, Lx, 0, Ly),
        aspect="equal",
    )
    axes[1, 1].set_title("fy_new")
    fig.colorbar(im, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    plt.savefig("forcing_comparison.png", dpi=150)
    plt.close(fig)

# -----------------------------
# Spectral diagnostics (isotropic energy spectra)
# -----------------------------
spec_old = spectral.compute_spectra_from_coeffs(
    f_old, dist=dist, xbasis=xbasis, ybasis=ybasis, Lx=Lx, Ly=Ly, comm=dist.comm
)
spec_new = spectral.compute_spectra_from_coeffs(
    f_new, dist=dist, xbasis=xbasis, ybasis=ybasis, Lx=Lx, Ly=Ly, comm=dist.comm
)

if dist.comm.rank == 0 and spec_old is not None and spec_new is not None:
    k_bins_old, E_k_old, _ = spec_old
    k_bins_new, E_k_new, _ = spec_new

    plt.figure(figsize=(6, 4))
    plt.loglog(k_bins_old, E_k_old, label="legacy forcing")
    plt.loglog(k_bins_new, E_k_new, label="distributed forcing", linestyle="--")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E_f(k)$")
    plt.title("Isotropic energy spectra of forcing")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forcing_spectra.png", dpi=150)
    plt.close()
