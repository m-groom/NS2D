"""Shared fixtures for the NS2D test suite.

These tests run serially (one MPI rank) on a small grid, so they are cheap
enough for a login node.
"""

import numpy as np
import pytest

from ns2d import domain, forcing


NX = NY = 32
LX = LY = 2.0 * np.pi
KMIN, KMAX = 4.0, 8.0


@pytest.fixture(scope="module")
def basis():
    """Build a small periodic 2D RealFourier domain and its wavenumber grids."""
    coords, dist, xbasis, ybasis, x, y = domain.build_domain(
        NX, NY, LX, LY, 1.5, np.float64
    )
    kx, ky, KX, KY, K2, K = domain.wavenumbers(NX, NY, LX, LY)
    return dict(coords=coords, dist=dist, xbasis=xbasis, ybasis=ybasis,
                KX=KX, KY=KY, K=K)


def make_generator(basis, seed, realisation, stype="ou", tau=0.3):
    """Build a distributed stochastic forcing generator on the test domain."""
    mask = forcing.build_forcing_mask(basis['K'], KMIN, KMAX)
    update, state = forcing.distributed_stochastic_forcing(
        basis['dist'], basis['coords'], basis['xbasis'], basis['ybasis'],
        basis['KX'], basis['KY'], mask,
        sigma_base=1.0, seed=seed, realisation=realisation, stype=stype, tau=tau,
    )
    return update, state


def forcing_series(basis, seed, realisation, nsteps, dt=0.01, **kwargs):
    """Step a generator *nsteps* times and return the grid-space forcing history."""
    update, _ = make_generator(basis, seed, realisation, **kwargs)
    out = []
    for _ in range(nsteps):
        f = update(dt)
        f.require_grid_space()
        out.append(np.copy(f['g']))
    return np.array(out)


def shell_energy(basis, field):
    """Bin the energy of a vector *field* into isotropic wavenumber shells."""
    field.change_scales(1)
    field.require_coeff_space()
    nx = np.arange(NX, dtype=np.int64)[:, None] // 2
    ny = np.arange(NY, dtype=np.int64)[None, :] // 2
    shell = np.rint(np.sqrt(nx.astype(float) ** 2 + ny.astype(float) ** 2)).astype(int)
    power = field['c'][0] ** 2 + field['c'][1] ** 2
    return np.bincount(shell.ravel(), weights=power.ravel(), minlength=NX)
