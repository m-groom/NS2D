"""Tests for the distributed stochastic forcing generator."""

import numpy as np
import pytest
import dedalus.public as d3

from conftest import KMIN, KMAX, forcing_series, make_generator, shell_energy


def _corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0.0 else float(a @ b / denom)


def test_realisations_are_independent(basis):
    """Different realisations share no forcing sequence, at any time shift.

    Guards against an additive per-step seed (seed + r + step), under which
    realisation r+1 reproduces realisation r shifted by one timestep and this
    test measures corr = 1.000000 at shift +1.
    """
    nsteps = 120
    f0 = forcing_series(basis, seed=20260823, realisation=0, nsteps=nsteps)
    f1 = forcing_series(basis, seed=20260823, realisation=1, nsteps=nsteps)

    # Compare well after the OU transient has decayed.
    n = 100
    worst = 0.0
    for k in range(-5, 6):
        worst = max(worst, abs(_corr(f0[n + k], f1[n])))
    # Two independent fields on this small grid correlate at |corr| ~ 0.1 from
    # sampling noise alone (a few hundred forced degrees of freedom); the defect
    # gave exactly 1.000000 at shift +1.
    assert worst < 0.3, f"realisations correlate at |corr| = {worst:.6f}"


def test_realisation_is_reproducible(basis):
    """A fixed (seed, realisation, dt) sequence reproduces bit-for-bit."""
    a = forcing_series(basis, seed=20260823, realisation=3, nsteps=20)
    b = forcing_series(basis, seed=20260823, realisation=3, nsteps=20)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("stype", ["ou", "white"])
def test_forcing_is_divergence_free(basis, stype):
    """The generated forcing has zero divergence to machine precision.

    Guards against an element-wise projection in RealFourier coefficient space,
    which cannot enforce the constraint at all: this test measures
    RMS(div f) / (k_f RMS|f|) ~ 0.5 for one.
    """
    update, _ = make_generator(basis, seed=20260823, realisation=0, stype=stype)
    for _ in range(20):
        f = update(0.01)

    divf = d3.div(f).evaluate()
    divf.change_scales(1)
    f.change_scales(1)
    rms_div = float(np.sqrt(np.mean(divf['g'] ** 2)))
    rms_f = float(np.sqrt(np.mean(f['g'][0] ** 2 + f['g'][1] ** 2)))
    k_f = 0.5 * (KMIN + KMAX)
    assert rms_div / (k_f * rms_f) < 1e-12, (
        f"RMS div(f) / (k_f RMS|f|) = {rms_div / (k_f * rms_f):.3e}"
    )


def test_forcing_stays_in_the_requested_band(basis):
    """All forcing energy sits in the shells requested by [kmin, kmax]."""
    update, _ = make_generator(basis, seed=20260823, realisation=0)
    for _ in range(20):
        f = update(0.01)

    shells = shell_energy(basis, f)
    inside = shells[int(np.floor(KMIN)):int(np.ceil(KMAX)) + 1].sum()
    assert shells.sum() > 0.0
    assert abs(inside / shells.sum() - 1.0) < 1e-12, (
        f"only {inside / shells.sum():.6f} of the forcing energy is in band"
    )
