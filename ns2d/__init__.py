"""
NS2D: 2D Incompressible Navier-Stokes Solver
=============================================

Forced 2D incompressible Navier-Stokes equations using the Dedalus spectral solver framework.

Modules:
    config: Configuration and command-line argument parsing
    domain: Domain setup, grid construction, and wavenumber utilities
    forcing: Stochastic forcing implementations (white noise, OU process)
    spectral: Spectral analysis (energy/enstrophy spectra and fluxes)
    solver: Main solver setup and time integration
    utils: MPI utilities and diagnostic functions
"""

__version__ = "0.1.0"
__author__ = "Michael Groom"

from . import config
from . import domain
from . import forcing
from . import spectral
from . import solver
from . import utils

__all__ = [
    "config",
    "domain",
    "forcing",
    "spectral",
    "solver",
    "utils",
]
