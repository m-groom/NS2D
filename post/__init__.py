"""
NS2D Post-Processing Toolkit
=============================

A modular post-processing and visualisation toolkit for NS2D simulation output.

Modules:
    io: Data loading and file I/O utilities
    visualisation: Plotting functions for time series, spectra, and snapshots
    analysis: Analysis utilities (statistics, averages, derived quantities)
"""

__version__ = "0.1.0"

from . import io
from . import visualisation
from . import analysis

__all__ = ["io", "visualisation", "analysis"]
