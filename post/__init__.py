"""
NS2D Post-Processing Toolkit
=============================

A modular post-processing and visualization toolkit for NS2D simulation output.

Modules:
    io: Data loading and file I/O utilities
    visualization: Plotting functions for time series, spectra, and snapshots
    analysis: Analysis utilities (statistics, averages, derived quantities)
"""

__version__ = "1.0.0"

from . import io
from . import visualization
from . import analysis

__all__ = ["io", "visualization", "analysis"]
