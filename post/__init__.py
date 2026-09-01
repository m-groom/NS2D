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
from . import analysis
try:
    # Optional: visualisation depends on Dedalus (plot_tools).
    from . import visualisation  # type: ignore
except ModuleNotFoundError as e:
    # Allow using io/analysis without having Dedalus installed.
    if getattr(e, "name", "").startswith("dedalus"):
        visualisation = None  # type: ignore
    else:
        raise

__all__ = ["io", "visualisation", "analysis"]
