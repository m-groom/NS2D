"""
Visualization functions for NS2D simulation output.

This module provides plotting functions for:
- Time series (energy, enstrophy, etc.)
- Spectra (energy and enstrophy)
- Spectral fluxes (energy and enstrophy transfer/cascade)
- 2D field snapshots (vorticity, pressure, streamfunction)
"""

import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend by default for batch processing
matplotlib.use("Agg")

try:
    from dedalus.extras import plot_tools
    DEDALUS_AVAILABLE = True
except ImportError:
    DEDALUS_AVAILABLE = False


def set_style(style="default"):
    """
    Set matplotlib style for plots.

    Args:
        style (str): Style name ("default", "paper", "presentation")
    """
    if style == "paper":
        plt.style.use("seaborn-v0_8-paper" if hasattr(plt.style, "available") else "default")
        plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    elif style == "presentation":
        plt.rcParams.update({"font.size": 14, "figure.dpi": 100, "lines.linewidth": 2})
    else:
        plt.style.use("default")


def plot_time_series(times, series_dict, outdir=".", dpi=300, show_balance=True):
    """
    Plot scalar time series (energy, enstrophy, etc.).

    Args:
        times (ndarray): Time values (N,)
        series_dict (dict): Dictionary of scalar arrays {name: (N,) array}
        outdir (str or Path): Output directory for figures
        dpi (int): Figure DPI
        show_balance (bool): Plot energy balance if all terms present

    Generates:
        - energy.png
        - enstrophy.png
        - palinstrophy.png
        - inj.png (if present)
        - drag_loss.png (if present)
        - visc_loss.png (if present)
        - energy_balance.png (if all terms present and show_balance=True)
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Standard plots
    plot_specs = {
        "energy": {"ylabel": "Energy E", "title": "Kinetic Energy vs Time"},
        "enstrophy": {"ylabel": "Enstrophy Z", "title": "Enstrophy vs Time"},
        "palinstrophy": {"ylabel": "Palinstrophy P", "title": "Palinstrophy vs Time"},
        "inj": {"ylabel": "ε_inj", "title": "Energy Injection Rate vs Time"},
        "drag_loss": {"ylabel": "ε_drag", "title": "Drag Dissipation vs Time"},
        "visc_loss": {"ylabel": "ε_visc", "title": "Viscous Dissipation vs Time"},
    }

    for key, spec in plot_specs.items():
        if key not in series_dict:
            continue

        plt.figure(figsize=(8, 4.5))
        plt.plot(times, series_dict[key], linewidth=1.5)
        plt.xlabel("Time t")
        plt.ylabel(spec["ylabel"])
        plt.title(spec["title"])
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(outdir / f"{key}.png", dpi=dpi, bbox_inches="tight")
        plt.close()

    # Energy balance plot
    if show_balance and all(k in series_dict for k in ["inj", "drag_loss", "visc_loss"]):
        residual = series_dict["inj"] - series_dict["drag_loss"] - series_dict["visc_loss"]

        plt.figure(figsize=(8, 4.5))
        plt.plot(times, residual, linewidth=1.5, label="Residual")
        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel("Time t")
        plt.ylabel("ε_inj - ε_drag - ε_visc")
        plt.title("Energy Balance Residual")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "energy_balance.png", dpi=dpi, bbox_inches="tight")
        plt.close()


def plot_spectra(times, kbins, Ek_list, Zk_list, outdir=".", max_curves=6,
                 loglog=True, k_nyquist=None, dpi=300):
    """
    Plot energy and enstrophy spectra.

    Args:
        times (ndarray): Snapshot times (T,)
        kbins (ndarray): Wavenumber bins (M,)
        Ek_list (list): List of energy spectra, each (M,)
        Zk_list (list): List of enstrophy spectra, each (M,)
        outdir (str or Path): Output directory
        max_curves (int): Maximum number of curves to overlay
        loglog (bool): Use log-log axes
        k_nyquist (float or None): Clip at Nyquist wavenumber
        dpi (int): Figure DPI

    Generates:
        - energy_spectrum.png
        - enstrophy_spectrum.png
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Select time indices to plot
    idxs = np.linspace(0, len(times) - 1, num=min(max_curves, len(times)), dtype=int)

    # Stack spectra
    Ek_stack = np.stack(Ek_list, axis=0)  # (T, M)
    Zk_stack = np.stack(Zk_list, axis=0)

    # Apply Nyquist clipping
    kbins_plot = kbins
    if k_nyquist is not None:
        mask = kbins <= k_nyquist
        kbins_plot = kbins[mask]
        Ek_stack = Ek_stack[:, mask]
        Zk_stack = Zk_stack[:, mask]

    # Energy spectrum
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    if loglog:
        start = 1 if kbins_plot.size > 0 and kbins_plot[0] == 0 else 0
        for i in idxs:
            ax.loglog(kbins_plot[start:], Ek_stack[i][start:],
                     alpha=0.6, linewidth=1.5, label=f"t = {times[i]:.2f}")
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("E(k)")
    else:
        for i in idxs:
            plt.plot(kbins_plot, Ek_stack[i], alpha=0.6, linewidth=1.5,
                    label=f"t = {times[i]:.2f}")
        plt.xlabel("Wavenumber k")
        plt.ylabel("E(k)")

    plt.title("Energy Spectrum E(k)")
    plt.grid(True, which="both", alpha=0.3, linestyle="--")
    plt.legend(ncol=2, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "energy_spectrum.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # Enstrophy spectrum
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    if loglog:
        start = 1 if kbins_plot.size > 0 and kbins_plot[0] == 0 else 0
        for i in idxs:
            ax.loglog(kbins_plot[start:], Zk_stack[i][start:],
                     alpha=0.6, linewidth=1.5, label=f"t = {times[i]:.2f}")
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("Z(k)")
    else:
        for i in idxs:
            plt.plot(kbins_plot, Zk_stack[i], alpha=0.6, linewidth=1.5,
                    label=f"t = {times[i]:.2f}")
        plt.xlabel("Wavenumber k")
        plt.ylabel("Z(k)")

    plt.title("Enstrophy Spectrum Z(k)")
    plt.grid(True, which="both", alpha=0.3, linestyle="--")
    plt.legend(ncol=2, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "enstrophy_spectrum.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_flux(times, kbins, T_list, Pi_list, outdir=".", flux_type="energy",
              max_curves=6, k_nyquist=None, dpi=300):
    """
    Plot spectral transfer and cumulative flux.

    Args:
        times (ndarray): Snapshot times (T,)
        kbins (ndarray): Wavenumber bins (M,)
        T_list (list): List of transfer spectra, each (M,)
        Pi_list (list): List of cumulative flux, each (M,)
        outdir (str or Path): Output directory
        flux_type (str): "energy" or "enstrophy" (for labeling)
        max_curves (int): Maximum number of curves to overlay
        k_nyquist (float or None): Clip at Nyquist wavenumber
        dpi (int): Figure DPI

    Generates:
        - {flux_type}_transfer.png
        - {flux_type}_flux.png
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Select time indices
    idxs = np.linspace(0, len(times) - 1, num=min(max_curves, len(times)), dtype=int)

    # Stack
    T_stack = np.stack(T_list, axis=0)
    Pi_stack = np.stack(Pi_list, axis=0)

    # Apply Nyquist clipping
    kbins_plot = kbins
    if k_nyquist is not None:
        mask = kbins <= k_nyquist
        kbins_plot = kbins[mask]
        T_stack = T_stack[:, mask]
        Pi_stack = Pi_stack[:, mask]

    # Use positive k for log scale (exclude k=0)
    kpos_mask = kbins_plot > 0

    # Transfer T(k)
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for i in idxs:
        ax.plot(kbins_plot[kpos_mask], T_stack[i][kpos_mask],
               alpha=0.6, linewidth=1.5, label=f"t = {times[i]:.2f}")
    ax.set_xscale('log')
    plt.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    plt.xlabel("Wavenumber k")
    plt.ylabel(f"T(k)")
    plt.title(f"{flux_type.capitalize()} Transfer T(k)")
    plt.grid(True, which="both", alpha=0.3, linestyle="--")
    plt.legend(ncol=2, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / f"{flux_type}_transfer.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # Cumulative flux Π(k)
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for i in idxs:
        ax.plot(kbins_plot[kpos_mask], Pi_stack[i][kpos_mask],
               alpha=0.6, linewidth=1.5, label=f"t = {times[i]:.2f}")
    ax.set_xscale('log')
    plt.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    plt.xlabel("Wavenumber k")
    plt.ylabel(f"Π(k)")
    plt.title(f"Cumulative {flux_type.capitalize()} Flux Π(k)")
    plt.grid(True, which="both", alpha=0.3, linestyle="--")
    plt.legend(ncol=2, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / f"{flux_type}_flux.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_snapshot(snapshot_path, write_index=0, tasks=None, outdir=".", dpi=300):
    """
    Plot 2D fields from a single snapshot.

    Args:
        snapshot_path (str or Path): Path to snapshot HDF5 file
        write_index (int): Write index to plot
        tasks (list or None): List of task names to plot (default: all available)
        outdir (str or Path): Output directory
        dpi (int): Figure DPI

    Requires:
        dedalus.extras.plot_tools (for advanced plotting)

    Generates:
        - snapshot_write_{write_number:06d}.png
    """
    if not DEDALUS_AVAILABLE:
        raise ImportError("Dedalus plot_tools not available. Cannot plot snapshots.")

    import h5py
    from . import io

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Get available tasks
    if tasks is None:
        tasks = io.list_snapshot_tasks(snapshot_path)

    # Default to velocity-related fields if available
    default_tasks = ["vorticity", "pressure", "streamfunction"]
    tasks = [t for t in default_tasks if t in tasks] if tasks is None else tasks

    if not tasks:
        raise ValueError("No tasks specified or found in snapshot file")

    nrows, ncols = 1, len(tasks)
    scale = 2.5
    image = plot_tools.Box(1, 2)
    pad = plot_tools.Frame(0.2, 0.0, 0.0, 0.0)
    margin = plot_tools.Frame(0.2, 0.1, 0.0, 0.0)

    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    with h5py.File(snapshot_path, "r") as f:
        times = np.array(f["scales/sim_time"])
        writes = np.array(f["scales/write_number"])

        if write_index >= len(times):
            raise IndexError(f"Write index {write_index} out of range (max: {len(times)-1})")

        # Compute symmetric color limits
        clims = {}
        for task in tasks:
            if f"tasks/{task}" not in f:
                raise KeyError(f"Task '{task}' not found in snapshot file")
            data = np.array(f[f"tasks/{task}"][write_index])
            abs_max = max(abs(data.min()), abs(data.max()))
            clims[task] = (-abs_max, abs_max)

        # Plot fields
        for n, task in enumerate(tasks):
            i, j = divmod(n, ncols)
            ax = mfig.add_axes(i, j, [0, 0, 1, 1])
            dset = f[f"tasks/{task}"]
            plot_tools.plot_bot_3d(dset, 0, write_index, axes=ax,
                                  title=task, clim=clims[task], visible_axes=False)

        # Title
        tstr = f"t = {times[write_index]:.3f}"
        title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
        fig.suptitle(tstr, x=0.45, y=title_height, ha="left")

        # Save
        savename = f"snapshot_write_{int(writes[write_index]):06d}.png"
        fig.savefig(str(outdir / savename), dpi=dpi, bbox_inches="tight")

    plt.close(fig)
