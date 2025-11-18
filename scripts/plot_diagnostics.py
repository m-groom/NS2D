#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot comparison of NS2D diagnostics between predictions and ground truth.

This script reads diagnostic HDF5 files produced by compute_diagnostics.py
(containing predictions and ground truth separately) and creates comparison
plots showing both datasets on the same axes.

Usage:
    python plot_diagnostics.py --pred_diag path/to/diagnostics_predictions.h5 \\
                                --truth_diag path/to/diagnostics_groundtruth.h5 \\
                                --outdir ./comparison_plots

    python plot_diagnostics.py --pred_diag diagnostics_predictions.h5 \\
                                --truth_diag diagnostics_groundtruth.h5 \\
                                --outdir ./plots \\
                                --dpi 150 \\
                                --no_flux

For help:
    python plot_diagnostics.py --help
"""

import argparse
import pathlib
import sys
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for batch processing
matplotlib.use("Agg")

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def get_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Plot comparison of predictions vs ground truth diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    ap.add_argument("--pred_diag", type=str, required=True,
                   help="Path to predictions diagnostic HDF5 file")
    ap.add_argument("--truth_diag", type=str, required=True,
                   help="Path to ground truth diagnostic HDF5 file")

    # Output
    ap.add_argument("--outdir", type=str, default="./comparison_plots",
                   help="Output directory for comparison figures")
    ap.add_argument("--dpi", type=int, default=300,
                   help="Figure DPI (resolution)")

    # Domain parameters (for Nyquist calculation)
    ap.add_argument("--Lx", type=float, default=2*np.pi,
                   help="Domain length in x (for Nyquist calculation)")
    ap.add_argument("--Nx", type=int, default=None,
                   help="Grid resolution in x (for Nyquist calculation)")

    # What to plot
    ap.add_argument("--no_scalars", action="store_true",
                   help="Skip scalar time-series comparison plots")
    ap.add_argument("--no_spectra", action="store_true",
                   help="Skip spectra comparison plots")
    ap.add_argument("--no_flux", action="store_true",
                   help="Skip flux comparison plots")

    # Spectra/flux options
    ap.add_argument("--spectra_max_curves", type=int, default=4,
                   help="Maximum number of time curves to overlay per dataset")
    ap.add_argument("--spectra_loglog", action="store_true",
                   help="Use log-log axes for spectra plots")

    return ap.parse_args()


def load_scalars_from_h5(h5_path):
    """
    Load scalar time series from diagnostic HDF5 file.

    Args:
        h5_path (Path): Path to diagnostic HDF5 file

    Returns:
        tuple: (times, series_dict)
            - times: (N,) array of simulation times
            - series_dict: Dictionary of scalar arrays {name: (N,) array}
    """
    series_dict = {}

    with h5py.File(h5_path, 'r') as f:
        if 'scalars' not in f:
            raise KeyError(f"'scalars' group not found in {h5_path}")

        scalars_grp = f['scalars']
        times = np.array(scalars_grp['sim_time'])

        # Load all scalar datasets
        for key in scalars_grp.keys():
            if key != 'sim_time':
                series_dict[key] = np.array(scalars_grp[key])

    return times, series_dict


def load_spectra_from_h5(h5_path):
    """
    Load spectra from diagnostic HDF5 file.

    Args:
        h5_path (Path): Path to diagnostic HDF5 file

    Returns:
        tuple: (times, kbins, Ek_list, Zk_list)
            - times: (T,) array of snapshot times
            - kbins: (M,) array of wavenumber bins
            - Ek_list: List of T energy spectra, each (M,)
            - Zk_list: List of T enstrophy spectra, each (M,)
    """
    times, Ek_list, Zk_list, kbins = [], [], [], None

    with h5py.File(h5_path, 'r') as f:
        if 'spectra' not in f:
            raise KeyError(f"'spectra' group not found in {h5_path}")

        spectra_grp = f['spectra']

        # Read all k_E_Z datasets
        for name in sorted(spectra_grp.keys()):
            if not name.startswith('k_E_Z_t'):
                continue

            # Extract time from dataset name
            t_str = name.replace('k_E_Z_t', '')
            t = float(t_str)

            # Load data: shape (M, 3) = [k, E(k), Z(k)]
            arr = np.array(spectra_grp[name])

            if kbins is None:
                kbins = arr[:, 0]

            times.append(t)
            Ek_list.append(arr[:, 1])
            Zk_list.append(arr[:, 2])

    times = np.array(times)
    order = np.argsort(times)
    times = times[order]
    Ek_list = [Ek_list[i] for i in order]
    Zk_list = [Zk_list[i] for i in order]

    return times, kbins, Ek_list, Zk_list


def load_flux_from_h5(h5_path, flux_type="energy"):
    """
    Load flux data from diagnostic HDF5 file.

    Args:
        h5_path (Path): Path to diagnostic HDF5 file
        flux_type (str): "energy" or "enstrophy"

    Returns:
        tuple: (times, kbins, T_list, Pi_list)
            - times: (T,) array of snapshot times
            - kbins: (M,) array of wavenumber bins
            - T_list: List of T transfer spectra, each (M,)
            - Pi_list: List of T cumulative flux, each (M,)
    """
    if flux_type == "energy":
        pattern = "flux_T_Pi_t"
    elif flux_type == "enstrophy":
        pattern = "enstrophy_flux_T_Pi_t"
    else:
        raise ValueError(f"Invalid flux_type '{flux_type}'. Use 'energy' or 'enstrophy'.")

    times, T_list, Pi_list, kbins = [], [], [], None

    with h5py.File(h5_path, 'r') as f:
        if 'flux' not in f:
            raise KeyError(f"'flux' group not found in {h5_path}")

        flux_grp = f['flux']

        for name in sorted(flux_grp.keys()):
            if not name.startswith(pattern):
                continue

            # Extract time from dataset name
            t_str = name.replace(pattern, '')
            t = float(t_str)

            # Load data: shape (M, 3) = [k, T(k), Pi(k)]
            arr = np.array(flux_grp[name])

            if kbins is None:
                kbins = arr[:, 0]

            times.append(t)
            T_list.append(arr[:, 1])
            Pi_list.append(arr[:, 2])

    times = np.array(times)
    order = np.argsort(times)
    times = times[order]
    T_list = [T_list[i] for i in order]
    Pi_list = [Pi_list[i] for i in order]

    return times, kbins, T_list, Pi_list


def plot_scalar_comparison(times_pred, series_pred, times_truth, series_truth,
                           outdir=".", dpi=300):
    """
    Plot scalar time series comparison (predictions vs ground truth).

    Args:
        times_pred (ndarray): Prediction time values (N,)
        series_pred (dict): Prediction scalar arrays {name: (N,) array}
        times_truth (ndarray): Ground truth time values (N,)
        series_truth (dict): Ground truth scalar arrays {name: (N,) array}
        outdir (str or Path): Output directory for figures
        dpi (int): Figure DPI
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Standard plots
    plot_specs = {
        "energy": {"ylabel": "Energy E", "title": "Kinetic Energy vs Time"},
        "enstrophy": {"ylabel": "Enstrophy Z", "title": "Enstrophy vs Time"},
        "palinstrophy": {"ylabel": "Palinstrophy P", "title": "Palinstrophy vs Time"},
        "Re_lambda": {"ylabel": "Re_λ", "title": "Taylor Reynolds Number vs Time"},
        # Energy budget terms
        "energy_injection": {"ylabel": "ε_inj", "title": "Energy Injection Rate vs Time"},
        "visc_loss": {"ylabel": "ε_visc", "title": "Energy Viscous Loss vs Time"},
        "drag_loss": {"ylabel": "ε_drag", "title": "Energy Drag Loss vs Time"},
        "energy_balance": {"ylabel": "ε_balance", "title": "Energy Budget Balance vs Time"},
        # Enstrophy budget terms
        "enstrophy_injection": {"ylabel": "Z_inj", "title": "Enstrophy Injection Rate vs Time"},
        "enstrophy_drag_loss": {"ylabel": "Z_drag", "title": "Enstrophy Drag Loss vs Time"},
        "enstrophy_visc_loss": {"ylabel": "Z_visc", "title": "Enstrophy Viscous Loss vs Time"},
        "enstrophy_balance": {"ylabel": "Z_balance", "title": "Enstrophy Budget Balance vs Time"},
    }

    # Get common keys
    common_keys = set(series_pred.keys()) & set(series_truth.keys())

    for key in plot_specs.keys():
        if key not in common_keys:
            continue

        plt.figure(figsize=(8, 4.5))

        # Plot ground truth (solid line)
        plt.plot(times_truth, series_truth[key],
                linewidth=1.5, linestyle='-', label='Ground Truth', alpha=0.8)

        # Plot prediction (dashed line)
        plt.plot(times_pred, series_pred[key],
                linewidth=1.5, linestyle='--', label='Prediction', alpha=0.8)

        plt.xlabel("Time t")
        plt.ylabel(plot_specs[key]["ylabel"])
        plt.title(plot_specs[key]["title"])
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(outdir / f"{key}_comparison.png", dpi=dpi, bbox_inches="tight")
        plt.close()


def plot_spectra_comparison(times_pred, kbins_pred, Ek_pred_list, Zk_pred_list,
                            times_truth, kbins_truth, Ek_truth_list, Zk_truth_list,
                            outdir=".", max_curves=4, loglog=True, k_nyquist=None, dpi=300):
    """
    Plot energy and enstrophy spectra comparison.

    Args:
        times_pred (ndarray): Prediction snapshot times (T,)
        kbins_pred (ndarray): Prediction wavenumber bins (M,)
        Ek_pred_list (list): Prediction energy spectra
        Zk_pred_list (list): Prediction enstrophy spectra
        times_truth (ndarray): Ground truth snapshot times (T,)
        kbins_truth (ndarray): Ground truth wavenumber bins (M,)
        Ek_truth_list (list): Ground truth energy spectra
        Zk_truth_list (list): Ground truth enstrophy spectra
        outdir (str or Path): Output directory
        max_curves (int): Maximum number of curves per dataset
        loglog (bool): Use log-log axes
        k_nyquist (float or None): Clip at Nyquist wavenumber
        dpi (int): Figure DPI
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Select time indices to plot (same times for both datasets)
    n_times = min(len(times_pred), len(times_truth))
    idxs = np.linspace(0, n_times - 1, num=min(max_curves, n_times), dtype=int)

    # Stack spectra
    Ek_pred_stack = np.stack(Ek_pred_list, axis=0)  # (T, M)
    Zk_pred_stack = np.stack(Zk_pred_list, axis=0)
    Ek_truth_stack = np.stack(Ek_truth_list, axis=0)
    Zk_truth_stack = np.stack(Zk_truth_list, axis=0)

    # Apply Nyquist clipping to prediction data
    kbins_pred_plot = kbins_pred
    if k_nyquist is not None:
        mask_pred = kbins_pred <= k_nyquist
        kbins_pred_plot = kbins_pred[mask_pred]
        Ek_pred_stack = Ek_pred_stack[:, mask_pred]
        Zk_pred_stack = Zk_pred_stack[:, mask_pred]

    # Apply Nyquist clipping to ground truth data
    kbins_truth_plot = kbins_truth
    if k_nyquist is not None:
        mask_truth = kbins_truth <= k_nyquist
        kbins_truth_plot = kbins_truth[mask_truth]
        Ek_truth_stack = Ek_truth_stack[:, mask_truth]
        Zk_truth_stack = Zk_truth_stack[:, mask_truth]

    # Color map for different times
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(idxs)))

    # ========== Energy spectrum ==========
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    if loglog:
        start_pred = 1 if kbins_pred_plot.size > 0 and kbins_pred_plot[0] == 0 else 0
        start_truth = 1 if kbins_truth_plot.size > 0 and kbins_truth_plot[0] == 0 else 0

        for plot_idx, i in enumerate(idxs):
            # Ground truth (solid)
            ax.loglog(kbins_truth_plot[start_truth:], Ek_truth_stack[i][start_truth:],
                     color=colors[plot_idx], linestyle='-', linewidth=2.0,
                     alpha=0.7, label=f"Truth t={times_truth[i]:.2f}")

            # Prediction (dashed)
            ax.loglog(kbins_pred_plot[start_pred:], Ek_pred_stack[i][start_pred:],
                     color=colors[plot_idx], linestyle='--', linewidth=2.0,
                     alpha=0.7, label=f"Pred t={times_pred[i]:.2f}")

        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("E(k)")
    else:
        for plot_idx, i in enumerate(idxs):
            # Ground truth (solid)
            ax.plot(kbins_truth_plot, Ek_truth_stack[i],
                   color=colors[plot_idx], linestyle='-', linewidth=2.0,
                   alpha=0.7, label=f"Truth t={times_truth[i]:.2f}")

            # Prediction (dashed)
            ax.plot(kbins_pred_plot, Ek_pred_stack[i],
                   color=colors[plot_idx], linestyle='--', linewidth=2.0,
                   alpha=0.7, label=f"Pred t={times_pred[i]:.2f}")

        plt.xlabel("Wavenumber k")
        plt.ylabel("E(k)")

    plt.title("Energy Spectrum E(k): Prediction vs Ground Truth")
    plt.grid(True, which="both", alpha=0.3, linestyle=":")
    plt.legend(ncol=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "energy_spectrum_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # ========== Enstrophy spectrum ==========
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    if loglog:
        start_pred = 1 if kbins_pred_plot.size > 0 and kbins_pred_plot[0] == 0 else 0
        start_truth = 1 if kbins_truth_plot.size > 0 and kbins_truth_plot[0] == 0 else 0

        for plot_idx, i in enumerate(idxs):
            # Ground truth (solid)
            ax.loglog(kbins_truth_plot[start_truth:], Zk_truth_stack[i][start_truth:],
                     color=colors[plot_idx], linestyle='-', linewidth=2.0,
                     alpha=0.7, label=f"Truth t={times_truth[i]:.2f}")

            # Prediction (dashed)
            ax.loglog(kbins_pred_plot[start_pred:], Zk_pred_stack[i][start_pred:],
                     color=colors[plot_idx], linestyle='--', linewidth=2.0,
                     alpha=0.7, label=f"Pred t={times_pred[i]:.2f}")

        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("Z(k)")
    else:
        for plot_idx, i in enumerate(idxs):
            # Ground truth (solid)
            ax.plot(kbins_truth_plot, Zk_truth_stack[i],
                   color=colors[plot_idx], linestyle='-', linewidth=2.0,
                   alpha=0.7, label=f"Truth t={times_truth[i]:.2f}")

            # Prediction (dashed)
            ax.plot(kbins_pred_plot, Zk_pred_stack[i],
                   color=colors[plot_idx], linestyle='--', linewidth=2.0,
                   alpha=0.7, label=f"Pred t={times_pred[i]:.2f}")

        plt.xlabel("Wavenumber k")
        plt.ylabel("Z(k)")

    plt.title("Enstrophy Spectrum Z(k): Prediction vs Ground Truth")
    plt.grid(True, which="both", alpha=0.3, linestyle=":")
    plt.legend(ncol=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "enstrophy_spectrum_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_flux_comparison(times_pred, kbins_pred, T_pred_list, Pi_pred_list,
                        times_truth, kbins_truth, T_truth_list, Pi_truth_list,
                        outdir=".", flux_type="energy", max_curves=4,
                        k_nyquist=None, dpi=300):
    """
    Plot spectral transfer and cumulative flux comparison.

    Args:
        times_pred (ndarray): Prediction snapshot times (T,)
        kbins_pred (ndarray): Prediction wavenumber bins (M,)
        T_pred_list (list): Prediction transfer spectra
        Pi_pred_list (list): Prediction cumulative flux
        times_truth (ndarray): Ground truth snapshot times (T,)
        kbins_truth (ndarray): Ground truth wavenumber bins (M,)
        T_truth_list (list): Ground truth transfer spectra
        Pi_truth_list (list): Ground truth cumulative flux
        outdir (str or Path): Output directory
        flux_type (str): "energy" or "enstrophy"
        max_curves (int): Maximum curves per dataset
        k_nyquist (float or None): Clip at Nyquist wavenumber
        dpi (int): Figure DPI
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Select time indices
    n_times = min(len(times_pred), len(times_truth))
    idxs = np.linspace(0, n_times - 1, num=min(max_curves, n_times), dtype=int)

    # Stack
    T_pred_stack = np.stack(T_pred_list, axis=0)
    Pi_pred_stack = np.stack(Pi_pred_list, axis=0)
    T_truth_stack = np.stack(T_truth_list, axis=0)
    Pi_truth_stack = np.stack(Pi_truth_list, axis=0)

    # Apply Nyquist clipping
    kbins_pred_plot = kbins_pred
    if k_nyquist is not None:
        mask_pred = kbins_pred <= k_nyquist
        kbins_pred_plot = kbins_pred[mask_pred]
        T_pred_stack = T_pred_stack[:, mask_pred]
        Pi_pred_stack = Pi_pred_stack[:, mask_pred]

    kbins_truth_plot = kbins_truth
    if k_nyquist is not None:
        mask_truth = kbins_truth <= k_nyquist
        kbins_truth_plot = kbins_truth[mask_truth]
        T_truth_stack = T_truth_stack[:, mask_truth]
        Pi_truth_stack = Pi_truth_stack[:, mask_truth]

    # Use positive k for log scale
    kpos_mask_pred = kbins_pred_plot > 0
    kpos_mask_truth = kbins_truth_plot > 0

    # Color map
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(idxs)))

    # ========== Transfer T(k) ==========
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for plot_idx, i in enumerate(idxs):
        # Ground truth (solid)
        ax.plot(kbins_truth_plot[kpos_mask_truth], T_truth_stack[i][kpos_mask_truth],
               color=colors[plot_idx], linestyle='-', linewidth=2.0,
               alpha=0.7, label=f"Truth t={times_truth[i]:.2f}")

        # Prediction (dashed)
        ax.plot(kbins_pred_plot[kpos_mask_pred], T_pred_stack[i][kpos_mask_pred],
               color=colors[plot_idx], linestyle='--', linewidth=2.0,
               alpha=0.7, label=f"Pred t={times_pred[i]:.2f}")

    ax.set_xscale('log')
    plt.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    plt.xlabel("Wavenumber k")
    plt.ylabel("T(k)")
    plt.title(f"{flux_type.capitalize()} Transfer T(k): Prediction vs Ground Truth")
    plt.grid(True, which="both", alpha=0.3, linestyle=":")
    plt.legend(ncol=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / f"{flux_type}_transfer_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # ========== Cumulative flux Π(k) ==========
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for plot_idx, i in enumerate(idxs):
        # Ground truth (solid)
        ax.plot(kbins_truth_plot[kpos_mask_truth], Pi_truth_stack[i][kpos_mask_truth],
               color=colors[plot_idx], linestyle='-', linewidth=2.0,
               alpha=0.7, label=f"Truth t={times_truth[i]:.2f}")

        # Prediction (dashed)
        ax.plot(kbins_pred_plot[kpos_mask_pred], Pi_pred_stack[i][kpos_mask_pred],
               color=colors[plot_idx], linestyle='--', linewidth=2.0,
               alpha=0.7, label=f"Pred t={times_pred[i]:.2f}")

    ax.set_xscale('log')
    plt.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    plt.xlabel("Wavenumber k")
    plt.ylabel("Π(k)")
    plt.title(f"Cumulative {flux_type.capitalize()} Flux Π(k): Prediction vs Ground Truth")
    plt.grid(True, which="both", alpha=0.3, linestyle=":")
    plt.legend(ncol=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / f"{flux_type}_flux_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    """Main execution function."""
    args = get_args()

    pred_path = pathlib.Path(args.pred_diag).resolve()
    truth_path = pathlib.Path(args.truth_diag).resolve()
    out_root = pathlib.Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NS2D Diagnostic Comparison Plotting")
    print("=" * 70)
    print(f"Prediction diagnostics: {pred_path}")
    print(f"Ground truth diagnostics: {truth_path}")
    print(f"Output directory: {out_root}")
    print(f"DPI: {args.dpi}")
    print("=" * 70)

    # Calculate Nyquist wavenumber if Nx provided
    k_nyquist = None
    if args.Nx is not None:
        k_nyquist = (np.pi * args.Nx) / args.Lx
        print(f"Nyquist wavenumber: {k_nyquist:.2f}")

    # 1) Scalar time series comparison
    if not args.no_scalars:
        print("\n[1/3] Plotting scalar time series comparison...")
        try:
            times_pred, series_pred = load_scalars_from_h5(pred_path)
            times_truth, series_truth = load_scalars_from_h5(truth_path)

            print(f"  Prediction: {len(times_pred)} time points")
            print(f"  Ground truth: {len(times_truth)} time points")
            print(f"  Common scalars: {', '.join(set(series_pred.keys()) & set(series_truth.keys()))}")

            plot_scalar_comparison(
                times_pred, series_pred,
                times_truth, series_truth,
                outdir=out_root / "scalars",
                dpi=args.dpi
            )
            print(f"  Saved to {out_root / 'scalars'}")

        except Exception as e:
            print(f"  Error plotting scalar comparison: {e}")

    # 2) Spectra comparison
    if not args.no_spectra:
        print("\n[2/3] Plotting spectra comparison...")
        try:
            times_pred, kbins_pred, Ek_pred, Zk_pred = load_spectra_from_h5(pred_path)
            times_truth, kbins_truth, Ek_truth, Zk_truth = load_spectra_from_h5(truth_path)

            print(f"  Prediction: {len(times_pred)} spectra snapshots")
            print(f"  Ground truth: {len(times_truth)} spectra snapshots")
            print(f"  Wavenumber range (pred): [{kbins_pred[1]:.2f}, {kbins_pred[-1]:.2f}]")
            print(f"  Wavenumber range (truth): [{kbins_truth[1]:.2f}, {kbins_truth[-1]:.2f}]")

            plot_spectra_comparison(
                times_pred, kbins_pred, Ek_pred, Zk_pred,
                times_truth, kbins_truth, Ek_truth, Zk_truth,
                outdir=out_root / "spectra",
                max_curves=args.spectra_max_curves,
                loglog=args.spectra_loglog,
                k_nyquist=k_nyquist,
                dpi=args.dpi
            )
            print(f"  Saved to {out_root / 'spectra'}")

        except Exception as e:
            print(f"  Error plotting spectra comparison: {e}")

    # 3) Flux comparison
    if not args.no_flux:
        print("\n[3/3] Plotting flux comparison...")

        # Energy flux
        try:
            times_pred_e, kbins_pred_e, T_pred_e, Pi_pred_e = load_flux_from_h5(pred_path, flux_type="energy")
            times_truth_e, kbins_truth_e, T_truth_e, Pi_truth_e = load_flux_from_h5(truth_path, flux_type="energy")

            print(f"  Prediction: {len(times_pred_e)} energy flux snapshots")
            print(f"  Ground truth: {len(times_truth_e)} energy flux snapshots")

            plot_flux_comparison(
                times_pred_e, kbins_pred_e, T_pred_e, Pi_pred_e,
                times_truth_e, kbins_truth_e, T_truth_e, Pi_truth_e,
                outdir=out_root / "flux",
                flux_type="energy",
                max_curves=args.spectra_max_curves,
                k_nyquist=k_nyquist,
                dpi=args.dpi
            )
            print(f"  Saved energy flux to {out_root / 'flux'}")

        except Exception as e:
            print(f"  Error plotting energy flux comparison: {e}")

        # Enstrophy flux
        try:
            times_pred_z, kbins_pred_z, T_pred_z, Pi_pred_z = load_flux_from_h5(pred_path, flux_type="enstrophy")
            times_truth_z, kbins_truth_z, T_truth_z, Pi_truth_z = load_flux_from_h5(truth_path, flux_type="enstrophy")

            print(f"  Prediction: {len(times_pred_z)} enstrophy flux snapshots")
            print(f"  Ground truth: {len(times_truth_z)} enstrophy flux snapshots")

            plot_flux_comparison(
                times_pred_z, kbins_pred_z, T_pred_z, Pi_pred_z,
                times_truth_z, kbins_truth_z, T_truth_z, Pi_truth_z,
                outdir=out_root / "flux",
                flux_type="enstrophy",
                max_curves=args.spectra_max_curves,
                k_nyquist=k_nyquist,
                dpi=args.dpi
            )
            print(f"  Saved enstrophy flux to {out_root / 'flux'}")

        except Exception as e:
            print(f"  Error plotting enstrophy flux comparison: {e}")

    print("\n" + "=" * 70)
    print("Comparison plotting complete!")
    print(f"All figures saved to: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
