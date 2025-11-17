#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare snapshot fields between predictions and ground truth in a 2x3 layout.

This script reads velocity (u, v) and pressure arrays from an npz file
(containing predictions and ground truth) and creates comparison plots in a
2x3 layout:
    Top row:    Ground Truth u | Ground Truth v | Ground Truth Pressure
    Bottom row: Prediction u   | Prediction v   | Prediction Pressure

Usage:
    python compare_snapshots.py --pred_path path/to/predictions.npz \\
                                 --outdir ./snapshot_comparison \\
                                 --snap_start 0 \\
                                 --snap_stride 10 \\
                                 --snap_count 20

    python compare_snapshots.py --pred_path test_data_prediction.npz \\
                                 --outdir ./snapshots \\
                                 --dpi 150

For help:
    python compare_snapshots.py --help
"""

import argparse
import pathlib
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm

# Use non-interactive backend for batch processing
matplotlib.use("Agg")


def get_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Compare prediction and ground truth snapshots in 2x2 layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    ap.add_argument("--pred_path", type=str, required=True,
                   help="Path to npz file containing predictions and ground truth")

    # Output
    ap.add_argument("--outdir", type=str, default="./snapshot_comparison",
                   help="Output directory for comparison figures")
    ap.add_argument("--dpi", type=int, default=200,
                   help="Figure DPI (resolution)")

    # Snapshot selection
    ap.add_argument("--snap_start", type=int, default=0,
                   help="Starting snapshot index")
    ap.add_argument("--snap_count", type=int, default=0,
                   help="Number of snapshots to plot (0 = all)")
    ap.add_argument("--snap_stride", type=int, default=1,
                   help="Stride between snapshots")

    # Time parameters
    ap.add_argument("--dt", type=float, default=0.2,
                   help="Time step between snapshots (for labeling)")
    ap.add_argument("--t_start", type=float, default=0.0,
                   help="Starting time for time array (for labeling)")

    # Color limits
    ap.add_argument("--u_clim", type=float, default=None,
                   help="Symmetric color limit for u velocity (auto if not specified)")
    ap.add_argument("--v_clim", type=float, default=None,
                   help="Symmetric color limit for v velocity (auto if not specified)")
    ap.add_argument("--pressure_clim", type=float, default=None,
                   help="Symmetric color limit for pressure (auto if not specified)")

    # Colormap
    ap.add_argument("--cmap", type=str, default="RdBu_r",
                   help="Colormap for field plots")

    return ap.parse_args()


def load_data_from_npz(npz_path):
    """
    Load prediction and ground truth data from npz file.

    Args:
        npz_path (Path): Path to npz file

    Returns:
        tuple: (pred_u, pred_v, pred_pressure, true_u, true_v, true_pressure)
            Each is a numpy array of shape (T, Nx, Ny)
    """
    data = np.load(npz_path, allow_pickle=True)

    # Load predictions
    pred_u = data['pred_velocity_x']  # (T, Nx, Ny)
    pred_v = data['pred_velocity_y']  # (T, Nx, Ny)
    pred_pressure = data['pred_pressure']  # (T, Nx, Ny)

    # Try to load ground truth from different possible keys
    truth_keys_u = ['output_u', 'true_u', 'gt_u', 'output_velocity_x']
    truth_keys_v = ['output_v', 'true_v', 'gt_v', 'output_velocity_y']
    truth_keys_p = ['output_pressure', 'true_pressure', 'gt_pressure', 'output_p', 'true_p', 'gt_p']

    true_u = None
    true_v = None
    true_pressure = None

    for key in truth_keys_u:
        if key in data:
            true_u = data[key]
            break

    for key in truth_keys_v:
        if key in data:
            true_v = data[key]
            break

    for key in truth_keys_p:
        if key in data:
            true_pressure = data[key]
            break

    if true_u is None or true_v is None or true_pressure is None:
        raise KeyError(
            f"Could not find ground truth data in npz file.\n"
            f"Looked for u keys: {truth_keys_u}\n"
            f"Looked for v keys: {truth_keys_v}\n"
            f"Looked for pressure keys: {truth_keys_p}\n"
            f"Available keys: {list(data.keys())}"
        )

    return pred_u, pred_v, pred_pressure, true_u, true_v, true_pressure


def compute_global_clims(pred_u, pred_v, pred_pressure,
                         true_u, true_v, true_pressure,
                         u_clim=None, v_clim=None, pressure_clim=None):
    """
    Compute global symmetric color limits for consistent visualisation.

    Args:
        pred_u (ndarray): Prediction u velocity (T, Nx, Ny)
        pred_v (ndarray): Prediction v velocity (T, Nx, Ny)
        pred_pressure (ndarray): Prediction pressure (T, Nx, Ny)
        true_u (ndarray): Ground truth u velocity (T, Nx, Ny)
        true_v (ndarray): Ground truth v velocity (T, Nx, Ny)
        true_pressure (ndarray): Ground truth pressure (T, Nx, Ny)
        u_clim (float or None): Manual u velocity limit
        v_clim (float or None): Manual v velocity limit
        pressure_clim (float or None): Manual pressure limit

    Returns:
        dict: Color limits {"u": (vmin, vmax), "v": (vmin, vmax), "pressure": (vmin, vmax)}
    """
    clims = {}

    # u velocity color limits
    if u_clim is not None:
        clims["u"] = (-u_clim, u_clim)
    else:
        # Compute from both datasets
        umax_pred = np.nanmax(np.abs(pred_u[np.isfinite(pred_u)]))
        umax_truth = np.nanmax(np.abs(true_u[np.isfinite(true_u)]))
        umax = max(umax_pred, umax_truth)
        clims["u"] = (-umax, umax)

    # v velocity color limits
    if v_clim is not None:
        clims["v"] = (-v_clim, v_clim)
    else:
        # Compute from both datasets
        vmax_pred = np.nanmax(np.abs(pred_v[np.isfinite(pred_v)]))
        vmax_truth = np.nanmax(np.abs(true_v[np.isfinite(true_v)]))
        vmax = max(vmax_pred, vmax_truth)
        clims["v"] = (-vmax, vmax)

    # Pressure color limits
    if pressure_clim is not None:
        clims["pressure"] = (-pressure_clim, pressure_clim)
    else:
        # Compute from both datasets
        pmax_pred = np.nanmax(np.abs(pred_pressure[np.isfinite(pred_pressure)]))
        pmax_truth = np.nanmax(np.abs(true_pressure[np.isfinite(true_pressure)]))
        pmax = max(pmax_pred, pmax_truth)
        clims["pressure"] = (-pmax, pmax)

    return clims


def plot_snapshot_comparison(pred_u, pred_v, pred_p, true_u, true_v, true_p,
                             time, clims, cmap, outdir, dpi, snapshot_idx):
    """
    Create 2x3 comparison plot for a single time snapshot.

    Layout:
        [Ground Truth u] [Ground Truth v] [Ground Truth Pressure]
        [Prediction u]   [Prediction v]   [Prediction Pressure]

    Args:
        pred_u (ndarray): Prediction u velocity (Nx, Ny)
        pred_v (ndarray): Prediction v velocity (Nx, Ny)
        pred_p (ndarray): Prediction pressure (Nx, Ny)
        true_u (ndarray): Ground truth u velocity (Nx, Ny)
        true_v (ndarray): Ground truth v velocity (Nx, Ny)
        true_p (ndarray): Ground truth pressure (Nx, Ny)
        time (float): Simulation time
        clims (dict): Color limits for each field
        cmap (str): Colormap name
        outdir (Path): Output directory
        dpi (int): Figure DPI
        snapshot_idx (int): Snapshot index for filename
    """
    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                          hspace=0.30, wspace=0.30,
                          left=0.05, right=0.95, top=0.93, bottom=0.05)

    # Titles for each subplot
    titles = [
        "Ground Truth: u",
        "Ground Truth: v",
        "Ground Truth: Pressure",
        "Prediction: u",
        "Prediction: v",
        "Prediction: Pressure"
    ]

    # Data for each subplot
    fields = [true_u, true_v, true_p, pred_u, pred_v, pred_p]
    field_names = ["u", "v", "pressure", "u", "v", "pressure"]

    # Plot each field
    for idx, (field, field_name, title) in enumerate(zip(fields, field_names, titles)):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        vmin, vmax = clims[field_name]

        # Create image
        im = ax.imshow(field.T, origin='lower', cmap=cmap,
                      vmin=vmin, vmax=vmax, aspect='auto', interpolation='bilinear')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        # Set title and labels
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        ax.tick_params(labelsize=9)

    # Overall title with time
    fig.suptitle(f"t = {time:.3f}", fontsize=14, fontweight='bold')

    # Save figure
    savename = f"comparison_snapshot_{snapshot_idx:06d}.png"
    fig.savefig(outdir / savename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main execution function."""
    args = get_args()

    pred_path = pathlib.Path(args.pred_path).resolve()
    outdir = pathlib.Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NS2D Snapshot Comparison")
    print("=" * 70)
    print(f"Input file: {pred_path}")
    print(f"Output directory: {outdir}")
    print(f"DPI: {args.dpi}")
    print(f"Colormap: {args.cmap}")
    print("=" * 70)

    # Load data
    print("\nLoading data from npz file...")
    pred_u, pred_v, pred_pressure, true_u, true_v, true_pressure = \
        load_data_from_npz(pred_path)

    print(f"Prediction shape: {pred_u.shape}")
    print(f"Ground truth shape: {true_u.shape}")

    T = pred_u.shape[0]

    # Create time array
    times = args.t_start + np.arange(T) * args.dt

    # Compute global color limits
    print("\nComputing global color limits...")
    clims = compute_global_clims(
        pred_u, pred_v, pred_pressure,
        true_u, true_v, true_pressure,
        args.u_clim, args.v_clim, args.pressure_clim
    )
    print(f"  u range: [{clims['u'][0]:.3e}, {clims['u'][1]:.3e}]")
    print(f"  v range: [{clims['v'][0]:.3e}, {clims['v'][1]:.3e}]")
    print(f"  Pressure range: [{clims['pressure'][0]:.3e}, {clims['pressure'][1]:.3e}]")

    # Determine which snapshots to plot
    stride = max(1, args.snap_stride)
    start = min(max(0, args.snap_start), max(0, T - 1))
    candidates = list(range(start, T, stride))

    if args.snap_count > 0:
        snapshot_indices = candidates[:args.snap_count]
    else:
        snapshot_indices = candidates

    total = len(snapshot_indices)
    print(f"\nPlotting {total} snapshot comparisons...")

    # Plot each snapshot with progress bar
    for snap_idx in tqdm(snapshot_indices, desc="Creating comparison plots"):
        plot_snapshot_comparison(
            pred_u=pred_u[snap_idx],
            pred_v=pred_v[snap_idx],
            pred_p=pred_pressure[snap_idx],
            true_u=true_u[snap_idx],
            true_v=true_v[snap_idx],
            true_p=true_pressure[snap_idx],
            time=times[snap_idx],
            clims=clims,
            cmap=args.cmap,
            outdir=outdir,
            dpi=args.dpi,
            snapshot_idx=snap_idx
        )

    print("\n" + "=" * 70)
    print("Snapshot comparison complete!")
    print(f"Generated {total} comparison plots")
    print(f"Output saved to: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
