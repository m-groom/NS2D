#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare snapshot fields between predictions and ground truth in a 2x2 layout.

This script reads vorticity and streamfunction arrays from an npz file
(containing predictions and ground truth) and creates comparison plots in a
2x2 layout:
    Top row:    Ground Truth Vorticity | Ground Truth Streamfunction
    Bottom row: Prediction Vorticity   | Prediction Streamfunction

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
    ap.add_argument("--vorticity_clim", type=float, default=None,
                   help="Symmetric color limit for vorticity (auto if not specified)")
    ap.add_argument("--streamfunction_clim", type=float, default=None,
                   help="Symmetric color limit for streamfunction (auto if not specified)")

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
        tuple: (pred_vorticity, pred_streamfunction, true_vorticity, true_streamfunction)
            Each is a numpy array of shape (T, Nx, Ny)
    """
    data = np.load(npz_path, allow_pickle=True)

    # Load predictions
    pred_vorticity = data['pred_vorticity']  # (T, Nx, Ny)
    pred_streamfunction = data['pred_streamfunction']  # (T, Nx, Ny)

    # Try to load ground truth from different possible keys
    truth_keys_vort = ['output_vorticity', 'true_vorticity', 'gt_vorticity']
    truth_keys_psi = ['output_streamfunction', 'true_streamfunction', 'gt_streamfunction']

    true_vorticity = None
    true_streamfunction = None

    for key in truth_keys_vort:
        if key in data:
            true_vorticity = data[key]
            break

    for key in truth_keys_psi:
        if key in data:
            true_streamfunction = data[key]
            break

    if true_vorticity is None or true_streamfunction is None:
        raise KeyError(
            f"Could not find ground truth data in npz file.\n"
            f"Looked for vorticity keys: {truth_keys_vort}\n"
            f"Looked for streamfunction keys: {truth_keys_psi}\n"
            f"Available keys: {list(data.keys())}"
        )

    return pred_vorticity, pred_streamfunction, true_vorticity, true_streamfunction


def compute_global_clims(pred_vorticity, pred_streamfunction,
                         true_vorticity, true_streamfunction,
                         vorticity_clim=None, streamfunction_clim=None):
    """
    Compute global symmetric color limits for consistent visualization.

    Args:
        pred_vorticity (ndarray): Prediction vorticity (T, Nx, Ny)
        pred_streamfunction (ndarray): Prediction streamfunction (T, Nx, Ny)
        true_vorticity (ndarray): Ground truth vorticity (T, Nx, Ny)
        true_streamfunction (ndarray): Ground truth streamfunction (T, Nx, Ny)
        vorticity_clim (float or None): Manual vorticity limit
        streamfunction_clim (float or None): Manual streamfunction limit

    Returns:
        dict: Color limits {"vorticity": (vmin, vmax), "streamfunction": (vmin, vmax)}
    """
    clims = {}

    # Vorticity color limits
    if vorticity_clim is not None:
        clims["vorticity"] = (-vorticity_clim, vorticity_clim)
    else:
        # Compute from both datasets
        vmax_pred = np.nanmax(np.abs(pred_vorticity[np.isfinite(pred_vorticity)]))
        vmax_truth = np.nanmax(np.abs(true_vorticity[np.isfinite(true_vorticity)]))
        vmax = max(vmax_pred, vmax_truth)
        clims["vorticity"] = (-vmax, vmax)

    # Streamfunction color limits
    if streamfunction_clim is not None:
        clims["streamfunction"] = (-streamfunction_clim, streamfunction_clim)
    else:
        # Compute from both datasets
        pmax_pred = np.nanmax(np.abs(pred_streamfunction[np.isfinite(pred_streamfunction)]))
        pmax_truth = np.nanmax(np.abs(true_streamfunction[np.isfinite(true_streamfunction)]))
        pmax = max(pmax_pred, pmax_truth)
        clims["streamfunction"] = (-pmax, pmax)

    return clims


def plot_snapshot_comparison(pred_vort, pred_psi, true_vort, true_psi,
                             time, clims, cmap, outdir, dpi, snapshot_idx):
    """
    Create 2x2 comparison plot for a single time snapshot.

    Layout:
        [Ground Truth Vorticity] [Ground Truth Streamfunction]
        [Prediction Vorticity]   [Prediction Streamfunction]

    Args:
        pred_vort (ndarray): Prediction vorticity (Nx, Ny)
        pred_psi (ndarray): Prediction streamfunction (Nx, Ny)
        true_vort (ndarray): Ground truth vorticity (Nx, Ny)
        true_psi (ndarray): Ground truth streamfunction (Nx, Ny)
        time (float): Simulation time
        clims (dict): Color limits for each field
        cmap (str): Colormap name
        outdir (Path): Output directory
        dpi (int): Figure DPI
        snapshot_idx (int): Snapshot index for filename
    """
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(12, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                          hspace=0.35, wspace=0.35,
                          left=0.08, right=0.92, top=0.93, bottom=0.05)

    # Titles for each subplot
    titles = [
        "Ground Truth: Vorticity",
        "Ground Truth: Streamfunction",
        "Prediction: Vorticity",
        "Prediction: Streamfunction"
    ]

    # Data for each subplot
    fields = [true_vort, true_psi, pred_vort, pred_psi]
    field_names = ["vorticity", "streamfunction", "vorticity", "streamfunction"]

    # Plot each field
    for idx, (field, field_name, title) in enumerate(zip(fields, field_names, titles)):
        row, col = divmod(idx, 2)
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
    pred_vorticity, pred_streamfunction, true_vorticity, true_streamfunction = \
        load_data_from_npz(pred_path)

    print(f"Prediction shape: {pred_vorticity.shape}")
    print(f"Ground truth shape: {true_vorticity.shape}")

    T = pred_vorticity.shape[0]

    # Create time array
    times = args.t_start + np.arange(T) * args.dt

    # Compute global color limits
    print("\nComputing global color limits...")
    clims = compute_global_clims(
        pred_vorticity, pred_streamfunction,
        true_vorticity, true_streamfunction,
        args.vorticity_clim, args.streamfunction_clim
    )
    print(f"  Vorticity range: [{clims['vorticity'][0]:.3e}, {clims['vorticity'][1]:.3e}]")
    print(f"  Streamfunction range: [{clims['streamfunction'][0]:.3e}, {clims['streamfunction'][1]:.3e}]")

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
            pred_vort=pred_vorticity[snap_idx],
            pred_psi=pred_streamfunction[snap_idx],
            true_vort=true_vorticity[snap_idx],
            true_psi=true_streamfunction[snap_idx],
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
