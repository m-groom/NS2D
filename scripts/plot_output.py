#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot NS2D simulation output: scalars, spectra, fluxes, and snapshots.

This script provides a simple interface to visualise all output from a single
NS2D realisation. It automatically detects available data and generates plots.

Usage:
    python plot_output.py --rundir path/to/realisation_0000 --outdir ./figures

    python plot_output.py --rundir snapshots/Nx1024_Ny1024_nu5e-05/realisation_0000 \\
                          --outdir ./my_plots --dpi 150 --no_snapshots

For help:
    python plot_output.py --help
"""

import argparse
import pathlib
import re
import sys
import numpy as np
import h5py

# Add parent directory to path to import post-processing module
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from post import io, visualisation, analysis


def get_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Plot scalars, spectra, fluxes, and snapshots for a single NS2D realisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    ap.add_argument("--rundir", type=str, required=True,
                   help="Path to realisation directory containing 'scalars', 'snapshots', and 'spectra.h5'")

    # Output
    ap.add_argument("--outdir", type=str, default="./figures",
                   help="Output directory for generated figures")
    ap.add_argument("--dpi", type=int, default=300,
                   help="Figure DPI (resolution)")

    # Domain parameters (for Nyquist calculation)
    ap.add_argument("--Lx", type=float, default=2*np.pi,
                   help="Domain length in x (for Nyquist calculation)")
    ap.add_argument("--Ly", type=float, default=2*np.pi,
                   help="Domain length in y (for Nyquist calculation)")

    # What to plot
    ap.add_argument("--no_scalars", action="store_true",
                   help="Skip scalar time-series plots")
    ap.add_argument("--no_spectra", action="store_true",
                   help="Skip spectra plots")
    ap.add_argument("--no_flux", action="store_true",
                   help="Skip flux plots")
    ap.add_argument("--no_snapshots", action="store_true",
                   help="Skip snapshot frame generation")

    # Snapshot selection
    ap.add_argument("--snap_start", type=int, default=0,
                   help="Starting write index for snapshots")
    ap.add_argument("--snap_count", type=int, default=0,
                   help="Number of snapshot writes to plot (0 = all)")
    ap.add_argument("--snap_stride", type=int, default=1,
                   help="Stride between snapshot writes")

    # Spectra/flux options
    ap.add_argument("--spectra_max_curves", type=int, default=6,
                   help="Maximum number of time curves to overlay")
    ap.add_argument("--spectra_loglog", action="store_true",
                   help="Use log-log axes for spectra plots")
    ap.add_argument("--spectra_tmin", type=float, default=None,
                   help="Minimum simulation time for selecting spectra/flux curves")
    ap.add_argument("--spectra_tmax", type=float, default=None,
                   help="Maximum simulation time for selecting spectra/flux curves")

    return ap.parse_args()


def infer_nyquist(rundir, Lx):
    """
    Infer Nyquist wavenumber from directory name.

    Args:
        rundir (Path): Run directory path
        Lx (float): Domain length

    Returns:
        float or None: Nyquist wavenumber
    """
    run_str = str(rundir)

    # Try to extract Nx and Ny from path
    m_nx = re.search(r"Nx(\d+)", run_str)
    m_ny = re.search(r"Ny(\d+)", run_str)

    if m_nx and m_ny:
        Nx = int(m_nx.group(1))
        Ny = int(m_ny.group(1))
        # Nyquist: k_N = π·N/L
        k_nyquist = (np.pi * min(Nx, Ny)) / max(Lx, 1e-12)
        return k_nyquist

    return None


def main():
    """Main execution function."""
    args = get_args()

    rundir = pathlib.Path(args.rundir).resolve()
    out_root = pathlib.Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NS2D Output Plotting")
    print("=" * 70)
    print(f"Run directory: {rundir}")
    print(f"Output directory: {out_root}")
    print(f"DPI: {args.dpi}")
    print("=" * 70)

    # Infer Nyquist wavenumber
    k_nyquist = infer_nyquist(rundir, args.Lx)
    if k_nyquist:
        print(f"Inferred Nyquist wavenumber: {k_nyquist:.2f}")
    else:
        print("Could not infer Nyquist from directory name")

    # 1) Scalar time series
    if not args.no_scalars:
        print("\n[1/4] Plotting scalar time series...")
        scalars_dir = rundir / "scalars"

        if not scalars_dir.exists():
            print(f"  Warning: No scalars directory found at {scalars_dir}")
        else:
            try:
                times, series_dict = io.read_scalars(scalars_dir)
                print(f"  Loaded {len(times)} time points")
                print(f"  Available scalars: {', '.join(series_dict.keys())}")

                visualisation.plot_time_series(
                    times, series_dict,
                    outdir=out_root / "scalars",
                    dpi=args.dpi
                )
                print(f"  Saved to {out_root / 'scalars'}")

                # Compute and print statistics
                stats = analysis.compute_statistics_summary(times, series_dict)
                print("\n  Statistics (full time range):")
                for name, stat in stats.items():
                    print(f"    {name:15s}: mean={stat['mean']:10.3e}, "
                          f"std={stat['std']:10.3e}")

            except Exception as e:
                print(f"  Error plotting scalars: {e}")

    # 2) Spectra
    if not args.no_spectra:
        print("\n[2/4] Plotting spectra...")
        spectra_path = rundir / "spectra.h5"

        if not spectra_path.exists():
            print(f"  Warning: No spectra file found at {spectra_path}")
        else:
            try:
                times, kbins, Ek_list, Zk_list = io.read_spectra(spectra_path)
                print(f"  Loaded {len(times)} spectra snapshots")
                print(f"  Wavenumber range: [{kbins[1]:.2f}, {kbins[-1]:.2f}]")

                visualisation.plot_spectra(
                    times, kbins, Ek_list, Zk_list,
                    outdir=out_root / "spectra",
                    max_curves=args.spectra_max_curves,
                    loglog=args.spectra_loglog,
                    k_nyquist=k_nyquist,
                    dpi=args.dpi,
                    tmin=args.spectra_tmin,
                    tmax=args.spectra_tmax
                )
                print(f"  Saved to {out_root / 'spectra'}")

            except Exception as e:
                print(f"  Error plotting spectra: {e}")

    # 3) Fluxes
    if not args.no_flux:
        print("\n[3/4] Plotting fluxes...")
        spectra_path = rundir / "spectra.h5"

        if not spectra_path.exists():
            print(f"  Warning: No spectra file found at {spectra_path}")
        else:
            # Energy flux
            try:
                times_f, kbins_f, T_list, Pi_list = io.read_flux(spectra_path, flux_type="energy")
                print(f"  Loaded {len(times_f)} energy flux snapshots")

                visualisation.plot_flux(
                    times_f, kbins_f, T_list, Pi_list,
                    outdir=out_root / "flux",
                    flux_type="energy",
                    max_curves=args.spectra_max_curves,
                    k_nyquist=k_nyquist,
                    dpi=args.dpi,
                    tmin=args.spectra_tmin,
                    tmax=args.spectra_tmax
                )
                print(f"  Saved energy flux to {out_root / 'flux'}")

            except Exception as e:
                print(f"  Error plotting energy flux: {e}")

            # Enstrophy flux
            try:
                times_fz, kbins_fz, TZ_list, PiZ_list = io.read_flux(spectra_path, flux_type="enstrophy")
                print(f"  Loaded {len(times_fz)} enstrophy flux snapshots")

                visualisation.plot_flux(
                    times_fz, kbins_fz, TZ_list, PiZ_list,
                    outdir=out_root / "flux",
                    flux_type="enstrophy",
                    max_curves=args.spectra_max_curves,
                    k_nyquist=k_nyquist,
                    dpi=args.dpi,
                    tmin=args.spectra_tmin,
                    tmax=args.spectra_tmax
                )
                print(f"  Saved enstrophy flux to {out_root / 'flux'}")

            except Exception as e:
                print(f"  Error plotting enstrophy flux: {e}")

    # 4) Snapshots
    if not args.no_snapshots:
        print("\n[4/4] Plotting snapshot frames...")
        snaps_dir = rundir / "snapshots"

        if not snaps_dir.exists():
            print(f"  Warning: No snapshots directory found at {snaps_dir}")
        else:
            snap_files = sorted(snaps_dir.glob("*.h5"))
            if not snap_files:
                print(f"  Warning: No HDF5 files found in {snaps_dir}")
            else:
                print(f"  Found {len(snap_files)} snapshot file(s)")

                # Tasks to plot and for which to compute global color limits
                plot_tasks = ["vorticity", "pressure", "streamfunction"]

                # Compute global color limits across ALL files for consistency.
                # For vorticity and pressure, use robust symmetric limits based on a
                # high percentile of |field| to reduce the influence of outliers.
                global_fixed_clims = {}
                try:
                    stats = {}  # tname -> dict(data_min, data_max, robust_abs)
                    for snap_h5 in snap_files:
                        with h5py.File(snap_h5, 'r') as f:
                            for tname in plot_tasks:
                                ds_key = f"tasks/{tname}"
                                if ds_key not in f:
                                    continue
                                arr = np.asarray(f[ds_key][:])
                                finite_mask = np.isfinite(arr)
                                if not np.any(finite_mask):
                                    continue

                                arr_f = arr[finite_mask]
                                dmin = float(np.nanmin(arr_f))
                                dmax = float(np.nanmax(arr_f))

                                # High-percentile absolute value for robust symmetric limits
                                try:
                                    q = float(np.nanpercentile(np.abs(arr_f), 99.0))
                                except Exception:
                                    q = np.nan

                                s = stats.setdefault(
                                    tname,
                                    {"data_min": np.inf, "data_max": -np.inf, "robust_abs": 0.0},
                                )
                                s["data_min"] = min(s["data_min"], dmin)
                                s["data_max"] = max(s["data_max"], dmax)
                                if np.isfinite(q):
                                    s["robust_abs"] = max(s["robust_abs"], q)

                    for tname in plot_tasks:
                        s = stats.get(tname)
                        if s is None:
                            # Nothing accumulated for this field
                            global_fixed_clims[tname] = (-0.0, 0.0)
                            continue

                        if tname in ("vorticity", "pressure"):
                            # Robust symmetric limits
                            a = s["robust_abs"]
                            if not np.isfinite(a) or a <= 0.0:
                                a = max(abs(s["data_min"]), abs(s["data_max"]))
                            if not np.isfinite(a) or a <= 0.0:
                                a = 1.0
                            global_fixed_clims[tname] = (-a, a)
                        else:
                            # Streamfunction (or others): use actual min/max
                            dmin = s["data_min"]
                            dmax = s["data_max"]
                            if (not np.isfinite(dmin) or not np.isfinite(dmax)
                                    or dmin == dmax):
                                global_fixed_clims[tname] = (-0.0, 0.0)
                            else:
                                global_fixed_clims[tname] = (dmin, dmax)

                except Exception as e:
                    print(f"    Warning: could not compute global clims: {e}")
                    global_fixed_clims = {}

                for snap_h5 in snap_files:
                    try:
                        info = io.get_snapshot_info(snap_h5)
                        print(f"  Processing {snap_h5.name}: "
                              f"{info['n_writes']} writes, "
                              f"tasks: {', '.join(info['tasks'])}")

                        subdir = out_root / "snapshots" / snap_h5.stem
                        subdir.mkdir(parents=True, exist_ok=True)

                        # Plot selected snapshots with progress output
                        # Apply stride across the full available writes, then limit by count.
                        stride = max(1, args.snap_stride)
                        # Clamp start within available range
                        start = min(max(0, args.snap_start), max(0, info['n_writes'] - 1))
                        candidates = list(range(start, info['n_writes'], stride))
                        if args.snap_count is not None and args.snap_count > 0:
                            idx_list = candidates[:args.snap_count]
                        else:
                            idx_list = candidates
                        total = len(idx_list)
                        bar_width = 30

                        for pos, idx in enumerate(idx_list, start=1):
                            visualisation.plot_snapshot(
                                snap_h5,
                                write_index=idx,
                                tasks=plot_tasks,
                                outdir=subdir,
                                dpi=args.dpi,
                                clims=global_fixed_clims
                            )

                            # Simple terminal progress bar
                            filled = int(bar_width * pos / max(1, total))
                            bar = "#" * filled + "-" * (bar_width - filled)
                            percent = int(100 * pos / max(1, total))
                            print(f"    [{bar}] {pos}/{total} ({percent}%)", end='\r', flush=True)

                        if total > 0:
                            print()  # newline after finishing bar

                        print(f"    Saved to {subdir}")

                    except Exception as e:
                        print(f"  Error plotting snapshots from {snap_h5.name}: {e}")

    print("\n" + "=" * 70)
    print("Plotting complete!")
    print(f"All figures saved to: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
