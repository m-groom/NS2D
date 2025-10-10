#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot NS2D simulation output: scalars, spectra, fluxes, and snapshots.

This script provides a simple interface to visualize all output from a single
NS2D realization. It automatically detects available data and generates plots.

Usage:
    python plot_output.py --run_dir path/to/realisation_0000 --out ./figures

    python plot_output.py --run_dir snapshots/Nx1024_Ny1024_nu5e-05/realisation_0000 \\
                          --out ./my_plots --dpi 150 --no_snapshots

For help:
    python plot_output.py --help
"""

import argparse
import pathlib
import re
import sys
import numpy as np

# Add parent directory to path to import post-processing module
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from post import io, visualization, analysis


def get_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Plot scalars, spectra, fluxes, and snapshots for a single NS2D realization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    ap.add_argument("--run_dir", type=str, required=True,
                   help="Path to realization directory containing 'scalars', 'snapshots', and 'spectra.h5'")

    # Output
    ap.add_argument("--out", type=str, default="./figures",
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
    ap.add_argument("--snap_count", type=int, default=100,
                   help="Number of snapshot writes to plot")
    ap.add_argument("--snap_stride", type=int, default=1,
                   help="Stride between snapshot writes")

    # Spectra/flux options
    ap.add_argument("--spectra_max_curves", type=int, default=6,
                   help="Maximum number of time curves to overlay")
    ap.add_argument("--spectra_loglog", action="store_true",
                   help="Use log-log axes for spectra plots")

    return ap.parse_args()


def infer_nyquist(run_dir, Lx):
    """
    Infer Nyquist wavenumber from directory name.

    Args:
        run_dir (Path): Run directory path
        Lx (float): Domain length

    Returns:
        float or None: Nyquist wavenumber
    """
    run_str = str(run_dir)

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

    run_dir = pathlib.Path(args.run_dir).resolve()
    out_root = pathlib.Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NS2D Output Plotting")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {out_root}")
    print(f"DPI: {args.dpi}")
    print("=" * 70)

    # Infer Nyquist wavenumber
    k_nyquist = infer_nyquist(run_dir, args.Lx)
    if k_nyquist:
        print(f"Inferred Nyquist wavenumber: {k_nyquist:.2f}")
    else:
        print("Could not infer Nyquist from directory name")

    # 1) Scalar time series
    if not args.no_scalars:
        print("\n[1/4] Plotting scalar time series...")
        scalars_dir = run_dir / "scalars"

        if not scalars_dir.exists():
            print(f"  Warning: No scalars directory found at {scalars_dir}")
        else:
            try:
                times, series_dict = io.read_scalars(scalars_dir)
                print(f"  Loaded {len(times)} time points")
                print(f"  Available scalars: {', '.join(series_dict.keys())}")

                visualization.plot_time_series(
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
        spectra_path = run_dir / "spectra.h5"

        if not spectra_path.exists():
            print(f"  Warning: No spectra file found at {spectra_path}")
        else:
            try:
                times, kbins, Ek_list, Zk_list = io.read_spectra(spectra_path)
                print(f"  Loaded {len(times)} spectra snapshots")
                print(f"  Wavenumber range: [{kbins[1]:.2f}, {kbins[-1]:.2f}]")

                visualization.plot_spectra(
                    times, kbins, Ek_list, Zk_list,
                    outdir=out_root / "spectra",
                    max_curves=args.spectra_max_curves,
                    loglog=args.spectra_loglog,
                    k_nyquist=k_nyquist,
                    dpi=args.dpi
                )
                print(f"  Saved to {out_root / 'spectra'}")

            except Exception as e:
                print(f"  Error plotting spectra: {e}")

    # 3) Fluxes
    if not args.no_flux:
        print("\n[3/4] Plotting fluxes...")
        spectra_path = run_dir / "spectra.h5"

        if not spectra_path.exists():
            print(f"  Warning: No spectra file found at {spectra_path}")
        else:
            # Energy flux
            try:
                times_f, kbins_f, T_list, Pi_list = io.read_flux(spectra_path, flux_type="energy")
                print(f"  Loaded {len(times_f)} energy flux snapshots")

                visualization.plot_flux(
                    times_f, kbins_f, T_list, Pi_list,
                    outdir=out_root / "flux",
                    flux_type="energy",
                    max_curves=args.spectra_max_curves,
                    k_nyquist=k_nyquist,
                    dpi=args.dpi
                )
                print(f"  Saved energy flux to {out_root / 'flux'}")

            except Exception as e:
                print(f"  Error plotting energy flux: {e}")

            # Enstrophy flux
            try:
                times_fz, kbins_fz, TZ_list, PiZ_list = io.read_flux(spectra_path, flux_type="enstrophy")
                print(f"  Loaded {len(times_fz)} enstrophy flux snapshots")

                visualization.plot_flux(
                    times_fz, kbins_fz, TZ_list, PiZ_list,
                    outdir=out_root / "enstrophy_flux",
                    flux_type="enstrophy",
                    max_curves=args.spectra_max_curves,
                    k_nyquist=k_nyquist,
                    dpi=args.dpi
                )
                print(f"  Saved enstrophy flux to {out_root / 'enstrophy_flux'}")

            except Exception as e:
                print(f"  Error plotting enstrophy flux: {e}")

    # 4) Snapshots
    if not args.no_snapshots:
        print("\n[4/4] Plotting snapshot frames...")
        snaps_dir = run_dir / "snapshots"

        if not snaps_dir.exists():
            print(f"  Warning: No snapshots directory found at {snaps_dir}")
        else:
            snap_files = sorted(snaps_dir.glob("*.h5"))
            if not snap_files:
                print(f"  Warning: No HDF5 files found in {snaps_dir}")
            else:
                print(f"  Found {len(snap_files)} snapshot file(s)")

                for snap_h5 in snap_files:
                    try:
                        info = io.get_snapshot_info(snap_h5)
                        print(f"  Processing {snap_h5.name}: "
                              f"{info['n_writes']} writes, "
                              f"tasks: {', '.join(info['tasks'])}")

                        subdir = out_root / "snapshots" / snap_h5.stem
                        subdir.mkdir(parents=True, exist_ok=True)

                        # Plot selected snapshots
                        for idx in range(args.snap_start,
                                       min(args.snap_start + args.snap_count, info['n_writes']),
                                       max(1, args.snap_stride)):
                            visualization.plot_snapshot(
                                snap_h5,
                                write_index=idx,
                                tasks=["vorticity", "pressure", "streamfunction"],
                                outdir=subdir,
                                dpi=args.dpi
                            )

                        print(f"    Saved to {subdir}")

                    except Exception as e:
                        print(f"  Error plotting snapshots from {snap_h5.name}: {e}")

    print("\n" + "=" * 70)
    print("Plotting complete!")
    print(f"All figures saved to: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
