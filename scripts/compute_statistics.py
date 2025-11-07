#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute statistics and derived quantities from NS2D simulation output.

This script analyses scalar time series and spectra to compute:
- Time-averaged statistics
- Spectral slopes and scaling exponents
- Integral and Taylor microscales
- Reynolds numbers (Taylor and integral scale)

Usage:
    python compute_statistics.py --rundir path/to/realisation_0000 --nu 5e-5

    python compute_statistics.py --rundir snapshots/Nx1024_Ny1024_nu5e-05/realisation_0000 \\
                                 --t_start 50 --t_end 200 --nu 5e-5 --k_range 20 100

For help:
    python compute_statistics.py --help
"""

import argparse
import pathlib
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from post import io, analysis


def get_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Compute statistics from NS2D simulation output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--rundir", type=str, required=True,
                   help="Path to realisation directory")

    ap.add_argument("--t_start", type=float, default=None,
                   help="Start time for statistics (default: first time)")
    ap.add_argument("--t_end", type=float, default=None,
                   help="End time for statistics (default: last time)")

    # Spectral analysis
    ap.add_argument("--k_range", nargs=2, type=float, default=None,
                   help="Wavenumber range [k_min k_max] for spectral slope fitting")

    # Reynolds numbers
    ap.add_argument("--nu", type=float, default=None,
                   help="Kinematic viscosity (required for Reynolds number computation)")

    ap.add_argument("--output", type=str, default=None,
                   help="Output file for statistics (default: print to stdout)")

    return ap.parse_args()


def main():
    """Main execution function."""
    args = get_args()

    rundir = pathlib.Path(args.rundir).resolve()

    print("=" * 70)
    print("NS2D Statistics Analysis")
    print("=" * 70)
    print(f"Run directory: {rundir}")

    # Load scalar time series
    scalars_dir = rundir / "scalars"
    if not scalars_dir.exists():
        print(f"Error: No scalars directory found at {scalars_dir}")
        return

    times, series_dict = io.read_scalars(scalars_dir)

    # Determine time range
    t_start = times[0] if args.t_start is None else args.t_start
    t_end = times[-1] if args.t_end is None else args.t_end

    print(f"\nTime range for statistics: [{t_start:.2f}, {t_end:.2f}]")
    print(f"Total simulation time: [{times[0]:.2f}, {times[-1]:.2f}]")

    # Compute statistics
    print("\n" + "=" * 70)
    print("Scalar Statistics")
    print("=" * 70)

    stats = analysis.compute_statistics_summary(times, series_dict, t_start, t_end)

    for name, stat in stats.items():
        print(f"\n{name}:")
        print(f"  Mean:   {stat['mean']:12.5e}")
        print(f"  Std:    {stat['std']:12.5e}")
        print(f"  Median: {stat['median']:12.5e}")
        print(f"  Min:    {stat['min']:12.5e}")
        print(f"  Max:    {stat['max']:12.5e}")

    # Derived quantities
    if "energy" in series_dict and "enstrophy" in series_dict:
        print("\n" + "=" * 70)
        print("Derived Quantities")
        print("=" * 70)

        E_mean = stats["energy"]["mean"]
        Z_mean = stats["enstrophy"]["mean"]

        # Taylor microscale
        lambda_T = analysis.compute_taylor_microscale(Z_mean, E_mean)
        print(f"\nTaylor microscale λ: {lambda_T:.5e}")

        # Reynolds numbers (if viscosity provided)
        if args.nu is not None:
            Re_lambda = analysis.compute_taylor_reynolds(E_mean, Z_mean, args.nu)
            print(f"Taylor Reynolds number Re_λ: {Re_lambda:.2f}")

        # Energy balance (if all terms available)
        if all(k in series_dict for k in ["inj", "drag_loss", "visc_loss"]):
            mask = (times >= t_start) & (times <= t_end)

            inj_mean = np.mean(series_dict["inj"][mask])
            drag_mean = np.mean(series_dict["drag_loss"][mask])
            visc_mean = np.mean(series_dict["visc_loss"][mask])
            residual = inj_mean - drag_mean - visc_mean

            print("\nEnergy Balance:")
            print(f"  Injection <u·f>:     {inj_mean:12.5e}")
            print(f"  Drag dissipation:    {drag_mean:12.5e}")
            print(f"  Viscous dissipation: {visc_mean:12.5e}")
            print(f"  Residual:            {residual:12.5e}")
            print(f"  Relative error:      {abs(residual/inj_mean)*100:.2f}%")

    # Spectral analysis
    spectra_path = rundir / "spectra.h5"
    if spectra_path.exists():
        print("\n" + "=" * 70)
        print("Spectral Analysis")
        print("=" * 70)

        try:
            times_spec, kbins, Ek_list, Zk_list = io.read_spectra(spectra_path)

            # Convert lists to arrays for indexing
            Ek_array = np.array(Ek_list)
            Zk_array = np.array(Zk_list)

            # Time-average spectra over [t_start, t_end]
            mask_spec = (times_spec >= t_start) & (times_spec <= t_end)
            if not np.any(mask_spec):
                print(f"Warning: No spectra in time range [{t_start}, {t_end}]")
                print(f"Using last spectrum at t = {times_spec[-1]:.2f}")
                Ek_mean = Ek_array[-1]
                Zk_mean = Zk_array[-1]
                t_range_str = f"t = {times_spec[-1]:.2f}"
            else:
                Ek_mean = np.mean(Ek_array[mask_spec], axis=0)
                Zk_mean = np.mean(Zk_array[mask_spec], axis=0)
                n_samples = np.sum(mask_spec)
                t_range_str = f"t ∈ [{t_start:.2f}, {t_end:.2f}] ({n_samples} snapshots)"

            print(f"\nAnalysing time-averaged spectrum over {t_range_str}")

            # Integral length scale
            L_int = analysis.compute_integral_scale(kbins, Ek_mean)
            print(f"Integral length scale L_int: {L_int:.5e}")

            # RMS velocity over [t_start, t_end]
            u_rms = None
            if "u_rms" in stats:
                u_rms = stats["u_rms"]["mean"]
            elif "energy" in stats:
                # E = 0.5 * <u^2 + v^2> => u_rms = sqrt(2 * E)
                u_rms = np.sqrt(2.0 * stats["energy"]["mean"])

            if u_rms is not None:
                print(f"RMS velocity u_rms: {u_rms:.5e}")

            # Geometric-mean length scale sqrt(L_int * L_taylor)
            if "energy" in stats and "enstrophy" in stats:
                lambda_t = analysis.compute_taylor_microscale(
                    stats["enstrophy"]["mean"], stats["energy"]["mean"]
                )
                L_geo = np.sqrt(L_int * lambda_t)
                print(f"Geometric-mean length scale L_geo: {L_geo:.5e}")

            # Integral Reynolds number (if viscosity and energy available)
            if args.nu is not None and "energy" in stats:
                Re_L = analysis.compute_integral_reynolds(L_int, stats["energy"]["mean"], args.nu)
                print(f"Integral Reynolds number Re_L: {Re_L:.2f}")

            # Spectral slope (if k_range specified)
            if args.k_range:
                k_min, k_max = args.k_range
                print(f"\nFitting spectral slope in range [{k_min}, {k_max}]:")

                try:
                    slope_E = analysis.compute_spectral_slope(kbins, Ek_mean, (k_min, k_max))
                    print(f"\nEnergy spectrum E(k):")
                    print(f"  Slope:      {slope_E['slope']:.3f}")
                    print(f"  R²:         {slope_E['r_squared']:.4f}")

                    slope_Z = analysis.compute_spectral_slope(kbins, Zk_mean, (k_min, k_max))
                    print(f"\nEnstrophy spectrum Z(k):")
                    print(f"  Slope:      {slope_Z['slope']:.3f}")
                    print(f"  R²:         {slope_Z['r_squared']:.4f}")

                except ValueError as e:
                    print(f"  Error: {e}")

        except Exception as e:
            print(f"Error in spectral analysis: {e}")

    # Save to file if requested
    if args.output:
        output_path = pathlib.Path(args.output)
        with open(output_path, "w") as f:
            f.write("# NS2D Statistics Summary\n")
            f.write(f"# Run directory: {rundir}\n")
            f.write(f"# Time range: [{t_start}, {t_end}]\n\n")

            f.write("# Scalar Statistics\n")
            for name, stat in stats.items():
                f.write(f"\n{name}:\n")
                for key, val in stat.items():
                    f.write(f"  {key}: {val:.5e}\n")

        print(f"\nStatistics saved to {output_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
