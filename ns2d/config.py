"""
Configuration and command-line argument parsing for NS2D simulations.

This module handles all command-line arguments and parameter validation
for the 2D Navier-Stokes solver.
"""

import argparse
import numpy as np


def get_args():
    """
    Parse command-line arguments for NS2D simulation.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing all
            simulation parameters (domain size, physics parameters, forcing
            options, output settings, etc.)
    """
    ap = argparse.ArgumentParser(
        description="Forced 2D incompressible Navier-Stokes (constant-power stochastic forcing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Domain / resolution parameters
    domain_group = ap.add_argument_group('Domain and Resolution')
    domain_group.add_argument(
        "--Nx", type=int, default=256,
        help="Number of Fourier modes in x direction"
    )
    domain_group.add_argument(
        "--Ny", type=int, default=256,
        help="Number of Fourier modes in y direction"
    )
    domain_group.add_argument(
        "--Lx", type=float, default=2.0 * np.pi,
        help="Domain length in x direction"
    )
    domain_group.add_argument(
        "--Ly", type=float, default=2.0 * np.pi,
        help="Domain length in y direction"
    )
    domain_group.add_argument(
        "--dealias", type=float, default=1.5,
        help="Dealias factor"
    )

    # MPI process mesh
    mpi_group = ap.add_argument_group('MPI Configuration')
    mpi_group.add_argument(
        "--procs_x", type=int, default=0,
        help="Number of processes along x direction (0 = auto/1D mesh)"
    )
    mpi_group.add_argument(
        "--procs_y", type=int, default=0,
        help="Number of processes along y direction (0 = auto/1D mesh)"
    )

    # Physics parameters
    physics_group = ap.add_argument_group('Physical Parameters')
    physics_group.add_argument(
        "--nu", type=float, default=5e-4,
        help="Kinematic viscosity"
    )
    physics_group.add_argument(
        "--alpha", type=float, default=0.025,
        help="Linear (Ekman) friction coefficient for large-scale damping (0 disables)"
    )

    # Forcing options
    forcing_group = ap.add_argument_group('Forcing Configuration')
    forcing_group.add_argument(
        "--forcing", type=str, default="stochastic",
        choices=["stochastic", "kolmogorov", "none"],
        help="Forcing type: stochastic (band-limited), deterministic Kolmogorov, or none"
    )
    forcing_group.add_argument(
        "--stoch_type", type=str, default="ou",
        choices=["white", "ou"],
        help="Stochastic forcing type: white (δ-correlated) or ou (Ornstein-Uhlenbeck)"
    )
    forcing_group.add_argument(
        "--kmin", type=float, default=8.0,
        help="Minimum forced wavenumber (physical units: rad/length)"
    )
    forcing_group.add_argument(
        "--kmax", type=float, default=12.0,
        help="Maximum forced wavenumber (physical units: rad/length)"
    )
    forcing_group.add_argument(
        "--f_sigma", type=float, default=0.02,
        help="Base spectral forcing amplitude (used directly if power_mode='sigma')"
    )
    forcing_group.add_argument(
        "--tau_ou", type=float, default=0.3,
        help="Correlation time for OU forcing (ignored for white noise)"
    )

    forcing_group.add_argument(
        "--kolmogorov_f0", type=float, default=0.1,
        help="Amplitude of deterministic Kolmogorov forcing"
    )
    forcing_group.add_argument(
        "--k_drive", type=float, default=4.0,
        help="Driving wavenumber for Kolmogorov forcing (physical units: rad/length)"
    )
    forcing_group.add_argument(
        "--k_phase", type=float, default=0.0,
        help="Phase offset for Kolmogorov forcing (radians)"
    )

    # Constant-power forcing controls
    const_power_group = ap.add_argument_group('Constant-Power Forcing')
    const_power_group.add_argument(
        "--eps_target", type=float, default=1.0e-3,
        help="Target domain-averaged energy injection rate ε = <u·f>"
    )
    const_power_group.add_argument(
        "--power_mode", type=str, default="constant",
        choices=["constant", "sigma"],
        help="Forcing power handling: 'constant' rescales to hit eps_target; 'sigma' uses f_sigma directly"
    )
    const_power_group.add_argument(
        "--eps_floor", type=float, default=1.0e-12,
        help="Floor value to prevent division by tiny <u·f> during rescaling"
    )
    const_power_group.add_argument(
        "--eps_clip", type=float, default=10.0,
        help="Maximum allowed per-step rescale factor (prevents extreme jumps)"
    )
    const_power_group.add_argument(
        "--eps_smooth", type=float, default=0.0,
        help="Exponential smoothing factor for scale (0 disables)"
    )

    # Time integration
    time_group = ap.add_argument_group('Time Integration')
    time_group.add_argument(
        "--t_end", type=float, default=200.0,
        help="Total simulation time"
    )
    time_group.add_argument(
        "--cfl_safety", type=float, default=0.4,
        help="CFL safety factor"
    )
    time_group.add_argument(
        "--cfl_threshold", type=float, default=0.1,
        help="CFL threshold to prevent tiny timestep adjustments"
    )
    time_group.add_argument(
        "--cfl_cadence", type=int, default=10,
        help="Number of iterations between CFL updates"
    )
    time_group.add_argument(
        "--cfl_max_dt", type=float, default=1e-1,
        help="Maximum allowed timestep"
    )
    time_group.add_argument(
        "--cfl_min_dt", type=float, default=1e-8,
        help="Minimum allowed timestep"
    )

    # Output cadences
    output_group = ap.add_argument_group('Output Settings')
    output_group.add_argument(
        "--snap_dt", type=float, default=1.0,
        help="Snapshot output interval (velocity, pressure, etc.)"
    )
    output_group.add_argument(
        "--spectra_dt", type=float, default=0.25,
        help="Spectra and flux output interval"
    )
    output_group.add_argument(
        "--scalars_dt", type=float, default=0.05,
        help="Scalar time-series output interval (energy, enstrophy, etc.)"
    )

    # Ensemble and reproducibility
    ensemble_group = ap.add_argument_group('Ensemble Configuration')
    ensemble_group.add_argument(
        "--n_realisations", type=int, default=1,
        help="Number of independent realisations to run"
    )
    ensemble_group.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for initial conditions (each realisation uses seed+r)"
    )

    # Initial conditions
    ic_group = ap.add_argument_group('Initial Conditions')
    ic_group.add_argument(
        "--ic_alpha", type=float, default=49.0,
        help="Spectral roll-off parameter for initial vorticity"
    )
    ic_group.add_argument(
        "--ic_power", type=float, default=2.5,
        help="Power-law exponent for initial vorticity spectrum"
    )
    ic_group.add_argument(
        "--ic_scale", type=float, default=7.0 ** 1.5,
        help="Overall amplitude scaling for initial vorticity"
    )
    ic_group.add_argument(
        "--ic_energy", type=float, default=None,
        help="Optional target domain-averaged kinetic energy (0.5<|u|^2>) for initial conditions"
    )
    ic_group.add_argument(
        "--ic_seed", type=int, default=None,
        help="Optional base seed for initial conditions (defaults to --seed if not set)",
    )

    # Output directories and precision
    misc_group = ap.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        "--outdir", type=str, default="snapshots",
        help="Root output directory for simulation data"
    )
    misc_group.add_argument(
        "--tag", type=str, default="",
        help="Optional tag to add to output directory name"
    )
    misc_group.add_argument(
        "--precision", type=str, default="float64",
        choices=["float64", "float32"],
        help="Floating-point precision for simulation"
    )

    return ap.parse_args()


def validate_args(args):
    """
    Validate command-line arguments for consistency.

    Args:
        args: Parsed arguments from get_args()

    Raises:
        ValueError: If arguments are inconsistent or invalid
    """
    # Check domain parameters
    if args.Nx <= 0 or args.Ny <= 0:
        raise ValueError("Grid dimensions Nx and Ny must be positive")

    if args.Lx <= 0 or args.Ly <= 0:
        raise ValueError("Domain lengths Lx and Ly must be positive")

    if args.dealias < 1.0:
        raise ValueError("Dealias factor must be >= 1.0 (1.5+ recommended)")

    # Check physics parameters
    if args.nu <= 0:
        raise ValueError("Viscosity nu must be positive")

    if args.alpha < 0:
        raise ValueError("Friction coefficient alpha must be non-negative")

    # Check forcing parameters
    if args.forcing == "stochastic":
        if args.kmin < 0 or args.kmax < 0:
            raise ValueError("Forcing wavenumbers kmin and kmax must be non-negative")
        if args.kmin >= args.kmax:
            raise ValueError("Must have kmin < kmax for forcing band")
        if args.f_sigma <= 0:
            raise ValueError("Base forcing amplitude f_sigma must be positive")
        if args.stoch_type == "ou" and args.tau_ou <= 0:
            raise ValueError("OU correlation time tau_ou must be positive")
        if args.power_mode == "constant" and args.eps_target <= 0:
            raise ValueError("Target injection rate eps_target must be positive when power_mode='constant'")

    if args.forcing == "kolmogorov":
        if args.kolmogorov_f0 <= 0:
            raise ValueError("Kolmogorov forcing amplitude kolmogorov_f0 must be positive")
        if args.k_drive <= 0:
            raise ValueError("Driving wavenumber k_drive must be positive")

    # Check time parameters
    if args.t_end <= 0:
        raise ValueError("End time t_end must be positive")

    if args.cfl_max_dt <= args.cfl_min_dt:
        raise ValueError("Must have cfl_min_dt < cfl_max_dt")

    # Check output parameters
    if args.snap_dt <= 0 or args.spectra_dt <= 0 or args.scalars_dt <= 0:
        raise ValueError("All output intervals must be positive")

    if args.n_realisations <= 0:
        raise ValueError("Number of realisations must be positive")

    if args.ic_energy is not None and args.ic_energy <= 0:
        raise ValueError("ic_energy must be positive when specified")
