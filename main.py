#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NS2D: 2D Incompressible Navier-Stokes Solver
=============================================

Forced 2D incompressible Navier-Stokes simulations using Dedalus.

Usage:
    # Single core
    python main.py --Nx 512 --Ny 512 --nu 1e-4 --t_end 100

    # MPI parallel (8 processes)
    mpiexec -n 8 python main.py --Nx 1024 --Ny 1024 --nu 5e-5 --t_end 100

For help:
    python main.py --help
"""

import logging
import numpy as np
from mpi4py import MPI

from ns2d import config, solver


def main():
    """
    Main entry point for NS2D simulations.

    Parses command-line arguments, validates configuration, sets up logging,
    and runs the requested number of realisations.
    """
    # Parse arguments
    args = config.get_args()

    # Setup logging (INFO on rank 0, WARNING on others)
    logging.basicConfig(
        level=logging.INFO if MPI.COMM_WORLD.rank == 0 else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Validate arguments
    try:
        config.validate_args(args)
    except ValueError as e:
        if MPI.COMM_WORLD.rank == 0:
            logger.error("Invalid configuration: %s", e)
        MPI.COMM_WORLD.Abort(1)

    # Determine data type
    dtype = np.float64 if args.precision == "float64" else np.float32

    # Log configuration on rank 0
    if MPI.COMM_WORLD.rank == 0:
        logger.info("=" * 70)
        logger.info("NS2D: 2D Incompressible Navier-Stokes Solver")
        logger.info("=" * 70)
        logger.info("Domain: %dx%d grid on [0,%.2f]x[0,%.2f]",
                    args.Nx, args.Ny, args.Lx, args.Ly)
        logger.info("Physics: nu=%.2e, alpha=%.2e", args.nu, args.alpha)
        logger.info("Forcing: %s (kmin=%.1f, kmax=%.1f, eps_target=%.2e)",
                    args.forcing, args.kmin, args.kmax, args.eps_target)
        logger.info("Time: t_end=%.2f, CFL safety=%.2f", args.t_end, args.cfl_safety)
        logger.info("Precision: %s", args.precision)
        logger.info("MPI processes: %d", MPI.COMM_WORLD.size)
        if args.procs_x > 0 and args.procs_y > 0:
            logger.info("Process mesh: %dx%d", args.procs_x, args.procs_y)
        logger.info("Output directory: %s", args.outdir)
        logger.info("Number of realisations: %d", args.n_realisations)
        logger.info("=" * 70)

    # Run realisations
    for r in range(args.n_realisations):
        if MPI.COMM_WORLD.rank == 0:
            logger.info("")
            logger.info("Starting realisation %d/%d", r + 1, args.n_realisations)
            logger.info("-" * 70)

        solver.run_single_realisation(args, r, dtype)

        if MPI.COMM_WORLD.rank == 0:
            logger.info("-" * 70)
            logger.info("Completed realisation %d/%d", r + 1, args.n_realisations)

    # Final summary
    if MPI.COMM_WORLD.rank == 0:
        logger.info("")
        logger.info("=" * 70)
        logger.info("All %d realisation(s) completed successfully!", args.n_realisations)
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
