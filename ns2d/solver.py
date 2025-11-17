"""
Main solver setup and time integration for 2D Navier-Stokes simulations.

This module contains the core simulation logic:
- Dedalus problem setup (equations, boundary conditions)
- Initial condition setup and MPI broadcasting
- Forcing initialisation and rescaling
- Time integration loop with CFL control
- Output handlers for snapshots, scalars, and spectra
"""

import logging
import pathlib
import h5py
import numpy as np
import dedalus.public as d3
from mpi4py import MPI

from . import domain
from . import forcing
from . import spectral
from . import utils

logger = logging.getLogger(__name__)


def setup_problem(u, p, tau_p, forcing_vec, nu, alpha, coords, xbasis, ybasis):
    """
    Build the Dedalus IVP for 2D incompressible Navier-Stokes equations.

    Equations:
        ∂u/∂t + ∇p - ν∇²u = -(u·∇)u - αu + f
        ∇·u + τ_p = 0
        ∫p = 0

    Args:
        u: Velocity vector field
        p: Pressure scalar field
        tau_p: Pressure gauge condition tau variable
        forcing_vec: External forcing vector field
        nu (float): Kinematic viscosity
        alpha (float): Linear friction coefficient
        coords: Dedalus coordinate system
        xbasis: x-direction Fourier basis
        ybasis: y-direction Fourier basis

    Returns:
        d3.IVP: Configured Dedalus initial value problem
    """
    # Define operators
    grad = lambda q: d3.grad(q)
    lap = lambda q: d3.lap(q)

    # Build problem
    problem = d3.IVP([u, p, tau_p], namespace=locals())
    problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u) - alpha*u + forcing_vec")
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("integ(p) = 0")

    return problem


def initialise_fields(args, dist, coords, xbasis, ybasis, dtype, comm, r):
    """
    initialise velocity, pressure, and forcing fields.

    Creates fields and sets initial conditions by:
    1. Generating random vorticity on rank 0
    2. Converting to velocity via streamfunction
    3. Broadcasting to all MPI ranks

    Args:
        args: Parsed command-line arguments
        dist: Dedalus distributor
        coords: Dedalus coordinate system
        xbasis: x-direction basis
        ybasis: y-direction basis
        dtype: NumPy data type
        comm: MPI communicator
        r (int): Realisation index (for seeding RNG)

    Returns:
        tuple: (u, p, tau_p, forcing_vec, psi, kx, ky, KX, KY, K2, K)
            All Dedalus fields and wavenumber grids needed for simulation
    """
    # Create fields
    u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))
    p = dist.Field(name='p', bases=(xbasis, ybasis))
    tau_p = dist.Field(name='tau_p')
    forcing_vec = dist.VectorField(coords, name='forcing', bases=(xbasis, ybasis))

    # Wavenumber grids
    kx, ky, KX, KY, K2, K = domain.wavenumbers(args.Nx, args.Ny, args.Lx, args.Ly)

    # RNG with realisation-dependent seed
    rng = np.random.default_rng(args.seed + r)

    # Generate initial condition on rank 0, then broadcast
    if comm.rank == 0:
        w_hat = domain.initial_condition(rng, K2, args.Ny)
        ux0_grid, uy0_grid, psi0_grid = domain.vorticity_to_velocity(
            w_hat, KX, KY, K2, args.Nx, args.Ny
        )
        # Convert to requested precision
        ux0_grid = ux0_grid.astype(dtype, copy=False)
        uy0_grid = uy0_grid.astype(dtype, copy=False)
    else:
        ux0_grid = np.empty((args.Nx, args.Ny), dtype=dtype)
        uy0_grid = np.empty((args.Nx, args.Ny), dtype=dtype)

    # Broadcast to all ranks
    comm.Bcast(ux0_grid, root=0)
    comm.Bcast(uy0_grid, root=0)

    # Assign to local slices
    u.change_scales(1)
    u_slices = utils.local_slices(u)
    u['g'][0] = ux0_grid[u_slices]
    u['g'][1] = uy0_grid[u_slices]

    # Log initial diagnostics
    u0_rms = utils.global_rms_u(u, comm, args.Nx, args.Ny)
    if comm.rank == 0:
        u0_max = utils.compute_max_velocity(ux0_grid, uy0_grid)
        Re0 = u0_rms * np.sqrt(args.Lx * args.Ly) / args.nu
        logger.info("[run %d] Initial conditions: max|u|=%.3e, Re_box=%.3e", r, u0_max, Re0)

    return u, p, tau_p, forcing_vec, kx, ky, KX, KY, K2, K


def setup_forcing(args, forcing_vec, coords, dist, xbasis, ybasis, KX, KY, K, comm, forcing_seed):
    """
    Setup forcing function with optional constant-power rescaling.

    Args:
        args: Command-line arguments
        forcing_vec: Dedalus forcing vector field
        coords: Coordinate system
        dist: Dedalus distributor
        xbasis, ybasis: RealFourier bases
        KX, KY, K: Wavenumber grids
        comm: MPI communicator
        forcing_seed (int): Seed for distributed forcing generator

    Returns:
        callable: update_forcing(dt, u) function that updates forcing_vec;
                  applies constant-power only if power_mode=='constant'.
    """
    forcing_vec.change_scales(1)

    if args.forcing == "none":
        def update_forcing(dt, u):
            forcing_vec['g'][0] = 0.0
            forcing_vec['g'][1] = 0.0
        return update_forcing

    shell_mask = forcing.build_forcing_mask(K, args.kmin, args.kmax)
    stype = "ou" if args.stoch_type == "ou" else "white"

    stoch_update = forcing.distributed_stochastic_forcing(
        dist,
        coords,
        xbasis,
        ybasis,
        KX,
        KY,
        shell_mask,
        sigma_base=args.f_sigma,
        seed=forcing_seed,
        stype=stype,
        tau=args.tau_ou,
    )

    # State for exponential smoothing
    scale_state = np.array([1.0], dtype=np.float64)

    def update_forcing(dt, u):
        """Update forcing, optionally applying constant-power rescaling."""
        forcing_field = stoch_update(dt)
        forcing_field.require_grid_space()

        forcing_vec.change_scales(1)
        u.change_scales(1)
        fx_loc = forcing_field['g'][0]
        fy_loc = forcing_field['g'][1]
        ux_loc = u['g'][0]
        uy_loc = u['g'][1]

        if getattr(args, 'power_mode', 'constant') == 'constant':
            # Apply constant-power rescaling
            fx_rescaled, fy_rescaled = forcing.constant_power_rescale(
                fx_loc, fy_loc, ux_loc, uy_loc, comm, args.Nx, args.Ny,
                args.eps_target, args.eps_floor, args.eps_clip,
                scale_state, args.eps_smooth
            )
            forcing_vec['g'][0] = fx_rescaled
            forcing_vec['g'][1] = fy_rescaled
        else:
            # 'sigma' mode: use generator amplitude directly
            forcing_vec['g'][0] = fx_loc
            forcing_vec['g'][1] = fy_loc

    return update_forcing


def setup_output_handlers(solver, u, p, omega_expr, forcing_vec, args, r, run_dir):
    """
    Setup Dedalus file handlers for snapshots, scalars, and time series.

    Args:
        solver: Dedalus solver instance
        u: Velocity field
        p: Pressure field
        omega_expr: Vorticity expression
        forcing_vec: Forcing vector field
        args: Command-line arguments
        r (int): Realisation index
        run_dir (Path): Output directory for this realisation

    Returns:
        dict: Dictionary of file handlers
    """
    # Snapshot handler (fields)
    snapshots = solver.evaluator.add_file_handler(
        str(run_dir / "snapshots"),
        sim_dt=args.snap_dt,
        max_writes=None
    )
    snapshots.add_task(u, name="velocity")
    snapshots.add_task(p, name="pressure")
    snapshots.add_task(omega_expr, name="vorticity")
    snapshots.add_task(forcing_vec, name="forcing")

    # Scalar handler (time series)
    scalars = solver.evaluator.add_file_handler(
        str(run_dir / "scalars"),
        sim_dt=args.scalars_dt,
        max_writes=None
    )

    # Energy, enstrophy and palinstrophy
    E = 0.5 * d3.integ(u @ u)
    Z = d3.integ(omega_expr * omega_expr)
    P = d3.integ(d3.grad(omega_expr) @ d3.grad(omega_expr))
    scalars.add_task(E, name="energy")
    scalars.add_task(Z, name="enstrophy")
    scalars.add_task(P, name="palinstrophy")

    # Energy budget terms
    scalars.add_task(d3.integ(u @ forcing_vec), name="energy_injection")  # ε_i = ∫ u·f
    scalars.add_task(2 * args.alpha * E,        name="drag_loss")         # ε_α = 2αE = α∫|u|²
    scalars.add_task(args.nu * Z,               name="visc_loss")         # ε_ν = ν∫ω²

    # Enstrophy budget terms
    omega_forcing = -d3.div(d3.skew(forcing_vec))   # F = (curl f)_z
    scalars.add_task(2 * d3.integ(omega_expr * omega_forcing),
                     name="enstrophy_injection")    # 2∫ ω F
    scalars.add_task(2 * args.alpha * Z,
                     name="enstrophy_drag_loss")    # 2α ∫ ω²
    scalars.add_task(2 * args.nu * P,
                     name="enstrophy_visc_loss")    # 2ν ∫ |∇ω|²

    return {'snapshots': snapshots, 'scalars': scalars}


def setup_spectra_output(run_dir, spectra_dt):
    """
    Create HDF5 file for spectra and flux output.

    Args:
        run_dir (Path): Output directory

    Returns:
        tuple: (spectra_file, last_spec_t, logged_flag)
    """
    spectra_file = run_dir / "spectra.h5"
    last_spec_t = -1e99
    next_spec_t = spectra_dt
    spectra_gather_logged = [False]  # Mutable flag for logging

    return spectra_file, last_spec_t, next_spec_t, spectra_gather_logged


def write_spectra(solver, u, dist, xbasis, ybasis, comm, args, spectra_file, last_spec_t, logged_flag, r, next_spec_t):
    """
    Compute and write spectral diagnostics to HDF5.

    Args:
        solver: Dedalus solver
        u: Velocity field
        dist: Dedalus distributor
        xbasis, ybasis: RealFourier bases
        comm: MPI communicator
        args: Command-line arguments
        spectra_file (Path): Output HDF5 file
        last_spec_t (float): Last output time (modified in place)
        logged_flag (list): [bool] for logging gather success
        r (int): Realisation index

    Returns:
        tuple: (last_spec_t, next_spec_t)
    """
    if solver.sim_time + 1e-12 < next_spec_t:
        return last_spec_t, next_spec_t

    last_spec_t = solver.sim_time
    next_spec_t = last_spec_t + args.spectra_dt

    spectra_data = spectral.compute_spectra_from_coeffs(u, dist, xbasis, ybasis, args.Lx, args.Ly)
    energy_flux_data = spectral.compute_energy_flux_from_coeffs(u, dist, xbasis, ybasis, args.Lx, args.Ly)
    enstrophy_flux_data = spectral.compute_enstrophy_flux_from_coeffs(u, dist, xbasis, ybasis, args.Lx, args.Ly)

    if comm.rank != 0:
        return last_spec_t, next_spec_t

    if spectra_data is None or energy_flux_data is None or enstrophy_flux_data is None:
        return last_spec_t, next_spec_t

    if not logged_flag[0]:
        logger.info(
            "[run %d] Spectra/flux diagnostics computed from coefficient space on %d MPI processes.",
            r,
            comm.size,
        )
        logged_flag[0] = True

    k_bins, E_k, Z_k = spectra_data
    with h5py.File(spectra_file, "a") as h5:
        dset = f"k_E_Z_t{solver.sim_time:.6f}"
        if dset in h5:
            del h5[dset]
        h5.create_dataset(dset, data=np.vstack([k_bins, E_k, Z_k]).T)

    k_bins2, T_k, Pi_k = energy_flux_data
    with h5py.File(spectra_file, "a") as h5:
        dname = f"flux_T_Pi_t{solver.sim_time:.6f}"
        if dname in h5:
            del h5[dname]
        h5.create_dataset(dname, data=np.vstack([k_bins2, T_k, Pi_k]).T)

    k_bins_Z, TZ_k, PiZ_k = enstrophy_flux_data
    with h5py.File(spectra_file, "a") as h5:
        dnameZ = f"enstrophy_flux_T_Pi_t{solver.sim_time:.6f}"
        if dnameZ in h5:
            del h5[dnameZ]
        h5.create_dataset(dnameZ, data=np.vstack([k_bins_Z, TZ_k, PiZ_k]).T)

    return last_spec_t, next_spec_t


def run_single_realisation(args, r, dtype):
    """
    Run a single realisation of the 2D Navier-Stokes simulation.

    This is the main simulation driver that:
    1. Sets up domain and fields
    2. initialises forcing and initial conditions
    3. Configures Dedalus solver and output
    4. Runs time integration loop with CFL control
    5. Writes diagnostic output

    Args:
        args: Parsed command-line arguments
        r (int): Realisation index (0, 1, 2, ...)
        dtype: NumPy data type for simulation (np.float64 or np.float32)
    """
    # Setup MPI mesh if requested
    mesh = None
    if args.procs_x > 0 and args.procs_y > 0:
        mesh = (args.procs_x, args.procs_y)

    # Build domain
    coords, dist, xbasis, ybasis, x, y = domain.build_domain(
        args.Nx, args.Ny, args.Lx, args.Ly, args.dealias, dtype, mesh=mesh
    )
    comm = dist.comm

    # Initialise fields
    u, p, tau_p, forcing_vec, kx, ky, KX, KY, K2, K = initialise_fields(
        args, dist, coords, xbasis, ybasis, dtype, comm, r
    )

    # Setup problem
    problem = setup_problem(u, p, tau_p, forcing_vec, args.nu, args.alpha, coords, xbasis, ybasis)

    # Build solver
    timestepper = d3.RK222
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = args.t_end

    # Setup forcing
    rng = np.random.default_rng(args.seed + r)
    update_forcing = setup_forcing(
        args,
        forcing_vec,
        coords,
        dist,
        xbasis,
        ybasis,
        KX,
        KY,
        K,
        comm,
        args.seed + r,
    )

    # Setup output directories
    tag = (args.tag + "_") if args.tag else ""
    nu_str = f"nu{args.nu:.0e}"
    root = pathlib.Path(args.outdir) / f"{tag}Nx{args.Nx}_Ny{args.Ny}_{nu_str}"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"realisation_{r:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup output handlers
    omega_expr = -d3.div(d3.skew(u))
    handlers = setup_output_handlers(solver, u, p, omega_expr, forcing_vec, args, r, run_dir)
    spectra_file, last_spec_t, next_spec_t, spectra_logged = setup_spectra_output(run_dir, args.spectra_dt)

    # CFL controller
    CFL = d3.CFL(
        solver,
        initial_dt=args.cfl_max_dt,
        cadence=args.cfl_cadence,
        safety=args.cfl_safety,
        threshold=args.cfl_threshold,
        max_change=1.5,
        min_change=0.5,
        max_dt=args.cfl_max_dt,
        min_dt=args.cfl_min_dt,
    )
    CFL.add_velocity(u)

    # Flow properties for monitoring
    flow = d3.GlobalFlowProperty(solver, cadence=args.cfl_cadence)
    flow.add_property(np.sqrt(u @ u), name='speed')

    # Main time integration loop
    try:
        if comm.rank == 0:
            logger.info("[run %d] Starting time integration", r)

        while solver.proceed:
            # Compute timestep
            dt = CFL.compute_timestep()

            # Update forcing
            update_forcing(dt, u)

            # Take timestep
            solver.step(dt)

            # Write spectra
            last_spec_t, next_spec_t = write_spectra(
                solver,
                u,
                dist,
                xbasis,
                ybasis,
                comm,
                args,
                spectra_file,
                last_spec_t,
                spectra_logged,
                r,
                next_spec_t,
            )

            # Log progress
            if (solver.iteration - 1) % 10 == 0:
                max_speed = flow.max('speed')
                Re_info = utils.compute_reynolds_numbers(
                    u, comm, args.Nx, args.Ny, args.Lx, args.Ly, args.nu,
                    args.kmin, args.kmax
                )

                if comm.rank == 0:
                    logger.info(
                        "[run %d] it=%6d t=%9.4f dt=%8.2e max|u|=%10.3e Re_box=%9.3e Re_f=%9.3e",
                        r, solver.iteration, solver.sim_time, dt, max_speed,
                        Re_info['Re_box'], Re_info['Re_f']
                    )

    except Exception:
        if comm.rank == 0:
            logger.exception("[run %d] Exception in main loop", r)
        raise
    finally:
        try:
            solver.log_stats()
        except Exception:
            pass

    if comm.rank == 0:
        logger.info("[run %d] Simulation complete", r)
