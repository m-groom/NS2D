"""
Main solver setup and time integration for 2D Navier-Stokes simulations.

This module contains the core simulation logic:
- Dedalus problem setup (equations, boundary conditions)
- Initial condition setup and MPI broadcasting
- Forcing initialisation and rescaling
- Time integration loop with CFL control
- Output handlers for snapshots, scalars, and spectra
"""

import datetime
import json
import logging
import pathlib
import platform
import subprocess
import h5py
import numpy as np
import dedalus.public as d3

from . import domain
from . import forcing
from . import spectral
from . import utils

logger = logging.getLogger(__name__)

# Sign conventions of the derived output fields, recorded with the data.
# omega = dx(v) - dy(u) is what -div(skew(u)) evaluates to; psi then satisfies
# -lap(psi) = omega, so u = (dy(psi), -dx(psi)). This is the convention used by
# post/fields.py, and it is the opposite of the more common lap(psi) = +omega.
FIELD_CONVENTIONS = {
    'vorticity_convention': 'omega = dx(v) - dy(u)',
    'streamfunction_convention': 'lap(psi) = -omega, u = (dy(psi), -dx(psi))',
}


def _git_provenance():
    """
    Return the current commit hash and dirty flag of the NS2D working tree.

    Returns:
        dict: {'commit': str or None, 'dirty': bool or None}
    """
    package_dir = pathlib.Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=package_dir,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=package_dir,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return {'commit': commit, 'dirty': bool(status)}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def write_run_metadata(run_dir, args, comm, extra=None):
    """
    Append this invocation's configuration and provenance to run_config.jsonl.

    One JSON object is appended per invocation, so a run that is restarted keeps
    the record of every command line that produced its output.

    Args:
        run_dir (Path): Run output directory
        args: Parsed command-line arguments
        comm: MPI communicator
        extra (dict or None): Additional key/value pairs to record

    Returns:
        dict: The record. Its 'git' entry is identical on every rank, so it is
            safe to derive collectively written file attributes from it; the
            timestamp and hostname are each rank's own. Rank 0 writes the file.
    """
    import dedalus

    # Only rank 0 shells out to git, then shares the result: two subprocesses
    # per rank against one .git directory does not scale.
    git = comm.bcast(_git_provenance() if comm.rank == 0 else None, root=0)

    record = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git': git,
        'args': vars(args),
        'mpi_size': comm.size,
        'hostname': platform.node(),
        'python_version': platform.python_version(),
        'dedalus_version': dedalus.__version__,
        'numpy_version': np.__version__,
    }
    if extra:
        record.update(extra)

    if comm.rank == 0:
        with open(run_dir / "run_config.jsonl", 'a') as f:
            f.write(json.dumps(record, default=str) + "\n")

    return record


def stamp_file_attributes(handler, attrs):
    """
    Write *attrs* into the root group of every HDF5 set a handler creates.

    Dedalus offers no hook for custom file metadata, so wrap the handler's own
    file-setup step. This covers sets created mid-run and after a restart.

    Args:
        handler: Dedalus file handler
        attrs (dict): Attribute names and values
    """
    setup_file = handler.setup_file

    def setup_file_with_attrs(file):
        setup_file(file)
        file.attrs.update(attrs)

    handler.setup_file = setup_file_with_attrs


def find_latest_checkpoint(run_dir):
    """
    Find the most recent checkpoint file in run_dir/checkpoints/.

    Dedalus checkpoint files are named checkpoints_s{N}.h5 where N is the
    write number. We find the file with the highest N.

    Args:
        run_dir (Path): Directory containing the checkpoints/ subdirectory

    Returns:
        Path or None: Path to the most recent checkpoint file, or None if none exist
    """
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    # Dedalus checkpoint files are named: checkpoints_s{N}.h5. Sort on the
    # integer N, since a lexicographic sort puts s9 after s10.
    checkpoint_files = sorted(
        checkpoint_dir.glob("checkpoints_s*.h5"),
        key=lambda f: int(f.stem.rsplit("_s", 1)[1]),
    )
    if not checkpoint_files:
        return None

    return checkpoint_files[-1]


def load_auxiliary_state(run_dir, target_time, comm):
    """
    Load auxiliary simulation state matching the target simulation time.

    Auxiliary state includes forcing OU state and spectra timing that are
    not captured by Dedalus's built-in checkpoint system.

    Args:
        run_dir (Path): Run output directory
        target_time (float): Target simulation time to match
        comm: MPI communicator

    Returns:
        dict or None: Dictionary with 'forcing' and 'spectra' sub-dicts,
                      or None if no matching auxiliary state found
    """
    aux_dir = run_dir / "auxiliary_state"
    if not aux_dir.exists():
        return None

    # Find auxiliary file closest to target_time
    aux_files = sorted(aux_dir.glob("aux_t*.h5"))
    if not aux_files:
        return None

    # Find best match by parsing time from filename
    best_file = None
    best_diff = float('inf')
    for f in aux_files:
        try:
            t = float(f.stem.split('_t')[1])
            if abs(t - target_time) < best_diff:
                best_diff = abs(t - target_time)
                best_file = f
        except (ValueError, IndexError):
            continue

    if best_file is None or best_diff > 1e-3:
        if comm.rank == 0:
            logger.warning(
                "No auxiliary state found matching t=%.6f (best diff=%.6f)",
                target_time, best_diff
            )
        return None

    result = {}
    per_rank_states = None
    error = None
    if comm.rank == 0:
        try:
            with h5py.File(best_file, 'r') as f:
                if 'forcing' in f:
                    n_ranks = int(f['forcing'].attrs['n_ranks'])
                    if n_ranks != comm.size:
                        raise ValueError(
                            f"auxiliary state was written by {n_ranks} MPI ranks "
                            f"but this run has {comm.size}; restart with the same "
                            "number of ranks"
                        )
                    per_rank_states = [
                        f[f'forcing/state_potential_rank{i}'][...] for i in range(n_ranks)
                    ]
                    result['forcing'] = {
                        'scale_state': f['forcing/scale_state'][...],
                        # step_counter is stored as a shape-(1,) array.
                        'step_counter': int(
                            np.asarray(f['forcing/step_counter']).ravel()[0]
                        ),
                    }
                if 'spectra' in f:
                    result['spectra'] = {
                        'next_spec_t': float(f['spectra'].attrs['next_spec_t']),
                    }
            logger.info("Loaded auxiliary state from %s", best_file)
        except Exception as exc:  # report collectively rather than deadlocking
            error = f"{type(exc).__name__}: {exc}"

    # Broadcast the outcome so that a failure on rank 0 raises on every rank
    # instead of leaving the others blocked in the broadcast below.
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(
            f"Failed to load auxiliary state from {best_file}: {error}"
        )

    result = comm.bcast(result, root=0)

    # Each rank takes its own slice of the forcing coefficients.
    if 'forcing' in result:
        result['forcing']['state_potential'] = comm.scatter(per_rank_states, root=0)
    return result


def save_auxiliary_state(run_dir, sim_time, forcing_state, spectra_state, comm):
    """
    Save auxiliary simulation state not captured by Dedalus checkpoints.

    This saves the internal state of stochastic forcing generators and
    spectra output timing to enable bit-identical restarts.

    Args:
        run_dir (Path): Run output directory
        sim_time (float): Current simulation time
        forcing_state (dict or None): Forcing state with 'state_potential',
                                       'scale_state', 'step_counter'
        spectra_state (dict): Spectra state with 'next_spec_t'
        comm: MPI communicator
    """
    # Every rank holds its own slice of the forcing coefficients, so collect all
    # of them before rank 0 writes. Restoring rank 0's slice onto every rank
    # corrupts a parallel restart.
    per_rank_states = None
    if forcing_state is not None:
        per_rank_states = comm.gather(forcing_state['state_potential'], root=0)

    if comm.rank != 0:
        return

    aux_dir = run_dir / "auxiliary_state"
    aux_dir.mkdir(parents=True, exist_ok=True)
    aux_file = aux_dir / f"aux_t{sim_time:.6f}.h5"

    with h5py.File(aux_file, 'w') as f:
        f.attrs['sim_time'] = sim_time

        # Forcing state (if stochastic)
        if forcing_state is not None:
            grp = f.create_group('forcing')
            grp.attrs['n_ranks'] = comm.size
            for rank, state in enumerate(per_rank_states):
                grp.create_dataset(f'state_potential_rank{rank}', data=state)
            grp.create_dataset('scale_state', data=forcing_state['scale_state'])
            grp.create_dataset('step_counter', data=forcing_state['step_counter'])

        # Spectra output state
        grp = f.create_group('spectra')
        grp.attrs['next_spec_t'] = spectra_state['next_spec_t']

    logger.debug("Saved auxiliary state to %s", aux_file)


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

    # RNG with realisation-dependent seed (allowing decoupled IC seed)
    ic_seed_base = args.ic_seed if args.ic_seed is not None else args.seed
    rng = np.random.default_rng(ic_seed_base + r)

    # Generate initial condition on rank 0, then broadcast
    if comm.rank == 0:
        w_hat = domain.initial_condition(
            rng,
            K2,
            args.Ny,
            alpha=args.ic_alpha,
            power=args.ic_power,
            scale=args.ic_scale,
            KX=KX,
            KY=KY,
            Lx=args.Lx,
            Ly=args.Ly,
            l_ref=1.0,
        )
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

    # Optionally rescale to a target kinetic energy
    if args.ic_energy is not None:
        # Every rank holds the whole broadcast field, so this is already the
        # global energy; reducing it across ranks would inflate it by the rank
        # count and leave the initial energy a factor of 1/nranks too small.
        total_energy = 0.5 * np.sum(ux0_grid**2 + uy0_grid**2, dtype=np.float64)
        # domain-averaged kinetic energy 0.5<|u|^2>
        current_ic_energy = total_energy / (args.Nx * args.Ny)
        if current_ic_energy > 0:
            rescale = np.sqrt(args.ic_energy / current_ic_energy)
            ux0_grid *= rescale
            uy0_grid *= rescale
            if comm.rank == 0:
                logger.info(
                    "[run %d] Rescaled IC to target energy %.3e (factor %.3e)",
                    r, args.ic_energy, rescale,
                )
        elif comm.rank == 0:
            logger.warning(
                "[run %d] Initial conditions had zero energy; skipping IC rescale", r
            )

    # Assign to local slices
    u.change_scales(1)
    u_slices = utils.local_slices(u)
    u['g'][0] = ux0_grid[u_slices]
    u['g'][1] = uy0_grid[u_slices]

    # Log initial diagnostics
    u0_rms = utils.global_rms_u(u, comm, args.Nx, args.Ny)
    if comm.rank == 0:
        ic_seed_used = ic_seed_base + r
        E0 = 0.5 * u0_rms * u0_rms
        u0_max = utils.compute_max_velocity(ux0_grid, uy0_grid)
        Re0 = u0_rms * np.sqrt(args.Lx * args.Ly) / args.nu
        logger.info(
            "[run %d] Initial conditions (seed=%d): max|u|=%.3e, RMS=%.3e, Re_box=%.3e, E=%.3e",
            r, ic_seed_used, u0_max, u0_rms, Re0, E0,
        )

    return u, p, tau_p, forcing_vec, kx, ky, KX, KY, K2, K


def setup_forcing(args, forcing_vec, coords, dist, xbasis, ybasis, KX, KY, K, comm,
                  forcing_seed, realisation=0, initial_forcing_state=None):
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
        forcing_seed (int): Base seed for distributed forcing generator
        realisation (int): Realisation index, hashed into the per-step seed so
            that realisations draw independent noise
        initial_forcing_state (dict or None): Initial state for restart, with keys
            'state_potential', 'scale_state', 'step_counter'

    Returns:
        tuple: (update_forcing, forcing_state_refs)
            - update_forcing: callable(dt, u) that updates forcing_vec
            - forcing_state_refs: dict with references to internal state arrays
              for checkpoint saving, or None for non-stochastic forcing
    """
    forcing_vec.change_scales(1)

    if args.forcing == "none":
        def update_forcing(dt, u):
            forcing_vec['g'][0] = 0.0
            forcing_vec['g'][1] = 0.0
        return update_forcing, None

    # State for exponential smoothing
    scale_state = np.array([1.0], dtype=np.float64)

    # Restore scale_state if restarting
    if initial_forcing_state is not None and 'scale_state' in initial_forcing_state:
        scale_state[0] = initial_forcing_state['scale_state'][0]

    # Select forcing generator based on requested type
    forcing_state_refs = None

    if args.forcing == "stochastic":
        shell_mask = forcing.build_forcing_mask(K, args.kmin, args.kmax)
        stype = "ou" if args.stoch_type == "ou" else "white"

        generator, generator_state_refs = forcing.distributed_stochastic_forcing(
            dist,
            coords,
            xbasis,
            ybasis,
            KX,
            KY,
            shell_mask,
            sigma_base=args.f_sigma,
            seed=forcing_seed,
            realisation=realisation,
            stype=stype,
            tau=args.tau_ou,
            initial_state=initial_forcing_state,
        )

        # Build combined state references for checkpointing
        forcing_state_refs = {
            'state_potential': generator_state_refs['state_potential'],
            'step_counter': generator_state_refs['step_counter'],
            'scale_state': scale_state,
        }

    elif args.forcing == "kolmogorov":
        generator = forcing.distributed_kolmogorov_forcing(
            dist,
            coords,
            xbasis,
            ybasis,
            amplitude=args.kolmogorov_f0,
            k_drive=args.k_drive,
            Lx=args.Lx,
            Ly=args.Ly,
            phase=args.k_phase,
        )
    else:
        raise ValueError(f"Unsupported forcing type {args.forcing}")

    def update_forcing(dt, u):
        """Update forcing, optionally applying constant-power rescaling."""
        forcing_field = generator(dt) if args.forcing == "stochastic" else generator()
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

    return update_forcing, forcing_state_refs


def setup_output_handlers(solver, u, p, omega_expr, forcing_vec, args, r, run_dir,
                          file_handler_mode='overwrite', psi=None, file_attrs=None):
    """
    Setup Dedalus file handlers for snapshots, scalars, checkpoints and time series.

    Args:
        solver: Dedalus solver instance
        u: Velocity field
        p: Pressure field
        omega_expr: Vorticity expression
        forcing_vec: Forcing vector field
        args: Command-line arguments
        r (int): Realisation index
        run_dir (Path): Output directory for this realisation
        file_handler_mode (str): 'overwrite' for new runs, 'append' for restarts
        psi: Streamfunction field for the coarse handler, or None
        file_attrs (dict or None): Attributes to stamp on every HDF5 set

    Returns:
        dict: Dictionary of file handlers
    """
    # Snapshot handler (fields)
    snapshots = solver.evaluator.add_file_handler(
        str(run_dir / "snapshots"),
        sim_dt=args.snap_dt,
        max_writes=None,
        mode=file_handler_mode
    )
    snapshots.add_task(u, name="velocity")
    snapshots.add_task(p, name="pressure")
    snapshots.add_task(omega_expr, name="vorticity")
    snapshots.add_task(forcing_vec, name="forcing")

    # Scalar handler (time series)
    scalars = solver.evaluator.add_file_handler(
        str(run_dir / "scalars"),
        sim_dt=args.scalars_dt,
        max_writes=None,
        mode=file_handler_mode
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

    # Checkpoint handler (saves full solver state for restart)
    checkpoints = solver.evaluator.add_file_handler(
        str(run_dir / "checkpoints"),
        sim_dt=args.checkpoint_dt,
        max_writes=None,  # Keep all checkpoints as requested
        mode=file_handler_mode
    )
    checkpoints.add_tasks(solver.state)

    handlers = {'snapshots': snapshots, 'scalars': scalars, 'checkpoints': checkpoints}

    # In-situ coarse handler: the same fields spectrally truncated to coarse_N,
    # written densely in time. add_task(..., scales=s) truncates in coefficient
    # space, so the result equals an offline sharp truncation of the full field.
    if args.coarse_N:
        coarse_scales = (args.coarse_N / args.Nx, args.coarse_N / args.Ny)
        coarse = solver.evaluator.add_file_handler(
            str(run_dir / "coarse"),
            sim_dt=args.coarse_dt,
            max_writes=None,
            mode=file_handler_mode
        )
        coarse.add_task(u, name="velocity", scales=coarse_scales)
        coarse.add_task(p, name="pressure", scales=coarse_scales)
        coarse.add_task(omega_expr, name="vorticity", scales=coarse_scales)
        coarse.add_task(forcing_vec, name="forcing", scales=coarse_scales)
        if psi is not None:
            coarse.add_task(psi, name="streamfunction", scales=coarse_scales)
        handlers['coarse'] = coarse

    if file_attrs:
        for handler in handlers.values():
            stamp_file_attributes(handler, file_attrs)

    return handlers


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
    2. Initialises forcing and initial conditions (or loads from checkpoint)
    3. Configures Dedalus solver and output
    4. Runs time integration loop with CFL control
    5. Writes diagnostic output and periodic checkpoints

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

    # Setup output directories (needed early to check for checkpoints)
    tag = (args.tag + "_") if args.tag else ""
    nu_str = f"nu{args.nu:.0e}"
    root = pathlib.Path(args.outdir) / f"{tag}Nx{args.Nx}_Ny{args.Ny}_{nu_str}"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"realisation_{r:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Record the full configuration and code provenance before doing any work.
    metadata = write_run_metadata(run_dir, args, comm, extra=FIELD_CONVENTIONS)
    file_attrs = dict(FIELD_CONVENTIONS)
    file_attrs['git_commit'] = metadata['git']['commit'] or 'unknown'
    file_attrs['git_dirty'] = str(metadata['git']['dirty'])

    # Determine restart mode and find checkpoint file
    restart_mode = False
    checkpoint_file = None
    initial_dt = args.cfl_max_dt

    if args.restart or args.restart_file:
        if args.restart_file:
            checkpoint_file = pathlib.Path(args.restart_file)
        else:
            checkpoint_file = find_latest_checkpoint(run_dir)

        if checkpoint_file is not None and checkpoint_file.exists():
            restart_mode = True
            if comm.rank == 0:
                logger.info("[run %d] Restart mode: found checkpoint %s", r, checkpoint_file)
        elif comm.rank == 0:
            if args.restart_file:
                logger.warning("[run %d] Specified checkpoint file not found: %s", r, args.restart_file)
            else:
                logger.info("[run %d] No checkpoint found, starting fresh", r)

    file_handler_mode = 'append' if restart_mode else 'overwrite'

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

    # Load checkpoint if restarting
    initial_forcing_state = None
    initial_spectra_state = None

    if restart_mode:
        write, initial_dt = solver.load_state(str(checkpoint_file))
        if comm.rank == 0:
            logger.info(
                "[run %d] Loaded checkpoint: write=%d, sim_time=%.6f, dt=%.2e",
                r, write, solver.sim_time, initial_dt
            )

        # Load auxiliary state (forcing OU state, spectra timing)
        aux_state = load_auxiliary_state(run_dir, solver.sim_time, comm)
        if aux_state is not None:
            initial_forcing_state = aux_state.get('forcing')
            initial_spectra_state = aux_state.get('spectra')
            if comm.rank == 0:
                logger.info("[run %d] Loaded auxiliary state for forcing and spectra", r)
        elif comm.rank == 0:
            logger.warning(
                "[run %d] No auxiliary state found; forcing will reinitialise (may affect statistics)",
                r
            )

    # Setup forcing (with optional restart state)
    update_forcing, forcing_state_refs = setup_forcing(
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
        args.seed,
        realisation=r,
        initial_forcing_state=initial_forcing_state,
    )

    # Setup output handlers with correct mode
    omega_expr = -d3.div(d3.skew(u))

    # Streamfunction for the coarse output, from -lap(psi) = omega. It is solved
    # only when the coarse handler is enabled; see FIELD_CONVENTIONS.
    psi = None
    psi_solver = None
    if args.coarse_N:
        psi = dist.Field(name='psi', bases=(xbasis, ybasis))
        tau_psi = dist.Field(name='tau_psi')
        psi_problem = d3.LBVP([psi, tau_psi])
        psi_problem.add_equation((-d3.lap(psi) + tau_psi, omega_expr))
        psi_problem.add_equation((d3.integ(psi), 0))
        psi_solver = psi_problem.build_solver()

    handlers = setup_output_handlers(
        solver, u, p, omega_expr, forcing_vec, args, r, run_dir,
        file_handler_mode=file_handler_mode, psi=psi, file_attrs=file_attrs
    )

    # Setup spectra output (with optional restart state)
    spectra_file, last_spec_t, next_spec_t, spectra_logged = setup_spectra_output(run_dir, args.spectra_dt)
    if initial_spectra_state is not None and 'next_spec_t' in initial_spectra_state:
        next_spec_t = initial_spectra_state['next_spec_t']
        last_spec_t = solver.sim_time

    # CFL controller
    CFL = d3.CFL(
        solver,
        initial_dt=initial_dt,
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

    # Track next checkpoint time for auxiliary state saving
    # We save auxiliary state BEFORE update_forcing when sim_time reaches a checkpoint time
    # This ensures the forcing state matches what Dedalus will save in the checkpoint
    if restart_mode:
        # After restart, next checkpoint will be at the next multiple of checkpoint_dt
        # Use ceiling to find next checkpoint time after current sim_time
        next_aux_checkpoint_time = (
            np.ceil(solver.sim_time / args.checkpoint_dt + 1e-10) * args.checkpoint_dt
        )
    else:
        # Fresh start: first checkpoint at t=0
        next_aux_checkpoint_time = 0.0

    # Main time integration loop
    try:
        if comm.rank == 0:
            if restart_mode:
                logger.info("[run %d] Resuming time integration from t=%.6f", r, solver.sim_time)
            else:
                logger.info("[run %d] Starting time integration", r)

        while solver.proceed:
            # Compute timestep
            dt = CFL.compute_timestep()

            # Check if we're at a checkpoint time and should save auxiliary state
            # This MUST happen BEFORE update_forcing to capture the correct forcing state
            # Dedalus will write checkpoint at this same sim_time during solver.step
            if solver.sim_time >= next_aux_checkpoint_time - 1e-10:
                spectra_state = {'next_spec_t': next_spec_t}
                save_auxiliary_state(run_dir, solver.sim_time, forcing_state_refs, spectra_state, comm)
                next_aux_checkpoint_time += args.checkpoint_dt

            # Update forcing (modifies forcing_state_refs in place)
            update_forcing(dt, u)

            # Solve for the streamfunction at the current state, so that the
            # coarse handler -- which Dedalus evaluates at the start of the
            # step -- writes psi at the same time as the other fields.
            if psi_solver is not None:
                psi_solver.solve()

            # Take timestep (Dedalus checkpoint may be written here)
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

    # Handle final checkpoint at simulation end
    # Dedalus file handlers only write during solver.step, so when t_end aligns with
    # checkpoint_dt, the final checkpoint may not be written (loop exits before next check)
    # Use evaluate_scheduled() to properly evaluate and write the current field state
    # (handler.process() writes stale cached data, not the current state)
    if solver.sim_time >= next_aux_checkpoint_time - 1e-10:
        # Trigger final checkpoint write with properly evaluated current state
        if psi_solver is not None:
            psi_solver.solve()
        solver.evaluator.evaluate_scheduled(
            wall_time=solver.wall_time,
            sim_time=solver.sim_time,
            iteration=solver.iteration,
            timestep=dt
        )
        if comm.rank == 0:
            logger.info("[run %d] Wrote final checkpoint at t=%.6f", r, solver.sim_time)

        # Save corresponding auxiliary state
        spectra_state = {'next_spec_t': next_spec_t}
        save_auxiliary_state(run_dir, solver.sim_time, forcing_state_refs, spectra_state, comm)
        if comm.rank == 0:
            logger.info("[run %d] Saved final auxiliary state at t=%.6f", r, solver.sim_time)

    if comm.rank == 0:
        logger.info("[run %d] Simulation complete", r)
