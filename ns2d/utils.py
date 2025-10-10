"""
MPI utilities and diagnostic functions for distributed Dedalus simulations.

This module provides helper functions for:
- Gathering distributed Dedalus fields to a single MPI rank
- Computing global diagnostics (RMS velocity, Reynolds numbers, etc.)
- Local grid slicing operations
"""

import logging
import numpy as np
from mpi4py import MPI

logger = logging.getLogger(__name__)


def gather_field_to_rank0(field, comm, global_shape):
    """
    Gather a distributed Dedalus field to rank 0.

    Attempts multiple strategies to gather field data from all MPI ranks
    to rank 0 for analysis or I/O. Falls back gracefully if gathering fails.

    Args:
        field: Dedalus field object (distributed across MPI ranks)
        comm: MPI communicator
        global_shape (tuple): Expected global shape of the field in grid space

    Returns:
        ndarray or None: Full field data on rank 0, None on other ranks.
            If gathering fails, returns local data on rank 0 with a warning.
    """
    field.change_scales(1)  # Ensure full resolution

    # Strategy 1: Try Dedalus built-in allgather (if available)
    try:
        if hasattr(field, 'allgather_data'):
            global_data = field.allgather_data('g')
            if comm.rank == 0:
                return global_data
            return None
    except (AttributeError, NotImplementedError):
        pass

    # Strategy 2: Check if data is already global on this rank
    try:
        local_data = field['g']
        if local_data.shape == global_shape:
            if comm.rank == 0:
                return local_data.copy()
            return None

        # Strategy 3: Manual gather using MPI
        layout = field.dist.grid_layout
        local_slices = layout.slices(field.domain, scales=1)

        all_slices = comm.gather(local_slices, root=0)
        all_data = comm.gather(local_data, root=0)

        if comm.rank == 0:
            global_array = np.zeros(global_shape, dtype=local_data.dtype)
            for rank_slices, rank_data in zip(all_slices, all_data):
                if rank_data is None:
                    continue
                target_index = rank_slices if isinstance(rank_slices, tuple) else (rank_slices,)

                # Handle vector fields (add leading dimension if needed)
                if rank_data.ndim == len(target_index) + 1 and global_array.ndim == len(target_index) + 1:
                    target_index = (slice(None),) + target_index

                global_array[target_index] = rank_data
            return global_array
        return None

    except Exception as exc:
        if comm.rank == 0:
            logger.warning("Failed to gather field: %s. Returning local data.", exc)
            return field['g']
        return None


def local_slices(field, scales=1):
    """
    Get the local grid slices for a Dedalus field.

    Args:
        field: Dedalus field object
        scales (float or tuple): Resolution scaling factor (1 = full resolution)

    Returns:
        tuple: Local slices indicating which part of the global grid this
            MPI rank owns.
    """
    layout = field.dist.grid_layout
    return layout.slices(field.domain, scales=scales)


def global_rms_u(u, comm, Nx, Ny):
    """
    Compute global RMS velocity from a distributed vector field.

    Uses MPI reduction to compute the domain-averaged RMS speed:
        u_rms = √(<|u|²>)

    Args:
        u: Dedalus vector field (velocity)
        comm: MPI communicator
        Nx (int): Global grid size in x
        Ny (int): Global grid size in y

    Returns:
        float: RMS velocity (same value on all ranks)
    """
    # Local sum of squared velocity
    usq_local = np.sum(u['g'][0]**2 + u['g'][1]**2, dtype=np.float64)

    # Global sum across all ranks
    usq_global = comm.allreduce(usq_local, op=MPI.SUM)

    # RMS velocity
    return np.sqrt(usq_global / (Nx * Ny))


def compute_reynolds_numbers(u, comm, Nx, Ny, Lx, Ly, nu, kmin=None, kmax=None):
    """
    Compute characteristic Reynolds numbers for the flow.

    Args:
        u: Dedalus vector field (velocity)
        comm: MPI communicator
        Nx (int): Global grid size in x
        Ny (int): Global grid size in y
        Lx (float): Domain length in x
        Ly (float): Domain length in y
        nu (float): Kinematic viscosity
        kmin (float, optional): Minimum forcing wavenumber (for Re_f)
        kmax (float, optional): Maximum forcing wavenumber (for Re_f)

    Returns:
        dict: Dictionary containing:
            - 'u_rms': RMS velocity
            - 'Re_box': Reynolds number based on box size
            - 'Re_f': Reynolds number based on forcing length scale (if kmin, kmax provided)
    """
    u_rms = global_rms_u(u, comm, Nx, Ny)
    L_box = np.sqrt(Lx * Ly)
    Re_box = u_rms * L_box / nu

    result = {
        'u_rms': u_rms,
        'Re_box': Re_box,
    }

    if kmin is not None and kmax is not None:
        kf = 0.5 * (kmin + kmax)
        if kf > 0:
            Lf = 2 * np.pi / kf
            Re_f = u_rms * Lf / nu
            result['Re_f'] = Re_f
        else:
            result['Re_f'] = np.nan
    else:
        result['Re_f'] = np.nan

    return result


def compute_max_velocity(ux_grid, uy_grid):
    """
    Compute maximum velocity magnitude from grid data.

    Args:
        ux_grid (ndarray): x-velocity in physical space
        uy_grid (ndarray): y-velocity in physical space

    Returns:
        float: Maximum velocity magnitude |u|_max
    """
    return np.sqrt(np.max(ux_grid * ux_grid + uy_grid * uy_grid))
