#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline streamfunction computation for NS2D snapshots.

This script reads existing snapshot files from an NS2D realisation,
computes the streamfunction from vorticity using the offline
`post.fields.vorticity_to_streamfunction` utility, and writes new
plot-ready snapshot files containing:

    - vorticity
    - pressure
    - streamfunction

The output layout is compatible with `scripts/plot_output.py`, so you
can point `plot_output.py --rundir` at the *destination* run directory
to generate figures that include the streamfunction field without
changing the plotting code.

Typical usage:

    python calculate_streamfunction.py \\
        --src-rundir path/to/original_run \\
        --dst-rundir path/to/plot_ready_run \\
        --Lx 6.283185307179586 --Ly 6.283185307179586

You can then run:

    python plot_output.py --rundir path/to/plot_ready_run --outdir ./figures
"""

import argparse
import pathlib
import sys
from typing import Tuple
import shutil

import h5py
import numpy as np

# Add parent directory to path to import post-processing modules
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from post import io, fields  # noqa: E402


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description=(
            "Compute streamfunction from vorticity in NS2D snapshot files "
            "and write plot-ready HDF5 snapshots containing "
            "vorticity, pressure, and streamfunction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required run directories
    ap.add_argument(
        "--src-rundir",
        type=str,
        required=True,
        help="Source run directory containing 'snapshots/' written by the solver.",
    )
    ap.add_argument(
        "--dst-rundir",
        type=str,
        required=True,
        help=(
            "Destination run directory where plot-ready 'snapshots/' will "
            "be written. This directory can be used with plot_output.py."
        ),
    )

    # Domain parameters for vorticity_to_streamfunction
    ap.add_argument(
        "--Lx",
        type=float,
        default=2 * np.pi,
        help="Domain length in x.",
    )
    ap.add_argument(
        "--Ly",
        type=float,
        default=2 * np.pi,
        help="Domain length in y.",
    )

    # Dedalus LBVP options
    ap.add_argument(
        "--dealias",
        type=float,
        default=1.0,
        help="Dealiasing factor for RealFourier bases in the LBVP.",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="float64",
        help="Floating-point dtype to use for the LBVP (e.g. 'float64', 'float32').",
    )

    # Behaviour flags
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing snapshot files in the destination directory.",
    )

    return ap.parse_args()


def _as_dtype(dtype_str: str) -> np.dtype:
    """Convert a string to a NumPy dtype with a clear error message."""
    try:
        return np.dtype(dtype_str)
    except TypeError as exc:
        raise ValueError(f"Invalid dtype '{dtype_str}'.") from exc


def _ensure_dirs(src_rundir: pathlib.Path, dst_rundir: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Validate and prepare snapshot directories.

    Returns:
        tuple: (src_snap_dir, dst_snap_dir)
    """
    src_snap_dir = src_rundir / "snapshots"
    if not src_snap_dir.exists():
        raise FileNotFoundError(f"Source snapshots directory not found: {src_snap_dir}")

    dst_snap_dir = dst_rundir / "snapshots"
    dst_snap_dir.mkdir(parents=True, exist_ok=True)
    return src_snap_dir, dst_snap_dir


def compute_streamfunction_for_file(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    Lx: float,
    Ly: float,
    dealias: float,
    dtype: np.dtype,
    overwrite: bool = False,
) -> None:
    """
    Compute streamfunction for all writes in a single snapshot HDF5 file.

    Strategy:
        1. Open the *source* snapshot file in-place (which may use HDF5 VDS
           with external per-write files) and read vorticity and pressure
           into NumPy arrays.
        2. Compute the streamfunction for each write using the offline LBVP.
        3. Create a new plain HDF5 snapshot file in the destination directory
           containing:
               - /scales group copied from the source file
               - /tasks/vorticity, /tasks/pressure, /tasks/streamfunction
                 as regular (non-virtual) datasets
           with dimension labels and scales matching the source vorticity
           dataset so that plotting tools see a consistent layout.
    """
    if dst_path.exists() and not overwrite:
        print(f"  Skipping existing file (use --overwrite to replace): {dst_path.name}")
        return

    print(f"  Processing {src_path.name}")

    # Read vorticity and pressure from the *source* file (which may use VDS),
    # then write a new plain snapshot file at dst_path.
    with h5py.File(src_path, "r") as f_src:
        # Basic validation on source tasks
        if "tasks/vorticity" not in f_src or "tasks/pressure" not in f_src:
            print(
                f"    Warning: required tasks 'vorticity' and 'pressure' not both present "
                f"in {src_path.name}; skipping."
            )
            return

        vort_src = f_src["tasks/vorticity"]
        pres_src = f_src["tasks/pressure"]

        # Warn explicitly if we are dealing with a virtual dataset, since
        # copying the file can invalidate its external mappings.
        try:
            if getattr(vort_src, "is_virtual", False):
                print("    Note: source vorticity dataset is a virtual dataset (VDS).")
        except Exception:
            pass

        omega_all = np.asarray(vort_src[:], dtype=dtype)
        pressure_all = np.asarray(pres_src[:], dtype=dtype)

        if omega_all.ndim != 3:
            raise ValueError(
                f"Expected vorticity dataset with shape (N_writes, Nx, Ny), "
                f"got {omega_all.shape} in {src_path}"
            )

        if pressure_all.ndim != 3:
            raise ValueError(
                f"Expected pressure dataset with shape (N_writes, Nx, Ny), "
                f"got {pressure_all.shape} in {src_path}"
            )

        if omega_all.shape != pressure_all.shape:
            raise ValueError(
                f"Vorticity and pressure shapes differ in {src_path}: "
                f"{omega_all.shape} vs {pressure_all.shape}"
            )

        n_writes, Nx, Ny = omega_all.shape

        # Handle edge case: empty file with no writes
        if n_writes == 0:
            print(f"    No writes found in vorticity for {src_path.name}; skipping.")
            return

        # Cache dimension labels and scale paths from vorticity so we can
        # replicate them in the destination file.
        dim_labels = []
        dim_scale_paths = []
        for axis in range(vort_src.ndim):
            label = vort_src.dims[axis].label
            dim_labels.append(label)

            # Under this h5py/HDF5 build, vort_src.dims[axis] yields strings
            # naming the scales (e.g., 'sim_time', 'x', 'y') rather than
            # dataset objects. We record those names and look up the
            # corresponding datasets in the copied /scales group in the
            # destination file.
            axis_scales = [str(s) for s in vort_src.dims[axis]]
            dim_scale_paths.append(axis_scales)

        # Allocate streamfunction array
        psi_all = np.empty((n_writes, Nx, Ny), dtype=dtype)

        # Compute streamfunction for each write; LBVP is internally cached in fields.vorticity_to_streamfunction
        for i in range(n_writes):
            omega_grid = np.asarray(omega_all[i], dtype=dtype, order="C")
            psi_grid = fields.vorticity_to_streamfunction(
                omega_grid,
                Lx=Lx,
                Ly=Ly,
                dealias=dealias,
                dtype=dtype,
            )
            if psi_grid.shape != (Nx, Ny):
                raise ValueError(
                    f"psi_grid has shape {psi_grid.shape}, expected {(Nx, Ny)} "
                    f"for write {i} in {src_path}"
                )
            psi_all[i] = psi_grid

        # Destination: create a fresh plain HDF5 snapshot file.
        if dst_path.exists() and overwrite:
            dst_path.unlink()

        with h5py.File(dst_path, "w") as f_dst:
            # Copy the /scales group wholesale so that time and spatial scales
            # (including dimension-scale attributes) are preserved.
            if "scales" in f_src:
                f_src.copy("scales", f_dst)

            tasks_grp = f_dst.create_group("tasks")

            vort_dst = tasks_grp.create_dataset(
                "vorticity",
                data=omega_all,
                compression="gzip",
            )
            pres_dst = tasks_grp.create_dataset(
                "pressure",
                data=pressure_all,
                compression="gzip",
            )
            psi_dset = tasks_grp.create_dataset(
                "streamfunction",
                data=psi_all,
                compression="gzip",
            )

            # Attach dimension labels and scales based on the source vorticity
            # dataset so that dedalus.extras.plot_tools sees a consistent layout.
            for dset in (vort_dst, pres_dst, psi_dset):
                for axis in range(vort_src.ndim):
                    label = dim_labels[axis]
                    if label is not None:
                        dset.dims[axis].label = label

                    for scale_name in dim_scale_paths[axis]:
                        # Time dimension scales live in /scales and are named
                        # like 'sim_time', 'write_number', etc. Spatial scales
                        # use the x/y hash datasets. We look up both cases by
                        # name in the copied /scales group.
                        candidates = []
                        if scale_name in f_dst["scales"]:
                            candidates.append(f_dst["scales"][scale_name])
                        # Hash-based spatial scales won't match the simple
                        # names directly; fall back to a linear search.
                        if not candidates:
                            for ds_name, ds in f_dst["scales"].items():
                                if getattr(ds, "dims", None):
                                    # Match on the dimension-scale NAME attribute
                                    name_attr = ds.attrs.get("NAME")
                                    if isinstance(name_attr, bytes):
                                        name_attr = name_attr.decode("utf-8")
                                    if name_attr == scale_name:
                                        candidates.append(ds)

                        for scale_ds in candidates:
                            try:
                                dset.dims[axis].attach_scale(scale_ds)
                            except Exception:
                                continue

        # Simple summary
        psi_min = float(np.nanmin(psi_all))
        psi_max = float(np.nanmax(psi_all))
        print(
            f"    Wrote {dst_path.name} with {n_writes} writes; "
            f"psi range = [{psi_min:.3e}, {psi_max:.3e}]"
        )


def main() -> None:
    """Main execution function."""
    args = get_args()

    src_rundir = pathlib.Path(args.src_rundir).resolve()
    dst_rundir = pathlib.Path(args.dst_rundir).resolve()

    dtype = _as_dtype(args.dtype)

    print("=" * 70)
    print("NS2D Offline Streamfunction Computation")
    print("=" * 70)
    print(f"Source run directory:      {src_rundir}")
    print(f"Destination run directory: {dst_rundir}")
    print(f"Lx, Ly:                    {args.Lx:.6g}, {args.Ly:.6g}")
    print(f"dealias:                   {args.dealias}")
    print(f"dtype:                     {dtype}")
    print(f"overwrite:                 {args.overwrite}")
    print("=" * 70)

    src_snap_dir, dst_snap_dir = _ensure_dirs(src_rundir, dst_rundir)

    # Collect and sort snapshot files using existing post.io helper
    snap_files = sorted(src_snap_dir.glob("*.h5"))
    if not snap_files:
        print(f"No snapshot files (*.h5) found in {src_snap_dir}")
        return

    snap_files = io.sorted_h5_by_write_number(snap_files)
    print(f"Found {len(snap_files)} snapshot file(s) in {src_snap_dir}")

    for src_file in snap_files:
        dst_file = dst_snap_dir / src_file.name
        try:
            compute_streamfunction_for_file(
                src_file,
                dst_file,
                Lx=args.Lx,
                Ly=args.Ly,
                dealias=args.dealias,
                dtype=dtype,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"  Error processing {src_file.name}: {exc}")

    print("\n" + "=" * 70)
    print("Streamfunction computation complete.")
    print(f"Plot-ready snapshots written to: {dst_snap_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


