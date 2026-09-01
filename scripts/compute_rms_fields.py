#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute RMS of velocity and forcing fields from NS2D snapshot outputs.

This script streams over Dedalus snapshot files and performs MPI-parallel
reductions to avoid loading full fields onto a single rank. When distributed
process files are available (snapshots_s*/snapshots_s*_p*.h5), each rank
processes a subset of process files (spatial chunks) and the global RMS is
assembled via MPI all-reduce. If only merged/virtual files are present, it
falls back to visit_writes to parallelise over time slices.

Example:
    mpiexec -n 4 python compute_rms_fields.py \\
        --rundir examples/Nx512_Ny512_nu1e-04/realisation_0000 \\
        --t_start 50 --t_end 200

For help:
    python compute_rms_fields.py --help
"""

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import h5py
import numpy as np
from mpi4py import MPI
from dedalus.tools import post


COMM = MPI.COMM_WORLD
RANK = COMM.rank
SIZE = COMM.size

# Add parent directory to path to allow `import post` if run from repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


@dataclass
class SnapshotSet:
    """Container for one Dedalus snapshot set."""
    name: str
    virtual_file: pathlib.Path
    process_files: List[pathlib.Path]
    times: np.ndarray


def get_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Compute RMS of velocity and forcing fields from NS2D snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--rundir", nargs="+", type=str, required=True,
                    help="Path(s) to realisation directory/directories containing snapshots/")

    ap.add_argument("--t_start", type=float, default=None,
                    help="Start time for RMS computation (default: first snapshot time)")
    ap.add_argument("--t_end", type=float, default=None,
                    help="End time for RMS computation (default: last snapshot time)")

    ap.add_argument("--snapshot_subdir", type=str, default="snapshots",
                    help="Subdirectory name containing snapshot sets")

    ap.add_argument("--chunk_size", type=int, default=16,
                    help="Number of consecutive writes to read at once (distributed mode)")

    ap.add_argument("--output", type=str, default=None,
                    help="Optional output file to append results (rank 0 only)")

    return ap.parse_args()


def natural_sort_paths(paths: Iterable[pathlib.Path]) -> List[pathlib.Path]:
    """Wrapper around dedalus natural_sort for pathlib paths."""
    return [pathlib.Path(p) for p in post.natural_sort([str(p) for p in paths])]


def find_snapshot_sets(rundir: pathlib.Path, snapshot_subdir: str) -> List[SnapshotSet]:
    """
    Discover snapshot sets (virtual file + distributed process files) in a run directory.

    Returns:
        list[SnapshotSet]
    """
    snap_dir = rundir / snapshot_subdir
    if not snap_dir.exists():
        raise FileNotFoundError(f"No snapshot directory found at {snap_dir}")

    virtual_files = natural_sort_paths(snap_dir.glob("snapshots_s*.h5"))
    sets: List[SnapshotSet] = []

    for vf in virtual_files:
        set_name = vf.stem  # e.g., snapshots_s1
        proc_dir = snap_dir / set_name
        proc_files = []
        if proc_dir.exists():
            proc_files = natural_sort_paths(proc_dir.glob(f"{set_name}_p*.h5"))

        with h5py.File(vf, "r") as f:
            times = np.array(f["scales/sim_time"])

        sets.append(SnapshotSet(name=set_name, virtual_file=vf, process_files=proc_files, times=times))

    return sets


def contiguous_runs(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Convert sorted indices into a list of inclusive contiguous runs."""
    if indices.size == 0:
        return []
    runs = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def process_distributed_set(set_info: SnapshotSet, valid_indices: np.ndarray, chunk_size: int):
    """
    Process a distributed snapshot set by assigning process files across ranks.

    Each rank reads only its assigned process files and accumulates local
    sums/counts for velocity and forcing.
    """
    local_u2 = 0.0
    local_f2 = 0.0
    local_ucount = 0
    local_fcount = 0

    if valid_indices.size == 0:
        return local_u2, local_ucount, local_f2, local_fcount

    assigned_proc_files = set_info.process_files[RANK::SIZE]
    runs = contiguous_runs(valid_indices)

    for proc_fp in assigned_proc_files:
        with h5py.File(proc_fp, "r") as f:
            vel_ds = f["tasks/velocity"]
            force_ds = f["tasks"].get("forcing", None)

            for run_start, run_end in runs:
                # Iterate in smaller chunks to control memory
                current = run_start
                while current <= run_end:
                    block_end = min(current + chunk_size - 1, run_end)
                    vel_block = vel_ds[current:block_end + 1]
                    local_u2 += np.sum(np.square(vel_block))
                    local_ucount += vel_block.size

                    if force_ds is not None:
                        force_block = force_ds[current:block_end + 1]
                        local_f2 += np.sum(np.square(force_block))
                        local_fcount += force_block.size

                    current = block_end + 1

    return local_u2, local_ucount, local_f2, local_fcount


def process_merged_sets(set_paths: List[pathlib.Path], t_start: float, t_end: float):
    """
    Fallback path: use visit_writes on merged/virtual files (loads full fields per write).
    """
    local_u2 = 0.0
    local_f2 = 0.0
    local_ucount = 0
    local_fcount = 0
    local_time_min = np.inf
    local_time_max = -np.inf
    local_snapshots = 0

    def accumulate(set_path, start, count, t_start=None, t_end=None):
        nonlocal local_u2, local_f2, local_ucount, local_fcount, local_time_min, local_time_max, local_snapshots
        with h5py.File(set_path, "r") as f:
            times = np.array(f["scales/sim_time"])[start:start + count]
            vel_ds = f["tasks/velocity"]
            force_ds = f["tasks"].get("forcing", None)
            for i, t in enumerate(times, start=start):
                if (t_start is not None and t < t_start) or (t_end is not None and t > t_end):
                    continue
                vel = vel_ds[i]
                local_u2 += np.sum(np.square(vel))
                local_ucount += vel.size
                if force_ds is not None:
                    f_block = force_ds[i]
                    local_f2 += np.sum(np.square(f_block))
                    local_fcount += f_block.size
                local_time_min = min(local_time_min, t)
                local_time_max = max(local_time_max, t)
                local_snapshots += 1

    post.visit_writes(set_paths, accumulate, t_start=t_start, t_end=t_end)
    return local_u2, local_ucount, local_f2, local_fcount, local_time_min, local_time_max, local_snapshots


def compute_rms_for_rundir(rundir: pathlib.Path, args) -> dict:
    """
    Compute RMS values for a single run directory.
    Returns a dictionary with results (valid on all ranks, root prints).
    """
    sets = find_snapshot_sets(rundir, args.snapshot_subdir)
    if not sets:
        raise FileNotFoundError(f"No snapshot sets found in {rundir}")

    has_distributed = any(s.process_files for s in sets)
    local_u2 = 0.0
    local_f2 = 0.0
    local_ucount = 0
    local_fcount = 0
    local_time_min = np.inf
    local_time_max = -np.inf
    local_snapshots = 0

    for sset in sets:
        times = sset.times
        mask = np.ones_like(times, dtype=bool)
        if args.t_start is not None:
            mask &= times >= args.t_start
        if args.t_end is not None:
            mask &= times <= args.t_end
        valid_indices = np.nonzero(mask)[0]
        if valid_indices.size == 0:
            continue

        if has_distributed and sset.process_files:
            local_time_min = min(local_time_min, float(times[valid_indices].min()))
            local_time_max = max(local_time_max, float(times[valid_indices].max()))
            local_snapshots += int(valid_indices.size)
            u2, ucount, f2, fcount = process_distributed_set(
                sset, valid_indices, args.chunk_size
            )
            local_u2 += u2
            local_ucount += ucount
            local_f2 += f2
            local_fcount += fcount
        else:
            (
                u2,
                ucount,
                f2,
                fcount,
                _tmin,
                _tmax,
                _nsnaps,
            ) = process_merged_sets([sset.virtual_file], args.t_start, args.t_end)
            local_u2 += u2
            local_ucount += ucount
            local_f2 += f2
            local_fcount += fcount
            local_time_min = min(local_time_min, _tmin)
            local_time_max = max(local_time_max, _tmax)
            local_snapshots += _nsnaps

    # Global reductions
    total_u2 = COMM.allreduce(local_u2, op=MPI.SUM)
    total_ucount = COMM.allreduce(local_ucount, op=MPI.SUM)
    total_f2 = COMM.allreduce(local_f2, op=MPI.SUM)
    total_fcount = COMM.allreduce(local_fcount, op=MPI.SUM)
    global_snapshots = COMM.allreduce(local_snapshots, op=MPI.SUM)
    global_tmin = COMM.allreduce(local_time_min, op=MPI.MIN)
    global_tmax = COMM.allreduce(local_time_max, op=MPI.MAX)

    if global_snapshots == 0:
        raise RuntimeError(f"No snapshots in requested time range for {rundir}")

    u_rms = np.sqrt(total_u2 / total_ucount) if total_ucount > 0 else np.nan
    f_rms = np.sqrt(total_f2 / total_fcount) if total_fcount > 0 else None

    return {
        "rundir": str(rundir),
        "u_rms": float(u_rms),
        "f_rms": None if f_rms is None else float(f_rms),
        "snapshots": int(global_snapshots),
        "t_min": None if not np.isfinite(global_tmin) else float(global_tmin),
        "t_max": None if not np.isfinite(global_tmax) else float(global_tmax),
        "mode": "distributed" if has_distributed else "merged_visit_writes",
    }


def write_output_line(output_path: pathlib.Path, result: dict):
    """Append a single-line summary to an output file (rank 0 only)."""
    with open(output_path, "a") as f:
        f.write(
            f"{result['rundir']}, snapshots={result['snapshots']}, "
            f"t=[{result['t_min']}, {result['t_max']}], "
            f"u_rms={result['u_rms']:.6e}, "
            f"f_rms={result['f_rms'] if result['f_rms'] is not None else 'NA'}\n"
        )


def main():
    args = get_args()
    results = []
    for rundir_str in args.rundir:
        rundir = pathlib.Path(rundir_str).resolve()
        try:
            res = compute_rms_for_rundir(rundir, args)
            results.append(res)
        except Exception as e:
            if RANK == 0:
                print(f"[ERROR] {rundir}: {e}")
            # Ensure all ranks see the failure to avoid deadlock
            COMM.Barrier()
            continue
        COMM.Barrier()

    if RANK == 0:
        print("=" * 70)
        print("NS2D Snapshot RMS")
        print("=" * 70)
        for res in results:
            print(f"\nRun directory: {res['rundir']}")
            print(f"Mode: {res['mode']}")
            print(f"Snapshots used: {res['snapshots']}")
            print(f"Time range: [{res['t_min']}, {res['t_max']}]")
            print(f"u_rms: {res['u_rms']:.6e}")
            if res["f_rms"] is None:
                print("f_rms: N/A (forcing not found)")
            else:
                print(f"f_rms: {res['f_rms']:.6e}")

        if args.output:
            out_path = pathlib.Path(args.output)
            for res in results:
                write_output_line(out_path, res)
            print(f"\nResults appended to {out_path}")


if __name__ == "__main__":
    main()
