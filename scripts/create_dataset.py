"""
Create ML training datasets from 2D Navier-Stokes simulation snapshots.

Loads snapshot data from multiple realisations, applies time-range filtering,
and creates train/validation/test splits for training ML models to predict the
next time step.

Each sample is a consecutive snapshot pair (input at time t, target at t + dt).
Channel layout depends on the chosen field representation:
    --field vorticity  →  (omega,)             1 channel
    --field velocity   →  (ux, uy, p)          3 channels

Usage:
    python scripts/create_dataset.py --data_root /path/to/Nx128_Ny128_nu2e-03 --n_realisations 100
    python scripts/create_dataset.py --data_root /path/to/data --field velocity --t_start 100 --t_end 101
    python scripts/create_dataset.py --data_root /path/to/data --n_realisations 50 --start_realisation 10
"""

import argparse
import glob
import re
from pathlib import Path

import h5py
import numpy as np


def natural_sort_key(filename):
    """Extract file number for proper numerical sorting of Dedalus snapshot files."""
    match = re.search(r"_s(\d+)\.h5$", filename)
    if match:
        return int(match.group(1))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create ML datasets from 2D Navier-Stokes simulation snapshots"
    )
    parser.add_argument(
        "--n_realisations",
        type=int,
        default=100,
        help="Number of realisations to load (default: 100)",
    )
    parser.add_argument(
        "--start_realisation",
        type=int,
        default=0,
        help="Starting realisation index (default: 0)",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="vorticity",
        choices=["vorticity", "velocity"],
        help=(
            "Field representation: 'vorticity' stores omega (1 channel), "
            "'velocity' stores ux, uy, p (3 channels). Default: vorticity"
        ),
    )
    parser.add_argument(
        "--t_start",
        type=float,
        default=None,
        help="Start of time window to include (default: include from beginning)",
    )
    parser.add_argument(
        "--t_end",
        type=float,
        default=None,
        help="End of time window to include (default: include to end)",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of realisations for training (default: 0.8)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of realisations for validation (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml_dataset.npz",
        help="Output filename (default: ml_dataset.npz)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling realisations (default: 42)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Directory containing realisation_NNNN folders (default: current directory)",
    )
    parser.add_argument(
        "--snapshot_subdir",
        type=str,
        default="snapshots",
        help="Snapshot subdirectory inside each realisation (default: snapshots)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Output data type (default: float32)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------

def _get_dim_scale(file_handle, task_dataset, axis):
    """Load a 1D coordinate array from a task dataset's HDF5 dimension scale."""
    dim_list = task_dataset.attrs.get("DIMENSION_LIST")
    if dim_list is None or len(dim_list) <= axis or len(dim_list[axis]) == 0:
        return None
    return file_handle[dim_list[axis][0]][:]


def _get_spatial_coordinates(file_handle, task_dataset):
    """
    Extract (x, y) coordinate arrays from a scalar Dedalus task dataset.

    Expects shape (time, x, y) so spatial axes are 1 and 2.
    """
    x = _get_dim_scale(file_handle, task_dataset, axis=1)
    y = _get_dim_scale(file_handle, task_dataset, axis=2)
    return x, y


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _time_slice(times, t_start, t_end):
    """
    Return a contiguous slice into *times* covering [t_start, t_end].

    Uses searchsorted for efficiency (times must be monotonically increasing).
    """
    i0 = 0 if t_start is None else int(np.searchsorted(times, t_start, side="left"))
    i1 = len(times) if t_end is None else int(np.searchsorted(times, t_end, side="right"))
    return slice(i0, i1)


def load_realisation_data(realisation_idx, data_root, snapshot_subdir,
                          field, t_start, t_end, out_dtype):
    """
    Load field data from a single realisation, filtered to [t_start, t_end].

    Returns
    -------
    data : ndarray, shape (n_times, C, Nx, Ny)
        C = 1 for vorticity, C = 3 for velocity.
    times : ndarray, shape (n_times,)
    x_coords, y_coords : ndarray or None
    """
    snapshots_path = (
        Path(data_root) / f"realisation_{realisation_idx:04d}" / snapshot_subdir
    )
    h5_files = sorted(glob.glob(str(snapshots_path / "*.h5")), key=natural_sort_key)

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {snapshots_path}")

    all_data = []
    all_times = []
    x_coords = None
    y_coords = None

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            file_times = f["scales"]["sim_time"][:]

            # Skip entire file if it falls outside the requested window
            if t_start is not None and file_times[-1] < t_start:
                continue
            if t_end is not None and file_times[0] > t_end:
                continue

            sl = _time_slice(file_times, t_start, t_end)
            if sl.start >= sl.stop:
                continue

            tasks = f["tasks"]

            if field == "vorticity":
                omega = tasks["vorticity"][sl].astype(out_dtype, copy=False)
                if omega.ndim == 3:
                    omega = omega[:, np.newaxis, :, :]
                all_data.append(omega)
            else:
                vel = tasks["velocity"][sl].astype(out_dtype, copy=False)
                pres = tasks["pressure"][sl].astype(out_dtype, copy=False)
                if pres.ndim == 3:
                    pres = pres[:, np.newaxis, :, :]
                all_data.append(np.concatenate([vel, pres], axis=1))

            all_times.append(file_times[sl])

            if x_coords is None:
                x_coords, y_coords = _get_spatial_coordinates(f, tasks["pressure"])

    if not all_data:
        raise ValueError(
            f"No snapshots in [{t_start}, {t_end}] for realisation {realisation_idx}"
        )

    data = np.concatenate(all_data, axis=0)
    times = np.concatenate(all_times, axis=0)

    return data, times, x_coords, y_coords


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def create_snapshot_pairs(data, times):
    """
    Create consecutive-snapshot pairs for next-step prediction.

    Returns
    -------
    X : ndarray, shape (n_pairs, C, Nx, Ny)   — input  at time t
    Y : ndarray, shape (n_pairs, C, Nx, Ny)   — target at time t + dt
    time_pairs : ndarray, shape (n_pairs, 2)   — (t_input, t_target)
    """
    X = data[:-1]
    Y = data[1:]
    time_pairs = np.stack([times[:-1], times[1:]], axis=1)
    return X, Y, time_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dtype = np.float32 if args.precision == "float32" else np.float64

    test_frac = 1.0 - args.train_frac - args.val_frac
    if test_frac < 0:
        raise ValueError("train_frac + val_frac must be <= 1.0")

    channels_desc = (
        "omega (1 channel)" if args.field == "vorticity"
        else "ux + uy + p (3 channels)"
    )

    if args.t_start is not None or args.t_end is not None:
        t0 = args.t_start if args.t_start is not None else "start"
        t1 = args.t_end if args.t_end is not None else "end"
        time_desc = f"t in [{t0}, {t1}]"
    else:
        time_desc = "all times"

    print(
        f"Loading {args.n_realisations} realisations starting from "
        f"{args.start_realisation}"
    )
    print(f"Fields: {channels_desc}")
    print(f"Time window: {time_desc}")
    print(f"Precision: {args.precision}")
    print(
        f"Split: {args.train_frac:.0%} train / "
        f"{args.val_frac:.0%} val / {test_frac:.0%} test"
    )
    print()

    # Assign realisations to splits ----------------------------------------
    realisation_indices = list(
        range(args.start_realisation, args.start_realisation + args.n_realisations)
    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(realisation_indices)

    n_train = int(args.n_realisations * args.train_frac)
    n_val = int(args.n_realisations * args.val_frac)

    train_indices = realisation_indices[:n_train]
    val_indices = realisation_indices[n_train:n_train + n_val]
    test_indices = realisation_indices[n_train + n_val:]

    print(f"Train realisations: {len(train_indices)}")
    print(f"Val realisations:   {len(val_indices)}")
    print(f"Test realisations:  {len(test_indices)}")
    print()

    # Load data for each split ----------------------------------------------
    x_coords = None
    y_coords = None

    def load_split(indices, split_name):
        nonlocal x_coords, y_coords
        X_list, Y_list, tp_list, id_list = [], [], [], []

        for i, idx in enumerate(indices):
            print(
                f"  Loading {split_name} realisation {idx:4d} "
                f"({i + 1}/{len(indices)})",
                end="\r",
            )
            try:
                data, times, x_real, y_real = load_realisation_data(
                    idx, args.data_root, args.snapshot_subdir,
                    args.field, args.t_start, args.t_end, out_dtype,
                )
                if x_coords is None and x_real is not None:
                    x_coords = x_real
                    y_coords = y_real

                X, Y, tp = create_snapshot_pairs(data, times)
                X_list.append(X)
                Y_list.append(Y)
                tp_list.append(tp)
                id_list.append(np.full(len(X), idx, dtype=np.int32))
            except (FileNotFoundError, KeyError, ValueError) as err:
                print(f"\n  Warning: Skipping realisation {idx}: {err}")
                continue

        print()

        if X_list:
            return (
                np.concatenate(X_list, axis=0),
                np.concatenate(Y_list, axis=0),
                np.concatenate(tp_list, axis=0),
                np.concatenate(id_list, axis=0),
            )
        return None, None, None, None

    print("Loading training data...")
    X_train, Y_train, times_train, ids_train = load_split(train_indices, "train")

    print("Loading validation data...")
    X_val, Y_val, times_val, ids_val = load_split(val_indices, "val")

    print("Loading test data...")
    X_test, Y_test, times_test, ids_test = load_split(test_indices, "test")

    # Summary ---------------------------------------------------------------
    print()
    print("Dataset shapes:")
    if X_train is not None:
        print(f"  Train: X={X_train.shape}, Y={Y_train.shape}, times={times_train.shape}")
    if X_val is not None:
        print(f"  Val:   X={X_val.shape}, Y={Y_val.shape}, times={times_val.shape}")
    if X_test is not None:
        print(f"  Test:  X={X_test.shape}, Y={Y_test.shape}, times={times_test.shape}")
    if x_coords is not None:
        print(f"  Coordinates: x={x_coords.shape}, y={y_coords.shape}")
    else:
        raise RuntimeError(
            "No valid realisations were loaded; coordinates could not be extracted."
        )

    # Save ------------------------------------------------------------------
    save_dict = {
        "field": np.array(args.field, dtype="S"),
        "train_realisations": np.array(train_indices),
        "val_realisations": np.array(val_indices),
        "test_realisations": np.array(test_indices),
        "x_coords": x_coords,
        "y_coords": y_coords,
    }
    if args.t_start is not None:
        save_dict["t_start"] = np.array(args.t_start)
    if args.t_end is not None:
        save_dict["t_end"] = np.array(args.t_end)

    if X_train is not None:
        save_dict["X_train"] = X_train
        save_dict["Y_train"] = Y_train
        save_dict["times_train"] = times_train
        save_dict["realisation_ids_train"] = ids_train
    if X_val is not None:
        save_dict["X_val"] = X_val
        save_dict["Y_val"] = Y_val
        save_dict["times_val"] = times_val
        save_dict["realisation_ids_val"] = ids_val
    if X_test is not None:
        save_dict["X_test"] = X_test
        save_dict["Y_test"] = Y_test
        save_dict["times_test"] = times_test
        save_dict["realisation_ids_test"] = ids_test

    print()
    print(f"Saving to {args.output}...")
    np.savez_compressed(args.output, **save_dict)

    file_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Done! File size: {file_size:.1f} MB")


if __name__ == "__main__":
    main()
