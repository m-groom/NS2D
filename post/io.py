"""
Data loading and file I/O utilities for NS2D post-processing.

This module provides functions to read simulation output files:
- Scalar time series from scalars/*.h5
- Spectra and fluxes from spectra.h5
- Snapshot fields from snapshots/*.h5
"""

import re
import pathlib
import h5py
import numpy as np


def sorted_h5_by_write_number(h5_paths):
    """
    Sort Dedalus HDF5 files by their write number.

    Args:
        h5_paths (list): List of Path objects to HDF5 files

    Returns:
        list: Sorted list of Path objects
    """
    def first_write_number(p):
        try:
            with h5py.File(p, "r") as f:
                wn = f["scales/write_number"][0]
            return int(wn)
        except Exception:
            # Fallback: extract number from filename
            m = re.search(r"(\d+)(?=\.h5$)", p.name)
            return int(m.group(1)) if m else 0
    return sorted(h5_paths, key=first_write_number)


def read_scalars(scalars_dir):
    """
    Read and concatenate scalar time series from all HDF5 files.

    Args:
        scalars_dir (str or Path): Directory containing scalar HDF5 files

    Returns:
        tuple: (times, series_dict)
            - times: (N,) array of simulation times
            - series_dict: Dictionary of scalar arrays {name: (N,) array}

    Raises:
        FileNotFoundError: If no HDF5 files found or required tasks missing
    """
    scalars_dir = pathlib.Path(scalars_dir)
    files = sorted_h5_by_write_number(sorted(scalars_dir.glob("*.h5")))

    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {scalars_dir}")

    # Required scalars (must be present in all files)
    required_keys = ["energy", "enstrophy", "palinstrophy"]
    # Optional scalars (included if present in all files)
    optional_keys = ["energy_injection", "drag_loss", "visc_loss",
                     "enstrophy_injection", "enstrophy_drag_loss", "enstrophy_visc_loss"]

    times_all = []
    series_req = {k: [] for k in required_keys}
    series_opt = {k: [] for k in optional_keys}
    opt_present_all = {k: True for k in optional_keys}

    for fp in files:
        with h5py.File(fp, "r") as f:
            t = np.array(f["scales/sim_time"])
            times_all.append(t)

            # Required scalars
            for key in required_keys:
                if key not in f["tasks"]:
                    raise KeyError(f"Required task '{key}' not found in {fp}")
                arr = np.array(f[f"tasks/{key}"]).squeeze()
                series_req[key].append(arr)

            # Optional scalars
            for key in optional_keys:
                if key in f["tasks"]:
                    arr = np.array(f[f"tasks/{key}"]).squeeze()
                    series_opt[key].append(arr)
                else:
                    opt_present_all[key] = False

    # Concatenate time and series
    times = np.concatenate(times_all)

    series_dict = {}
    for k in required_keys:
        series_dict[k] = np.concatenate(series_req[k])
    for k in optional_keys:
        if opt_present_all[k]:
            series_dict[k] = np.concatenate(series_opt[k])

    # Sort by time
    order = np.argsort(times)
    times = times[order]
    for k in series_dict:
        series_dict[k] = series_dict[k][order]

    return times, series_dict


def read_spectra(spectra_path, pattern="k_E_Z"):
    """
    Read energy and enstrophy spectra from spectra.h5 file.

    Args:
        spectra_path (str or Path): Path to spectra.h5
        pattern (str): Dataset name pattern to match (default: "k_E_Z")

    Returns:
        tuple: (times, kbins, Ek_list, Zk_list)
            - times: (T,) array of snapshot times
            - kbins: (M,) array of wavenumber bins
            - Ek_list: List of T energy spectra, each (M,)
            - Zk_list: List of T enstrophy spectra, each (M,)

    Raises:
        FileNotFoundError: If no matching datasets found
    """
    times, Ek_list, Zk_list, kbins = [], [], [], None

    with h5py.File(spectra_path, "r") as f:
        for name in sorted(f.keys()):
            m = re.match(rf"{pattern}_t([0-9]+\.[0-9]+)", name)
            if not m:
                continue

            t = float(m.group(1))
            arr = np.array(f[name])  # shape (M, 3): [k, E(k), Z(k)]

            if kbins is None:
                kbins = arr[:, 0]

            times.append(t)
            Ek_list.append(arr[:, 1])
            Zk_list.append(arr[:, 2])

    if kbins is None:
        raise FileNotFoundError(f"No datasets matching '{pattern}_t*' found in {spectra_path}")

    times = np.array(times)
    order = np.argsort(times)
    times = times[order]
    Ek_list = [Ek_list[i] for i in order]
    Zk_list = [Zk_list[i] for i in order]

    return times, kbins, Ek_list, Zk_list


def read_flux(spectra_path, flux_type="energy"):
    """
    Read spectral flux data from spectra.h5 file.

    Args:
        spectra_path (str or Path): Path to spectra.h5
        flux_type (str): "energy" or "enstrophy"

    Returns:
        tuple: (times, kbins, T_list, Pi_list)
            - times: (T,) array of snapshot times
            - kbins: (M,) array of wavenumber bins
            - T_list: List of T transfer spectra, each (M,)
            - Pi_list: List of T cumulative flux, each (M,)

    Raises:
        FileNotFoundError: If no matching datasets found
        ValueError: If invalid flux_type
    """
    if flux_type == "energy":
        pattern = "flux_T_Pi"
    elif flux_type == "enstrophy":
        pattern = "enstrophy_flux_T_Pi"
    else:
        raise ValueError(f"Invalid flux_type '{flux_type}'. Use 'energy' or 'enstrophy'.")

    times, T_list, Pi_list, kbins = [], [], [], None

    with h5py.File(spectra_path, "r") as f:
        for name in sorted(f.keys()):
            m = re.match(rf"{pattern}_t([0-9]+\.[0-9]+)", name)
            if not m:
                continue

            t = float(m.group(1))
            arr = np.array(f[name])  # shape (M, 3): [k, T(k), Pi(k)]

            if kbins is None:
                kbins = arr[:, 0]

            times.append(t)
            T_list.append(arr[:, 1])
            Pi_list.append(arr[:, 2])

    if kbins is None:
        raise FileNotFoundError(f"No datasets matching '{pattern}_t*' found in {spectra_path}")

    times = np.array(times)
    order = np.argsort(times)
    times = times[order]
    T_list = [T_list[i] for i in order]
    Pi_list = [Pi_list[i] for i in order]

    return times, kbins, T_list, Pi_list


def read_snapshot_field(snapshot_path, task_name, write_index=0):
    """
    Read a single field from a snapshot HDF5 file.

    Args:
        snapshot_path (str or Path): Path to snapshots HDF5 file
        task_name (str): Name of task (e.g., "velocity", "vorticity")
        write_index (int): Write index to read (default: 0)

    Returns:
        tuple: (time, field_data)
            - time: Simulation time at this write
            - field_data: Field array (shape depends on field type)

    Raises:
        KeyError: If task not found
        IndexError: If write_index out of range
    """
    with h5py.File(snapshot_path, "r") as f:
        if f"tasks/{task_name}" not in f:
            available = [k.replace("tasks/", "") for k in f.keys() if k.startswith("tasks/")]
            raise KeyError(f"Task '{task_name}' not found. Available: {available}")

        times = np.array(f["scales/sim_time"])
        if write_index >= len(times):
            raise IndexError(f"Write index {write_index} out of range (max: {len(times)-1})")

        time = times[write_index]
        field = np.array(f[f"tasks/{task_name}"][write_index])

    return time, field


def list_snapshot_tasks(snapshot_path):
    """
    List available tasks in a snapshot file.

    Args:
        snapshot_path (str or Path): Path to snapshots HDF5 file

    Returns:
        list: List of available task names
    """
    with h5py.File(snapshot_path, "r") as f:
        return [k.replace("tasks/", "") for k in f.keys() if k.startswith("tasks/")]


def get_snapshot_info(snapshot_path):
    """
    Get metadata about a snapshot file.

    Args:
        snapshot_path (str or Path): Path to snapshots HDF5 file

    Returns:
        dict: Metadata dictionary with keys:
            - n_writes: Number of writes in file
            - times: Array of simulation times
            - write_numbers: Array of write numbers
            - tasks: List of available tasks
    """
    with h5py.File(snapshot_path, "r") as f:
        times = np.array(f["scales/sim_time"])
        write_numbers = np.array(f["scales/write_number"])
        tasks = list_snapshot_tasks(snapshot_path)

    return {
        "n_writes": len(times),
        "times": times,
        "write_numbers": write_numbers,
        "tasks": tasks
    }
