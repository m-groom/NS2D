"""
Analysis utilities for NS2D simulation output.

This module provides functions for statistical analysis and derived quantities:
- Time averaging and statistics
- Spectral slopes and scaling analysis
- Reynolds number computation from time series
- Cascade direction detection
"""

import numpy as np
from scipy import stats


def time_average(times, series, t_start=None, t_end=None):
    """
    Compute time average of a series over a specified interval.

    Args:
        times (ndarray): Time values (N,)
        series (ndarray): Data series (N,) or (N, M)
        t_start (float or None): Start time (default: first time)
        t_end (float or None): End time (default: last time)

    Returns:
        float or ndarray: Time-averaged value (scalar if 1D input, array if 2D)
    """
    t_start = times[0] if t_start is None else t_start
    t_end = times[-1] if t_end is None else t_end

    mask = (times >= t_start) & (times <= t_end)
    if not np.any(mask):
        raise ValueError(f"No data points in time range [{t_start}, {t_end}]")

    if series.ndim == 1:
        return np.mean(series[mask])
    else:
        return np.mean(series[mask], axis=0)


def time_std(times, series, t_start=None, t_end=None):
    """
    Compute standard deviation of a series over a specified interval.

    Args:
        times (ndarray): Time values (N,)
        series (ndarray): Data series (N,) or (N, M)
        t_start (float or None): Start time
        t_end (float or None): End time

    Returns:
        float or ndarray: Standard deviation
    """
    t_start = times[0] if t_start is None else t_start
    t_end = times[-1] if t_end is None else t_end

    mask = (times >= t_start) & (times <= t_end)
    if not np.any(mask):
        raise ValueError(f"No data points in time range [{t_start}, {t_end}]")

    if series.ndim == 1:
        return np.std(series[mask])
    else:
        return np.std(series[mask], axis=0)


def compute_spectral_slope(kbins, spectrum, k_range=None, log_units=True):
    """
    Compute power-law slope of a spectrum via linear regression in log-log space.

    Args:
        kbins (ndarray): Wavenumber bins (M,)
        spectrum (ndarray): Spectrum values (M,)
        k_range (tuple or None): (k_min, k_max) for fitting range
        log_units (bool): If True, fit log(spectrum) vs log(k); otherwise use linear

    Returns:
        dict: {"slope": slope, "intercept": intercept, "r_squared": R²,
               "k_fit": k_range_used, "spectrum_fit": fitted_values}
    """
    if k_range is None:
        k_range = (kbins[1], kbins[-1])  # Exclude k=0

    mask = (kbins >= k_range[0]) & (kbins <= k_range[1]) & (spectrum > 0)
    k_fit = kbins[mask]
    s_fit = spectrum[mask]

    if len(k_fit) < 3:
        raise ValueError("Insufficient data points in k_range for fitting")

    if log_units:
        log_k = np.log(k_fit)
        log_s = np.log(s_fit)
        slope, intercept, r_value, _, _ = stats.linregress(log_k, log_s)
        spectrum_fit = np.exp(intercept) * k_fit**slope
    else:
        slope, intercept, r_value, _, _ = stats.linregress(k_fit, s_fit)
        spectrum_fit = intercept + slope * k_fit

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "k_fit": k_fit,
        "spectrum_fit": spectrum_fit
    }


def compute_integral_scale(kbins, Ek):
    """
    Compute integral length scale from energy spectrum.

    L_int = π/2 * ∫E(k)/k dk / ∫E(k) dk

    Args:
        kbins (ndarray): Wavenumber bins (M,)
        Ek (ndarray): Energy spectrum (M,)

    Returns:
        float: Integral length scale
    """
    # Exclude k=0
    mask = kbins > 0
    k = kbins[mask]
    E = Ek[mask]

    # Trapezoidal integration
    total_energy = np.trapz(E, k)
    integral_num = np.trapz(E / k, k)

    L_int = (np.pi / 2) * integral_num / total_energy
    return L_int


def compute_taylor_microscale(enstrophy, energy):
    """
    Compute Taylor microscale from energy and enstrophy.

    λ = √(E/Z) where E is kinetic energy and Z is enstrophy.

    Args:
        enstrophy (float): Total enstrophy ∫ω² dx
        energy (float): Total kinetic energy ∫u²/2 dx

    Returns:
        float: Taylor microscale
    """
    if enstrophy <= 0:
        raise ValueError("Enstrophy must be positive")
    if energy <= 0:
        raise ValueError("Energy must be positive")

    return np.sqrt(energy / enstrophy)


def moving_average(array, window_size):
    """
    Compute moving average of a time series.

    Args:
        array (ndarray): Input array (N,)
        window_size (int): Window size for averaging

    Returns:
        ndarray: Smoothed array (N-window_size+1,)
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    return np.convolve(array, np.ones(window_size) / window_size, mode='valid')


def compute_statistics_summary(times, series_dict, t_start=None, t_end=None):
    """
    Compute comprehensive statistics for all scalar series.

    Args:
        times (ndarray): Time values (N,)
        series_dict (dict): Dictionary of scalar arrays
        t_start (float or None): Start time for statistics
        t_end (float or None): End time for statistics

    Returns:
        dict: Statistics dictionary {name: {"mean": ..., "std": ..., "min": ..., "max": ...}}
    """
    t_start = times[0] if t_start is None else t_start
    t_end = times[-1] if t_end is None else t_end

    mask = (times >= t_start) & (times <= t_end)
    stats_dict = {}

    for name, series in series_dict.items():
        data = series[mask]
        stats_dict[name] = {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "median": np.median(data)
        }

    return stats_dict


def compute_dissipation_rate(series_dict, nu, alpha=0):
    """
    Compute total dissipation rate from enstrophy and drag.

    ε_total = ν·Z + 2α·E

    Args:
        series_dict (dict): Dictionary containing 'enstrophy' and 'energy'
        nu (float): Kinematic viscosity
        alpha (float): Drag coefficient (default: 0)

    Returns:
        ndarray: Total dissipation rate time series
    """
    if "enstrophy" not in series_dict or "energy" not in series_dict:
        raise KeyError("Required 'enstrophy' and 'energy' not in series_dict")

    eps_visc = nu * series_dict["enstrophy"]
    eps_drag = 2 * alpha * series_dict["energy"]

    return eps_visc + eps_drag


def compute_taylor_reynolds(energy, enstrophy, nu):
    """
    Compute Taylor-scale Reynolds number.

    Re_λ = u_rms * λ / ν where λ = √(E/Z) and u_rms = √(2E)

    Args:
        energy (float): Total kinetic energy
        enstrophy (float): Total enstrophy
        nu (float): Kinematic viscosity

    Returns:
        float: Taylor-scale Reynolds number
    """
    if energy <= 0 or enstrophy <= 0:
        raise ValueError("Energy and enstrophy must be positive")
    if nu <= 0:
        raise ValueError("Viscosity must be positive")

    lambda_T = np.sqrt(energy / enstrophy)
    u_rms = np.sqrt(2 * energy)
    Re_lambda = u_rms * lambda_T / nu

    return Re_lambda


def compute_integral_reynolds(L_int, energy, nu):
    """
    Compute integral-scale Reynolds number.

    Re_L = u_rms * L_int / ν where u_rms = √(2E)

    Args:
        L_int (float): Integral length scale
        energy (float): Total kinetic energy
        nu (float): Kinematic viscosity

    Returns:
        float: Integral-scale Reynolds number
    """
    if energy <= 0:
        raise ValueError("Energy must be positive")
    if nu <= 0:
        raise ValueError("Viscosity must be positive")
    if L_int <= 0:
        raise ValueError("Integral length scale must be positive")

    u_rms = np.sqrt(2 * energy)
    Re_L = u_rms * L_int / nu

    return Re_L
