#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute comprehensive NS2D diagnostics from prediction and ground truth data.

This script reads velocity (u, v) and pressure arrays from an npz file
(containing predictions and ground truth from FNO or similar models) and
computes all available diagnostics from the post-processing module:
- Scalar time series: energy, enstrophy, palinstrophy
- Spectra: E(k), Z(k)
- Fluxes: energy and enstrophy transfer and cumulative flux
- Reynolds numbers and derived quantities

Usage:
    python compute_diagnostics.py --pred_path path/to/predictions.npz \\
                                   --outdir ./diagnostics \\
                                   --nu 5e-5 \\
                                   --dt 0.2 \\
                                   --Lx 6.283185307179586 \\
                                   --Ly 6.283185307179586

For help:
    python compute_diagnostics.py --help
"""

import argparse
import pathlib
import sys
import numpy as np
import h5py
from tqdm import tqdm

# Add scripts directory to path to import spectral_standalone
scripts_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import standalone spectral functions (NumPy-only, no Dedalus dependency)
from spectral_standalone import compute_spectra, compute_energy_flux, compute_enstrophy_flux


def get_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Compute NS2D diagnostics from prediction and ground truth arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    ap.add_argument("--pred_path", type=str, required=True,
                   help="Path to npz file containing predictions and ground truth")

    # Output
    ap.add_argument("--outdir", type=str, default="./diagnostics",
                   help="Output directory for diagnostic files")

    # Physics parameters
    ap.add_argument("--nu", type=float, default=5e-4,
                   help="Kinematic viscosity (for Reynolds number calculation)")
    ap.add_argument("--alpha", type=float, default=0.023,
                   help="Linear drag coefficient (for dissipation calculation)")

    # Domain parameters
    ap.add_argument("--Lx", type=float, default=2*np.pi,
                   help="Domain length in x")
    ap.add_argument("--Ly", type=float, default=2*np.pi,
                   help="Domain length in y")

    # Time parameters
    ap.add_argument("--dt", type=float, default=0.2,
                   help="Time step between snapshots")
    ap.add_argument("--t_start", type=float, default=0.0,
                   help="Starting time for time array")

    # Computation options
    ap.add_argument("--compute_flux", action="store_true",
                   help="Compute spectral fluxes (computationally expensive)")
    ap.add_argument("--skip_pred", action="store_true",
                   help="Skip computing diagnostics for predictions")
    ap.add_argument("--skip_truth", action="store_true",
                   help="Skip computing diagnostics for ground truth")

    return ap.parse_args()


def velocity_to_vorticity(ux_grid, uy_grid, Lx, Ly):
    """
    Compute vorticity from velocity in Fourier domain.

    For 2D incompressible flow:
        ω = ∂v/∂x - ∂u/∂y  =>  ω̂ = i k_x v̂ - i k_y û

    Args:
        ux_grid (ndarray): x-velocity in physical space (Nx, Ny)
        uy_grid (ndarray): y-velocity in physical space (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        ndarray: Vorticity in physical space (Nx, Ny)
    """
    Nx, Ny = ux_grid.shape

    # Wavenumber grids for rfft2 layout
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Transform to spectral space
    ux_hat = np.fft.rfft2(ux_grid)
    uy_hat = np.fft.rfft2(uy_grid)

    # Compute vorticity: ω̂ = i k_x v̂ - i k_y û
    omega_hat = 1j * KX * uy_hat - 1j * KY * ux_hat

    # Transform back to physical space
    omega_grid = np.fft.irfft2(omega_hat, s=(Nx, Ny))

    return omega_grid


def compute_forcing_curl(fx_grid, fy_grid, Lx, Ly):
    """
    Compute curl of forcing field in Fourier domain.

    For 2D forcing field f = (fx, fy):
        curl_f = ∂fy/∂x - ∂fx/∂y  =>  curl_f̂ = i k_x f̂_y - i k_y f̂_x

    Args:
        fx_grid (ndarray): x-component of forcing in physical space (Nx, Ny)
        fy_grid (ndarray): y-component of forcing in physical space (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        ndarray: Curl of forcing in physical space (Nx, Ny)
    """
    Nx, Ny = fx_grid.shape

    # Wavenumber grids for rfft2 layout
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Transform to spectral space
    fx_hat = np.fft.rfft2(fx_grid)
    fy_hat = np.fft.rfft2(fy_grid)

    # Compute curl: curl_f̂ = i k_x f̂_y - i k_y f̂_x
    curl_hat = 1j * KX * fy_hat - 1j * KY * fx_hat

    # Transform back to physical space
    curl_grid = np.fft.irfft2(curl_hat, s=(Nx, Ny))

    return curl_grid


def compute_scalar_diagnostics(ux_grid, uy_grid, vorticity_grid, Lx, Ly, fx_grid=None, fy_grid=None):
    """
    Compute scalar diagnostics: energy, enstrophy, palinstrophy, and injection terms.

    Args:
        ux_grid (ndarray): x-velocity (Nx, Ny)
        uy_grid (ndarray): y-velocity (Nx, Ny)
        vorticity_grid (ndarray): Vorticity (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y
        fx_grid (ndarray, optional): x-component of forcing (Nx, Ny)
        fy_grid (ndarray, optional): y-component of forcing (Nx, Ny)

    Returns:
        dict: Dictionary with keys 'energy', 'enstrophy', 'palinstrophy',
              and optionally 'energy_injection', 'enstrophy_injection'
    """
    Nx, Ny = ux_grid.shape
    area = Lx * Ly

    # Energy: E = (1/2) ∫ u² dx / Area
    energy = 0.5 * np.sum(ux_grid**2 + uy_grid**2) * area / (Nx * Ny)

    # Enstrophy: Z = ∫ ω² dx / Area
    enstrophy = np.sum(vorticity_grid**2) * area / (Nx * Ny)

    # Palinstrophy: P = ∫ (∇ω)² dx / Area
    # Compute ∇ω in Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    omega_hat = np.fft.rfft2(vorticity_grid)
    domega_dx_hat = 1j * KX * omega_hat
    domega_dy_hat = 1j * KY * omega_hat

    domega_dx = np.fft.irfft2(domega_dx_hat, s=(Nx, Ny))
    domega_dy = np.fft.irfft2(domega_dy_hat, s=(Nx, Ny))

    palinstrophy = np.sum(domega_dx**2 + domega_dy**2) * area / (Nx * Ny)

    result = {
        'energy': energy,
        'enstrophy': enstrophy,
        'palinstrophy': palinstrophy
    }

    # Compute injection terms if forcing is provided
    if fx_grid is not None and fy_grid is not None:
        # Energy injection: ε_i = ∫ u·f dA
        energy_injection = np.sum(ux_grid * fx_grid + uy_grid * fy_grid) * area / (Nx * Ny)

        # Enstrophy injection: Z_i = 2 ∫ ω·curl(f) dA
        curl_f = compute_forcing_curl(fx_grid, fy_grid, Lx, Ly)
        enstrophy_injection = 2.0 * np.sum(vorticity_grid * curl_f) * area / (Nx * Ny)

        result['energy_injection'] = energy_injection
        result['enstrophy_injection'] = enstrophy_injection

    return result


def compute_all_diagnostics(ux_series, uy_series, times,
                           Lx, Ly, nu, alpha=0.0, compute_flux=False,
                           fx_series=None, fy_series=None):
    """
    Compute all diagnostics for a time series of fields.

    Args:
        ux_series (ndarray): x-velocity fields (T, Nx, Ny)
        uy_series (ndarray): y-velocity fields (T, Nx, Ny)
        times (ndarray): Time values (T,)
        Lx (float): Domain length in x
        Ly (float): Domain length in y
        nu (float): Kinematic viscosity
        alpha (float): Linear drag coefficient
        compute_flux (bool): Whether to compute spectral fluxes (expensive)
        fx_series (ndarray, optional): x-forcing fields (T, Nx, Ny)
        fy_series (ndarray, optional): y-forcing fields (T, Nx, Ny)
                  Note: forcing at index i is the forcing applied at timestep i-1,
                  so forcing[i+1] corresponds to the forcing acting on velocity[i]

    Returns:
        dict: Dictionary containing all diagnostics
    """
    T_original, Nx, Ny = ux_series.shape

    # Check if forcing is provided
    has_forcing = (fx_series is not None) and (fy_series is not None)

    # Adjust for forcing time offset: forcing[i] is at timestep previous to velocity[i]
    # So we need forcing[1:] with velocity[:-1] to align them correctly
    # This gives us T-1 timesteps when forcing is available
    if has_forcing:
        T = T_original - 1
        times_out = times[:-1]
        ux_series_use = ux_series[:-1]
        uy_series_use = uy_series[:-1]
        fx_series_use = fx_series[1:]
        fy_series_use = fy_series[1:]
        print(f"Note: Using velocity[:-1] and forcing[1:] to align timesteps (T={T} instead of {T_original})")
    else:
        T = T_original
        times_out = times
        ux_series_use = ux_series
        uy_series_use = uy_series
        fx_series_use = None
        fy_series_use = None

    # Initialize storage
    diagnostics = {
        'times': times_out,
        'energy': np.zeros(T),
        'enstrophy': np.zeros(T),
        'palinstrophy': np.zeros(T),
        'visc_loss': np.zeros(T),
        'drag_loss': np.zeros(T),
        'enstrophy_drag_loss': np.zeros(T),
        'enstrophy_visc_loss': np.zeros(T),
        'Re_lambda': np.zeros(T),
        'spectra_times': [],
        'spectra_kbins': None,
        'spectra_Ek': [],
        'spectra_Zk': [],
    }

    # Add injection and balance terms storage if forcing is available
    if has_forcing:
        diagnostics['energy_injection'] = np.zeros(T)
        diagnostics['enstrophy_injection'] = np.zeros(T)
        diagnostics['energy_balance'] = np.zeros(T)
        diagnostics['enstrophy_balance'] = np.zeros(T)

    if compute_flux:
        diagnostics['flux_energy_times'] = []
        diagnostics['flux_energy_kbins'] = None
        diagnostics['flux_energy_T'] = []
        diagnostics['flux_energy_Pi'] = []
        diagnostics['flux_enstrophy_times'] = []
        diagnostics['flux_enstrophy_kbins'] = None
        diagnostics['flux_enstrophy_T'] = []
        diagnostics['flux_enstrophy_Pi'] = []

    print(f"Computing diagnostics for {T} time steps...")

    for t_idx in tqdm(range(T)):
        # Get fields at this time
        ux_grid = ux_series_use[t_idx]
        uy_grid = uy_series_use[t_idx]

        # Get forcing if available (already aligned)
        fx_grid = fx_series_use[t_idx] if has_forcing else None
        fy_grid = fy_series_use[t_idx] if has_forcing else None

        # Compute vorticity from velocity
        omega_grid = velocity_to_vorticity(ux_grid, uy_grid, Lx, Ly)

        # Scalar diagnostics (including injection terms if forcing is provided)
        scalars = compute_scalar_diagnostics(ux_grid, uy_grid, omega_grid, Lx, Ly,
                                            fx_grid=fx_grid, fy_grid=fy_grid)
        diagnostics['energy'][t_idx] = scalars['energy']
        diagnostics['enstrophy'][t_idx] = scalars['enstrophy']
        diagnostics['palinstrophy'][t_idx] = scalars['palinstrophy']

        # Energy budget terms
        diagnostics['visc_loss'][t_idx] = nu * scalars['enstrophy']
        diagnostics['drag_loss'][t_idx] = 2 * alpha * scalars['energy']

        # Enstrophy budget terms
        diagnostics['enstrophy_drag_loss'][t_idx] = 2 * alpha * scalars['enstrophy']
        diagnostics['enstrophy_visc_loss'][t_idx] = 2 * nu * scalars['palinstrophy']

        # Store injection terms and compute balances if available
        if has_forcing:
            diagnostics['energy_injection'][t_idx] = scalars['energy_injection']
            diagnostics['enstrophy_injection'][t_idx] = scalars['enstrophy_injection']

            # Energy balance: injection - drag_loss - visc_loss
            diagnostics['energy_balance'][t_idx] = (
                scalars['energy_injection']
                - diagnostics['drag_loss'][t_idx]
                - diagnostics['visc_loss'][t_idx]
            )

            # Enstrophy balance: injection - drag_loss - visc_loss
            diagnostics['enstrophy_balance'][t_idx] = (
                scalars['enstrophy_injection']
                - diagnostics['enstrophy_drag_loss'][t_idx]
                - diagnostics['enstrophy_visc_loss'][t_idx]
            )

        # Taylor Reynolds number: Re_λ = u_rms * λ / ν
        # where λ = √(E/Z) and u_rms = √(2E)
        if scalars['enstrophy'] > 1e-16:
            lambda_T = np.sqrt(scalars['energy'] / scalars['enstrophy'])
            u_rms = np.sqrt(2 * scalars['energy'])
            diagnostics['Re_lambda'][t_idx] = u_rms * lambda_T / nu
        else:
            diagnostics['Re_lambda'][t_idx] = 0.0

        # Spectra (every time step)
        k_bins, Ek, Zk = compute_spectra(ux_grid, uy_grid, Lx, Ly)
        diagnostics['spectra_times'].append(times_out[t_idx])
        diagnostics['spectra_Ek'].append(Ek)
        diagnostics['spectra_Zk'].append(Zk)
        if diagnostics['spectra_kbins'] is None:
            diagnostics['spectra_kbins'] = k_bins

        # Fluxes (if requested)
        if compute_flux:
            # Energy flux
            k_bins_e, T_e, Pi_e = compute_energy_flux(ux_grid, uy_grid, Lx, Ly)
            diagnostics['flux_energy_times'].append(times_out[t_idx])
            diagnostics['flux_energy_T'].append(T_e)
            diagnostics['flux_energy_Pi'].append(Pi_e)
            if diagnostics['flux_energy_kbins'] is None:
                diagnostics['flux_energy_kbins'] = k_bins_e

            # Enstrophy flux
            k_bins_z, T_z, Pi_z = compute_enstrophy_flux(ux_grid, uy_grid, Lx, Ly)
            diagnostics['flux_enstrophy_times'].append(times_out[t_idx])
            diagnostics['flux_enstrophy_T'].append(T_z)
            diagnostics['flux_enstrophy_Pi'].append(Pi_z)
            if diagnostics['flux_enstrophy_kbins'] is None:
                diagnostics['flux_enstrophy_kbins'] = k_bins_z

    # Convert lists to arrays
    diagnostics['spectra_times'] = np.array(diagnostics['spectra_times'])
    diagnostics['spectra_Ek'] = np.array(diagnostics['spectra_Ek'])
    diagnostics['spectra_Zk'] = np.array(diagnostics['spectra_Zk'])

    if compute_flux:
        diagnostics['flux_energy_times'] = np.array(diagnostics['flux_energy_times'])
        diagnostics['flux_energy_T'] = np.array(diagnostics['flux_energy_T'])
        diagnostics['flux_energy_Pi'] = np.array(diagnostics['flux_energy_Pi'])
        diagnostics['flux_enstrophy_times'] = np.array(diagnostics['flux_enstrophy_times'])
        diagnostics['flux_enstrophy_T'] = np.array(diagnostics['flux_enstrophy_T'])
        diagnostics['flux_enstrophy_Pi'] = np.array(diagnostics['flux_enstrophy_Pi'])

    return diagnostics


def save_diagnostics_hdf5(diagnostics, output_path, label=""):
    """
    Save diagnostics to HDF5 file in a format compatible with post/io.py.

    Args:
        diagnostics (dict): Dictionary of diagnostic arrays
        output_path (Path): Output file path
        label (str): Label for this dataset (e.g., "pred" or "truth")
    """
    with h5py.File(output_path, 'w') as f:
        # Scalars group
        scalars_grp = f.create_group('scalars')
        scalars_grp.create_dataset('sim_time', data=diagnostics['times'])
        scalars_grp.create_dataset('energy', data=diagnostics['energy'])
        scalars_grp.create_dataset('enstrophy', data=diagnostics['enstrophy'])
        scalars_grp.create_dataset('palinstrophy', data=diagnostics['palinstrophy'])

        # Energy budget terms
        scalars_grp.create_dataset('visc_loss', data=diagnostics['visc_loss'])
        scalars_grp.create_dataset('drag_loss', data=diagnostics['drag_loss'])
        if 'energy_injection' in diagnostics:
            scalars_grp.create_dataset('energy_injection', data=diagnostics['energy_injection'])
        if 'energy_balance' in diagnostics:
            scalars_grp.create_dataset('energy_balance', data=diagnostics['energy_balance'])

        # Enstrophy budget terms
        scalars_grp.create_dataset('enstrophy_drag_loss', data=diagnostics['enstrophy_drag_loss'])
        scalars_grp.create_dataset('enstrophy_visc_loss', data=diagnostics['enstrophy_visc_loss'])
        if 'enstrophy_injection' in diagnostics:
            scalars_grp.create_dataset('enstrophy_injection', data=diagnostics['enstrophy_injection'])
        if 'enstrophy_balance' in diagnostics:
            scalars_grp.create_dataset('enstrophy_balance', data=diagnostics['enstrophy_balance'])

        # Other derived quantities
        scalars_grp.create_dataset('Re_lambda', data=diagnostics['Re_lambda'])

        # Spectra
        spectra_grp = f.create_group('spectra')
        for i, t in enumerate(diagnostics['spectra_times']):
            # Format: k_E_Z_t{time}
            ds_name = f"k_E_Z_t{t:.6f}"
            # Stack as (M, 3): [k, E(k), Z(k)]
            arr = np.stack([
                diagnostics['spectra_kbins'],
                diagnostics['spectra_Ek'][i],
                diagnostics['spectra_Zk'][i]
            ], axis=1)
            spectra_grp.create_dataset(ds_name, data=arr)

        # Fluxes (if present)
        if 'flux_energy_times' in diagnostics and len(diagnostics['flux_energy_times']) > 0:
            flux_grp = f.create_group('flux')

            # Energy flux
            for i, t in enumerate(diagnostics['flux_energy_times']):
                ds_name = f"flux_T_Pi_t{t:.6f}"
                arr = np.stack([
                    diagnostics['flux_energy_kbins'],
                    diagnostics['flux_energy_T'][i],
                    diagnostics['flux_energy_Pi'][i]
                ], axis=1)
                flux_grp.create_dataset(ds_name, data=arr)

            # Enstrophy flux
            for i, t in enumerate(diagnostics['flux_enstrophy_times']):
                ds_name = f"enstrophy_flux_T_Pi_t{t:.6f}"
                arr = np.stack([
                    diagnostics['flux_enstrophy_kbins'],
                    diagnostics['flux_enstrophy_T'][i],
                    diagnostics['flux_enstrophy_Pi'][i]
                ], axis=1)
                flux_grp.create_dataset(ds_name, data=arr)

        # Metadata
        f.attrs['label'] = label
        f.attrs['Lx'] = diagnostics.get('Lx', 2*np.pi)
        f.attrs['Ly'] = diagnostics.get('Ly', 2*np.pi)
        f.attrs['nu'] = diagnostics.get('nu', 0.0)
        f.attrs['alpha'] = diagnostics.get('alpha', 0.0)

    print(f"Saved diagnostics to {output_path}")


def print_statistics(diagnostics, label=""):
    """Print summary statistics for diagnostics."""
    print(f"\n{'='*70}")
    print(f"Statistics for {label}")
    print(f"{'='*70}")
    print(f"Time range: [{diagnostics['times'][0]:.3f}, {diagnostics['times'][-1]:.3f}]")
    print(f"Number of snapshots: {len(diagnostics['times'])}")

    print(f"\nScalar diagnostics (mean ± std):")
    print(f"  Energy:         {np.mean(diagnostics['energy']):.6e} ± {np.std(diagnostics['energy']):.6e}")
    print(f"  Enstrophy:      {np.mean(diagnostics['enstrophy']):.6e} ± {np.std(diagnostics['enstrophy']):.6e}")
    print(f"  Palinstrophy:   {np.mean(diagnostics['palinstrophy']):.6e} ± {np.std(diagnostics['palinstrophy']):.6e}")
    print(f"  Re_lambda:      {np.mean(diagnostics['Re_lambda']):.2f} ± {np.std(diagnostics['Re_lambda']):.2f}")

    print(f"\nEnergy budget (mean ± std):")
    if 'energy_injection' in diagnostics:
        print(f"  Injection:      {np.mean(diagnostics['energy_injection']):.6e} ± {np.std(diagnostics['energy_injection']):.6e}")
    print(f"  Visc loss:      {np.mean(diagnostics['visc_loss']):.6e} ± {np.std(diagnostics['visc_loss']):.6e}")
    print(f"  Drag loss:      {np.mean(diagnostics['drag_loss']):.6e} ± {np.std(diagnostics['drag_loss']):.6e}")
    if 'energy_balance' in diagnostics:
        print(f"  Balance:        {np.mean(diagnostics['energy_balance']):.6e} ± {np.std(diagnostics['energy_balance']):.6e}")

    print(f"\nEnstrophy budget (mean ± std):")
    if 'enstrophy_injection' in diagnostics:
        print(f"  Injection:      {np.mean(diagnostics['enstrophy_injection']):.6e} ± {np.std(diagnostics['enstrophy_injection']):.6e}")
    print(f"  Drag loss:      {np.mean(diagnostics['enstrophy_drag_loss']):.6e} ± {np.std(diagnostics['enstrophy_drag_loss']):.6e}")
    print(f"  Visc loss:      {np.mean(diagnostics['enstrophy_visc_loss']):.6e} ± {np.std(diagnostics['enstrophy_visc_loss']):.6e}")
    if 'enstrophy_balance' in diagnostics:
        print(f"  Balance:        {np.mean(diagnostics['enstrophy_balance']):.6e} ± {np.std(diagnostics['enstrophy_balance']):.6e}")

    print(f"{'='*70}\n")


def main():
    """Main execution function."""
    args = get_args()

    pred_path = pathlib.Path(args.pred_path)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NS2D Diagnostic Computation")
    print("=" * 70)
    print(f"Input file: {pred_path}")
    print(f"Output directory: {outdir}")
    print(f"Domain: Lx={args.Lx:.6f}, Ly={args.Ly:.6f}")
    print(f"Physics: nu={args.nu:.6e}, alpha={args.alpha:.6e}")
    print(f"Time step: dt={args.dt:.6f}, t_start={args.t_start:.6f}")
    print(f"Compute fluxes: {args.compute_flux}")
    print("=" * 70)

    # Load data
    print("\nLoading data from npz file...")
    data = np.load(pred_path, allow_pickle=True)
    print(f"Available keys: {list(data.keys())}")

    # Extract arrays
    pred_u = data['pred_velocity_x']  # (T, Nx, Ny)
    pred_v = data['pred_velocity_y']  # (T, Nx, Ny)
    pred_pressure = data['pred_pressure']  # (T, Nx, Ny)

    # Try to load ground truth from different possible keys
    truth_keys_u = ['output_velocity_x', 'true_u', 'gt_u']
    truth_keys_v = ['output_velocity_y', 'true_v', 'gt_v']
    truth_keys_p = ['output_pressure', 'true_pressure', 'gt_pressure', 'output_p', 'true_p', 'gt_p']

    true_u = None
    true_v = None
    true_pressure = None

    for key in truth_keys_u:
        if key in data:
            true_u = data[key]
            break

    for key in truth_keys_v:
        if key in data:
            true_v = data[key]
            break

    for key in truth_keys_p:
        if key in data:
            true_pressure = data[key]
            break

    if true_u is None or true_v is None or true_pressure is None:
        print("Warning: Could not find ground truth data in npz file")
        print(f"Looked for u keys: {truth_keys_u}")
        print(f"Looked for v keys: {truth_keys_v}")
        print(f"Looked for pressure keys: {truth_keys_p}")
        if not args.skip_truth:
            print("Forcing --skip_truth")
            args.skip_truth = True

    # Try to load forcing data
    forcing_keys_x = ['input_forcing_x', 'forcing_x', 'fx']
    forcing_keys_y = ['input_forcing_y', 'forcing_y', 'fy']

    forcing_x = None
    forcing_y = None

    for key in forcing_keys_x:
        if key in data:
            forcing_x = data[key]
            break

    for key in forcing_keys_y:
        if key in data:
            forcing_y = data[key]
            break

    has_forcing = (forcing_x is not None) and (forcing_y is not None)

    if has_forcing:
        print(f"\nForcing data found: fx={forcing_x.shape}, fy={forcing_y.shape}")
    else:
        print("\nWarning: Could not find forcing data in npz file")
        print(f"Looked for fx keys: {forcing_keys_x}")
        print(f"Looked for fy keys: {forcing_keys_y}")
        print("Energy and enstrophy injection terms will not be computed")

    print(f"\nPrediction shape: u={pred_u.shape}, v={pred_v.shape}, pressure={pred_pressure.shape}")
    if not args.skip_truth:
        print(f"Ground truth shape: u={true_u.shape}, v={true_v.shape}, pressure={true_pressure.shape}")

    # Create time array
    T = pred_u.shape[0]
    times = args.t_start + np.arange(T) * args.dt

    # Compute diagnostics for predictions
    if not args.skip_pred:
        print("\n" + "=" * 70)
        print("Computing diagnostics for PREDICTIONS")
        print("=" * 70)
        pred_diagnostics = compute_all_diagnostics(
            pred_u, pred_v, times,
            args.Lx, args.Ly, args.nu, args.alpha, args.compute_flux,
            fx_series=forcing_x if has_forcing else None,
            fy_series=forcing_y if has_forcing else None
        )
        pred_diagnostics['Lx'] = args.Lx
        pred_diagnostics['Ly'] = args.Ly
        pred_diagnostics['nu'] = args.nu
        pred_diagnostics['alpha'] = args.alpha

        # Save to HDF5
        pred_output_path = outdir / "diagnostics_predictions.h5"
        save_diagnostics_hdf5(pred_diagnostics, pred_output_path, label="predictions")

        # Print statistics
        print_statistics(pred_diagnostics, label="PREDICTIONS")

    # Compute diagnostics for ground truth
    if not args.skip_truth:
        print("\n" + "=" * 70)
        print("Computing diagnostics for GROUND TRUTH")
        print("=" * 70)
        truth_diagnostics = compute_all_diagnostics(
            true_u, true_v, times,
            args.Lx, args.Ly, args.nu, args.alpha, args.compute_flux,
            fx_series=forcing_x if has_forcing else None,
            fy_series=forcing_y if has_forcing else None
        )
        truth_diagnostics['Lx'] = args.Lx
        truth_diagnostics['Ly'] = args.Ly
        truth_diagnostics['nu'] = args.nu
        truth_diagnostics['alpha'] = args.alpha

        # Save to HDF5
        truth_output_path = outdir / "diagnostics_groundtruth.h5"
        save_diagnostics_hdf5(truth_diagnostics, truth_output_path, label="ground_truth")

        # Print statistics
        print_statistics(truth_diagnostics, label="GROUND TRUTH")

        # Compute comparison metrics
        if not args.skip_pred:
            print("\n" + "=" * 70)
            print("COMPARISON METRICS (Prediction vs Ground Truth)")
            print("=" * 70)

            # Relative errors for scalars
            rel_err_energy = np.abs(pred_diagnostics['energy'] - truth_diagnostics['energy']) / (np.abs(truth_diagnostics['energy']) + 1e-16)
            rel_err_enstrophy = np.abs(pred_diagnostics['enstrophy'] - truth_diagnostics['enstrophy']) / (np.abs(truth_diagnostics['enstrophy']) + 1e-16)

            print(f"Relative error (mean ± std):")
            print(f"  Energy:     {np.mean(rel_err_energy):.4%} ± {np.std(rel_err_energy):.4%}")
            print(f"  Enstrophy:  {np.mean(rel_err_enstrophy):.4%} ± {np.std(rel_err_enstrophy):.4%}")

            # MSE for fields
            mse_u = np.mean((pred_u - true_u)**2)
            mse_v = np.mean((pred_v - true_v)**2)
            mse_pressure = np.mean((pred_pressure - true_pressure)**2)

            print(f"\nMSE:")
            print(f"  u:        {mse_u:.6e}")
            print(f"  v:        {mse_v:.6e}")
            print(f"  Pressure: {mse_pressure:.6e}")

            # Normalized MSE (divide by variance)
            norm_mse_u = mse_u / (np.var(true_u) + 1e-16)
            norm_mse_v = mse_v / (np.var(true_v) + 1e-16)
            norm_mse_pressure = mse_pressure / (np.var(true_pressure) + 1e-16)

            print(f"\nNormalised MSE (MSE / variance):")
            print(f"  u:        {norm_mse_u:.6e}")
            print(f"  v:        {norm_mse_v:.6e}")
            print(f"  Pressure: {norm_mse_pressure:.6e}")
            print("=" * 70)

    print("\n" + "=" * 70)
    print("Diagnostic computation complete!")
    print(f"Output saved to: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
