#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute comprehensive NS2D diagnostics from prediction and ground truth data.

This script reads vorticity and streamfunction arrays from an npz file
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

# Add parent directory to path to import ns2d and post modules
parent_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import spectral functions directly to avoid Dedalus dependency
# We use importlib to bypass the ns2d.__init__.py which imports domain (needs Dedalus)
import importlib.util
spec = importlib.util.spec_from_file_location("spectral_module", parent_dir / "ns2d" / "spectral.py")
spectral_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spectral_module)

compute_spectra = spectral_module.compute_spectra
compute_energy_flux = spectral_module.compute_energy_flux
compute_enstrophy_flux = spectral_module.compute_enstrophy_flux


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


def streamfunction_to_velocity(psi_grid, Lx, Ly):
    """
    Convert streamfunction to velocity in Fourier domain.

    For 2D incompressible flow:
        u = ∂ψ/∂y  =>  û = i k_y ψ̂
        v = -∂ψ/∂x  =>  v̂ = -i k_x ψ̂

    Args:
        psi_grid (ndarray): Streamfunction in physical space (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        tuple: (ux_grid, uy_grid) velocity components in physical space
    """
    Nx, Ny = psi_grid.shape

    # Wavenumber grids for rfft2 layout
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Transform to spectral space
    psi_hat = np.fft.rfft2(psi_grid)

    # Compute velocity: û = ik_y ψ̂,  v̂ = -ik_x ψ̂
    ux_hat = 1j * KY * psi_hat
    uy_hat = -1j * KX * psi_hat

    # Transform back to physical space
    ux_grid = np.fft.irfft2(ux_hat, s=(Nx, Ny))
    uy_grid = np.fft.irfft2(uy_hat, s=(Nx, Ny))

    return ux_grid, uy_grid


def compute_scalar_diagnostics(ux_grid, uy_grid, vorticity_grid, Lx, Ly):
    """
    Compute scalar diagnostics: energy, enstrophy, palinstrophy.

    Args:
        ux_grid (ndarray): x-velocity (Nx, Ny)
        uy_grid (ndarray): y-velocity (Nx, Ny)
        vorticity_grid (ndarray): Vorticity (Nx, Ny)
        Lx (float): Domain length in x
        Ly (float): Domain length in y

    Returns:
        dict: Dictionary with keys 'energy', 'enstrophy', 'palinstrophy'
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

    return {
        'energy': energy,
        'enstrophy': enstrophy,
        'palinstrophy': palinstrophy
    }


def compute_all_diagnostics(vorticity_series, streamfunction_series, times,
                           Lx, Ly, nu, alpha=0.0, compute_flux=False):
    """
    Compute all diagnostics for a time series of fields.

    Args:
        vorticity_series (ndarray): Vorticity fields (T, Nx, Ny)
        streamfunction_series (ndarray): Streamfunction fields (T, Nx, Ny)
        times (ndarray): Time values (T,)
        Lx (float): Domain length in x
        Ly (float): Domain length in y
        nu (float): Kinematic viscosity
        alpha (float): Linear drag coefficient
        compute_flux (bool): Whether to compute spectral fluxes (expensive)

    Returns:
        dict: Dictionary containing all diagnostics
    """
    T, Nx, Ny = vorticity_series.shape

    # Initialize storage
    diagnostics = {
        'times': times,
        'energy': np.zeros(T),
        'enstrophy': np.zeros(T),
        'palinstrophy': np.zeros(T),
        'visc_loss': np.zeros(T),
        'drag_loss': np.zeros(T),
        'Re_lambda': np.zeros(T),
        'spectra_times': [],
        'spectra_kbins': None,
        'spectra_Ek': [],
        'spectra_Zk': [],
    }

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
        psi_grid = streamfunction_series[t_idx]
        omega_grid = vorticity_series[t_idx]

        # Compute velocity from streamfunction
        ux_grid, uy_grid = streamfunction_to_velocity(psi_grid, Lx, Ly)

        # Scalar diagnostics
        scalars = compute_scalar_diagnostics(ux_grid, uy_grid, omega_grid, Lx, Ly)
        diagnostics['energy'][t_idx] = scalars['energy']
        diagnostics['enstrophy'][t_idx] = scalars['enstrophy']
        diagnostics['palinstrophy'][t_idx] = scalars['palinstrophy']

        # Dissipation rates
        diagnostics['visc_loss'][t_idx] = nu * scalars['enstrophy']
        diagnostics['drag_loss'][t_idx] = 2 * alpha * scalars['energy']

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
        diagnostics['spectra_times'].append(times[t_idx])
        diagnostics['spectra_Ek'].append(Ek)
        diagnostics['spectra_Zk'].append(Zk)
        if diagnostics['spectra_kbins'] is None:
            diagnostics['spectra_kbins'] = k_bins

        # Fluxes (if requested)
        if compute_flux:
            # Energy flux
            k_bins_e, T_e, Pi_e = compute_energy_flux(ux_grid, uy_grid, Lx, Ly)
            diagnostics['flux_energy_times'].append(times[t_idx])
            diagnostics['flux_energy_T'].append(T_e)
            diagnostics['flux_energy_Pi'].append(Pi_e)
            if diagnostics['flux_energy_kbins'] is None:
                diagnostics['flux_energy_kbins'] = k_bins_e

            # Enstrophy flux
            k_bins_z, T_z, Pi_z = compute_enstrophy_flux(ux_grid, uy_grid, Lx, Ly)
            diagnostics['flux_enstrophy_times'].append(times[t_idx])
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
        scalars_grp.create_dataset('visc_loss', data=diagnostics['visc_loss'])
        scalars_grp.create_dataset('drag_loss', data=diagnostics['drag_loss'])
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
    print(f"  Visc loss:      {np.mean(diagnostics['visc_loss']):.6e} ± {np.std(diagnostics['visc_loss']):.6e}")
    print(f"  Drag loss:      {np.mean(diagnostics['drag_loss']):.6e} ± {np.std(diagnostics['drag_loss']):.6e}")
    print(f"  Re_lambda:      {np.mean(diagnostics['Re_lambda']):.2f} ± {np.std(diagnostics['Re_lambda']):.2f}")
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
    pred_vorticity = data['pred_vorticity']  # (T, Nx, Ny)
    pred_streamfunction = data['pred_streamfunction']  # (T, Nx, Ny)

    # Try to load ground truth from different possible keys
    truth_keys_vort = ['output_vorticity', 'true_vorticity', 'gt_vorticity']
    truth_keys_psi = ['output_streamfunction', 'true_streamfunction', 'gt_streamfunction']

    true_vorticity = None
    true_streamfunction = None

    for key in truth_keys_vort:
        if key in data:
            true_vorticity = data[key]
            break

    for key in truth_keys_psi:
        if key in data:
            true_streamfunction = data[key]
            break

    if true_vorticity is None or true_streamfunction is None:
        print("Warning: Could not find ground truth data in npz file")
        print(f"Looked for vorticity keys: {truth_keys_vort}")
        print(f"Looked for streamfunction keys: {truth_keys_psi}")
        if not args.skip_truth:
            print("Forcing --skip_truth")
            args.skip_truth = True

    print(f"\nPrediction shape: vorticity={pred_vorticity.shape}, streamfunction={pred_streamfunction.shape}")
    if not args.skip_truth:
        print(f"Ground truth shape: vorticity={true_vorticity.shape}, streamfunction={true_streamfunction.shape}")

    # Create time array
    T = pred_vorticity.shape[0]
    times = args.t_start + np.arange(T) * args.dt

    # Compute diagnostics for predictions
    if not args.skip_pred:
        print("\n" + "=" * 70)
        print("Computing diagnostics for PREDICTIONS")
        print("=" * 70)
        pred_diagnostics = compute_all_diagnostics(
            pred_vorticity, pred_streamfunction, times,
            args.Lx, args.Ly, args.nu, args.alpha, args.compute_flux
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
            true_vorticity, true_streamfunction, times,
            args.Lx, args.Ly, args.nu, args.alpha, args.compute_flux
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
            mse_vorticity = np.mean((pred_vorticity - true_vorticity)**2)
            mse_streamfunction = np.mean((pred_streamfunction - true_streamfunction)**2)

            print(f"\nMSE:")
            print(f"  Vorticity:       {mse_vorticity:.6e}")
            print(f"  Streamfunction:  {mse_streamfunction:.6e}")

            # Normalized MSE (divide by variance)
            norm_mse_vorticity = mse_vorticity / (np.var(true_vorticity) + 1e-16)
            norm_mse_streamfunction = mse_streamfunction / (np.var(true_streamfunction) + 1e-16)

            print(f"\nNormalized MSE (MSE / variance):")
            print(f"  Vorticity:       {norm_mse_vorticity:.6e}")
            print(f"  Streamfunction:  {norm_mse_streamfunction:.6e}")
            print("=" * 70)

    print("\n" + "=" * 70)
    print("Diagnostic computation complete!")
    print(f"Output saved to: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
