# NS2D: 2D Incompressible Navier-Stokes Solver

Forced 2D incompressible Navier-Stokes (velocity-pressure formulation) on a torus with periodic boundary conditions using the [Dedalus](https://dedalus-project.org/) spectral solver framework.

## Features

### Simulation
- Pseudo-spectral solver using Fourier basis
- Distributed memory parallelisation via MPI
- Band-limited white noise or Ornstein-Uhlenbeck forcing with optional constant-power rescaling
- Diagnostics:
  - Energy/enstrophy spectra and spectral fluxes
  - Time series of integrated quantities (energy, enstrophy, palinstrophy, energy budget terms)
  - Field snapshots (velocity, vorticity, pressure, streamfunction)
- Multiple independent realisations with reproducible forcing

### Post-Processing
- Reusable modules for data loading, visualisation, and analysis

## Quick Start

### Prerequisites

- Python 3.8+
- MPI implementation (e.g., OpenMPI, MPICH)
- Dedalus v3
- Standard scientific Python stack (NumPy, h5py, mpi4py)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/m-groom/NS2D
cd NS2D
```

2. Install dependencies (see [Dependencies](#dependencies) section below):
```bash
pip install -r requirements.txt
```

3. Run a simple test:
```bash
python main.py --t_end 10
```

### Basic Usage

**Single-core simulation:**
```bash
python main.py --t_end 200
```

**MPI parallel simulation (8 processes):**
```bash
mpiexec -n 8 python main.py --t_end 200
```

**Custom forcing parameters:**
```bash
python main.py \
    --forcing stochastic \
    --stoch_type ou \
    --kmin 25 --kmax 35 \
    --eps_target 0.01 \
    --tau_ou 0.3
```

**Run multiple realisations:**
```bash
mpiexec -n 16 python main.py \
    --n_realisations 5 \
    --seed 42 \
    --outdir my_ensemble
```

**Get help:**
```bash
python main.py --help
```

## Project Structure

```
NS2D/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── main.py                      # Main entry point
│
├── examples/                    # Example run scripts
│   └── run_simulation.sh
│
├── ns2d/                        # Main simulation package
│   ├── __init__.py              # Package initialisation
│   ├── config.py                # Configuration and argument parsing
│   ├── domain.py                # Domain setup and initial conditions
│   ├── forcing.py               # Stochastic forcing implementations
│   ├── spectral.py              # Spectral analysis (spectra, fluxes)
│   ├── solver.py                # Main solver and time integration
│   └── utils.py                 # MPI utilities and diagnostics
│
├── post/                        # Post-processing toolkit
│   ├── __init__.py              # Package initialisation
│   ├── io.py                    # Data loading utilities
│   ├── visualisation.py         # Plotting functions
│   └── analysis.py              # Statistical analysis
│
└── scripts/                     # Ready-to-use analysis scripts
    ├── plot_output.py           # Auto-generate all plots
    ├── compute_statistics.py    # Compute statistics
    └── animate_snapshots.sh     # Create animations from snapshots
```

### Module Overview

**Simulation (ns2d/):**
- **config.py**: Command-line argument parsing and validation
- **domain.py**: Domain construction, wavenumber grids, initial conditions
- **forcing.py**: Band-limited stochastic forcing (white noise, OU process) with optional constant-power rescaling
- **spectral.py**: Computation of energy/enstrophy spectra and spectral fluxes
- **solver.py**: Dedalus problem setup, time integration, output management
- **utils.py**: MPI field gathering, global diagnostics, Reynolds number computation

**Post-Processing (post/):**
- **io.py**: Load simulation output (scalars, spectra, fluxes, snapshots)
- **visualisation.py**: Create plots
- **analysis.py**: Compute statistics, spectral slopes, derived quantities

## Physics

The code solves the 2D incompressible Navier-Stokes equations (plus a linear drag term) on a periodic domain [0, Lx] × [0, Ly]:

```
∂u/∂t + (u·∇)u + ∇p = ν∇²u - αu + f
∇·u = 0
```

where:
- **u** = (u, v): velocity field
- **p**: pressure
- **ν**: kinematic viscosity
- **α**: linear (Ekman) friction coefficient (optional, for large-scale damping)
- **f**: external forcing

### Forcing

Two stochastic forcing types are supported:

1. **White-in-time**: δ-correlated random forcing
2. **Ornstein-Uhlenbeck (OU)**: Exponentially correlated forcing with correlation time τ

Both forcing types:
- Are band-limited to wavenumber shell [kmin, kmax]
- Enforce incompressibility (∇·f = 0)
- Support two power modes:
  - Default: constant-power rescaling to maintain target energy injection rate ε = ⟨u·f⟩
  - `--power_mode sigma`: no rescaling; uses `f_sigma` directly as the forcing amplitude

## Configuration

All simulation parameters are controlled via command-line arguments. Key options include:

### Domain & Resolution
- `--Nx`, `--Ny`: Grid resolution (default: 256 × 256)
- `--Lx`, `--Ly`: Domain size (default: 2π × 2π)
- `--dealias`: Dealiasing factor (default: 1.5)

### Physics
- `--nu`: Kinematic viscosity (default: 5×10⁻⁴)
- `--alpha`: Linear friction coefficient (default: 0.025, use 0 to disable)

### Forcing
- `--forcing`: `stochastic` or `none` (default: stochastic)
- `--stoch_type`: `white` or `ou` (default: ou)
- `--kmin`, `--kmax`: Forcing wavenumber band (default: 8.0–12.0)
- `--power_mode`: `constant` (default) or `sigma`. `constant` rescales to hit `eps_target`; `sigma` uses `f_sigma` directly.
- `--eps_target`: Target energy injection rate (used when `--power_mode constant`, default: 0.001)
- `--f_sigma`: Base forcing amplitude (used directly when `--power_mode sigma`, default: 0.02)
- `--tau_ou`: OU correlation time (default: 0.3)
- `--eps_floor`: Floor value to prevent division by tiny values (default: 1.0e-12)
- `--eps_smooth`: Exponential smoothing for rescaling (default: 0.0)
- `--eps_clip`: Maximum rescale factor per step (default: 10.0)

### Time Integration
- `--t_end`: Total simulation time (default: 200.0)
- `--cfl_safety`: CFL safety factor (default: 0.4)
- `--cfl_threshold`: CFL threshold to prevent tiny timestep adjustments (default: 0.1)
- `--cfl_cadence`: Number of iterations between CFL updates (default: 10)
- `--cfl_max_dt`, `--cfl_min_dt`: Timestep bounds (default: 0.1, 10⁻⁸)

### Output
- `--outdir`: Output directory (default: `snapshots`)
- `--tag`: Optional tag for output directory name
- `--snap_dt`: Snapshot output interval (default: 1.0)
- `--scalars_dt`: Scalar time series interval (default: 0.05)
- `--spectra_dt`: Spectra/flux output interval (default: 0.25)

### Ensemble
- `--n_realisations`: Number of independent realisations (default: 1)
- `--seed`: Base random seed (default: 42)

### MPI
- `--procs_x`, `--procs_y`: 2D process mesh dimensions (default: 0 = auto/1D)

### Miscellaneous
- `--precision`: `float64` or `float32` (default: float64)

Run `python main.py --help` for the complete list.

## Output

Output is organised by run parameters and realisation, for example:

```
snapshots/
└── Nx1024_Ny1024_nu5e-05/
    ├── realisation_0000/
    │   ├── snapshots/          # HDF5 snapshots of fields
    │   ├── scalars/            # HDF5 time series (energy, enstrophy, etc.)
    │   └── spectra.h5          # Spectra and fluxes at multiple times
    ├── realisation_0001/
    └── ...
```

### Output Files

1. **snapshots/**: Field snapshots (velocity, pressure, vorticity, streamfunction)
   - Written every `snap_dt` time units
   - Dedalus HDF5 format (use `h5py` or Dedalus post-processing tools)

2. **scalars/**: Time series of integrated quantities
   - `energy`: Total kinetic energy E = ½∫|u|² dx
   - `enstrophy`: Total enstrophy Z = ∫|ω|² dx
   - `palinstrophy`: Palinstrophy P = ∫|∇ω|² dx
   - `inj`: Energy injection rate εᵢ = ∫u·f dx
   - `drag_loss`: Drag dissipation εₐ = α∫|u|² dx
   - `visc_loss`: Viscous dissipation εᵥ = ν∫|ω|² dx
   - Written every `scalars_dt` time units

3. **spectra.h5**: Spectral diagnostics
   - `k_E_Z_t{time}`: Energy spectrum E(k), enstrophy spectrum Z(k)
   - `flux_T_Pi_t{time}`: Energy transfer T(k) and flux Π(k)
   - `enstrophy_flux_T_Pi_t{time}`: Enstrophy transfer and flux
   - Written every `spectra_dt` time units

## Post-Processing

NS2D includes a comprehensive post-processing toolkit for analysing and visualising simulation output.

### Quick Start: Visualise Output

Generate all standard plots automatically:

```bash
python scripts/plot_output.py --rundir snapshots/Nx1024_Ny1024_nu5e-05/realisation_0000 --outdir ./figures
```

This creates:
- Time series plots (energy, enstrophy, palinstrophy, budget terms)
- Energy and enstrophy spectra
- Energy and enstrophy fluxes (transfer and cascade)
- Field snapshots (vorticity, pressure, streamfunction)

### Compute Statistics

```bash
python scripts/compute_statistics.py --rundir snapshots/Nx1024_Ny1024_nu5e-05/realisation_0000 --t_start 100 --t_end 500 --nu 5e-5 --k_range 20 100
```

This computes:
- Time-averaged statistics (mean, std, min, max) between `t_start` and `t_end`
- Energy balance verification
- Spectral slopes and power-law fits over specified `k_range`
- Integral and Taylor microscales

### Post-Processing Options

**plot_output.py options:**
- `--rundir PATH`: Path to realisation directory (required)
- `--outdir PATH`: Output directory for figures (default: ./figures)
- `--dpi DPI`: Figure resolution (default: 300)
- `--no_scalars`, `--no_spectra`, `--no_flux`, `--no_snapshots`: Skip specific plots
- `--spectra_max_curves N`: Number of time curves to overlay (default: 6)
- `--spectra_loglog`: Use log-log axes for spectra plots

**compute_statistics.py options:**
- `--rundir PATH`: Path to realisation directory (required)
- `--t_start`, `--t_end`: Time range for statistics (default: full time range)
- `--k_range K_MIN K_MAX`: Wavenumber range for spectral slope fitting
- `--nu VALUE`: Kinematic viscosity for Reynolds number computation
- `--output FILE`: Save statistics to file (default: print to stdout)

## Dependencies

### Required

- **Python** ≥ 3.8
- **Dedalus** v3 (spectral PDE solver)
- **NumPy** (array operations)
- **h5py** (HDF5 I/O)
- **mpi4py** (MPI parallelisation)

### Post-Processing

- **matplotlib** ≥ 3.3.0 (plotting)
- **scipy** ≥ 1.6.0 (statistical analysis)

### Installation

The easiest way to install Dedalus is via conda:

```bash
conda create -n ns2d python=3.11
conda activate ns2d
conda install -c conda-forge dedalus
```

Then install remaining dependencies:

```bash
pip install -r requirements.txt
```

For detailed Dedalus installation instructions, see: https://dedalus-project.readthedocs.io/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with the [Dedalus Project](https://dedalus-project.org/)
