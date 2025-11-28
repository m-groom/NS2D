#!/bin/bash
#
# Example script for running the Kolmogorov flow benchmark with deterministic forcing.
#
# Usage (single rank):
#   bash run_kolmogorov.sh
#
# Usage (MPI, e.g. 8 ranks):
#   mpiexec -n 8 bash run_kolmogorov.sh
#
# Results are written to OUTDIR/<timestamp>/run_<r>/ for each realisation.
# The parameters below mirror the Li et al. (ICLR 2021) configuration and can
# be tweaked to explore different initial-condition amplitudes or driving
# strengths.

set -e  # Exit on error

# Simulation parameters
NX=256
NY=256
LX=6.283185307179586  # 2*pi
LY=6.283185307179586  # 2*pi

# Physics
NU=5e-4
ALPHA=0.025

# Kolmogorov forcing (f_x = F0 * sin(k_drive * y + phase), f_y = 0)
FORCING_TYPE="kolmogorov"
KOLMOGOROV_F0=0.1
K_DRIVE=4.0
K_PHASE=0.0

# Initial-condition spectrum (controls amplitude/realisations)
IC_ALPHA=49.0
IC_POWER=2.5
IC_SCALE=18.520259177452136  # 7**1.5; increase/decrease to adjust IC energy
# Optional: pin the initial kinetic energy (domain-average 0.5<|u|^2>) so that
# different seeds give comparable amplitudes. Leave empty to skip rescaling.
IC_ENERGY=""
# Optional: choose a dedicated IC seed (defaults to --seed when empty).
IC_SEED=""

# Time integration
T_END=200.0
CFL_SAFETY=0.4
CFL_MAX_DT=1e-2

# Output
OUTDIR="./kolmogorov_runs"
SNAP_DT=0.5
SPECTRA_DT=0.2
SCALARS_DT=0.05
N_REALISATIONS=1

# Reproducibility / ensemble control
SEED=42

# Run simulation
CMD=(
python ../main.py \
    --Nx $NX \
    --Ny $NY \
    --Lx $LX \
    --Ly $LY \
    --nu $NU \
    --alpha $ALPHA \
    --forcing $FORCING_TYPE \
    --kolmogorov_f0 $KOLMOGOROV_F0 \
    --k_drive $K_DRIVE \
    --k_phase $K_PHASE \
    --ic_alpha $IC_ALPHA \
    --ic_power $IC_POWER \
    --ic_scale $IC_SCALE \
    --t_end $T_END \
    --cfl_safety $CFL_SAFETY \
    --cfl_max_dt $CFL_MAX_DT \
    --outdir $OUTDIR \
    --snap_dt $SNAP_DT \
    --spectra_dt $SPECTRA_DT \
    --scalars_dt $SCALARS_DT \
    --n_realisations $N_REALISATIONS \
    --seed $SEED
)

if [ -n "$IC_ENERGY" ]; then
    CMD+=(--ic_energy $IC_ENERGY)
fi
if [ -n "$IC_SEED" ]; then
    CMD+=(--ic_seed $IC_SEED)
fi

"${CMD[@]}"

echo ""
echo "Kolmogorov benchmark complete! Output written to: $OUTDIR"
