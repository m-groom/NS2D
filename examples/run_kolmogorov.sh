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
# The parameters below can be tweaked to explore different initial-condition
# amplitudes or driving strengths. See Li et al. (ICLR 2021) for the reference
# Kolmogorov configuration.

set -e  # Exit on error

# Simulation parameters
NX=128
NY=128
LX=6.283185307179586  # 2*pi
LY=6.283185307179586  # 2*pi

# Physics
NU=2e-3
ALPHA=1e-8

# Kolmogorov forcing (f_x = F0 * sin(k_drive * y + phase), f_y = 0)
FORCING_TYPE="kolmogorov"
KOLMOGOROV_F0=2.8284271247461903
K_DRIVE=4.0
K_PHASE=3.9269908169872414 # 5*pi/4
POWER_MODE="sigma"

# Initial-condition spectrum (controls amplitude/realisations)
IC_ALPHA=49.0
IC_POWER=2.5
IC_SCALE=18.520259177452132  # 7**1.5; increase/decrease to adjust IC energy
# Optional: pin the initial kinetic energy (domain-average 0.5<|u|^2>) so that
# different seeds give comparable amplitudes. Leave empty to skip rescaling.
IC_ENERGY="0.2"
# Optional: choose a dedicated IC seed (defaults to --seed when empty).
IC_SEED=""

# Time integration
T_END=101.0
CFL_SAFETY=0.4
CFL_MAX_DT=0.001953125

# Output
OUTDIR="/scratch3/gro175/NS2D/kolmogorov"
SNAP_DT=0.015625
SPECTRA_DT=0.015625
SCALARS_DT=0.015625
N_REALISATIONS=3605

# Reproducibility / ensemble control
SEED=420

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
    --power_mode $POWER_MODE \
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
