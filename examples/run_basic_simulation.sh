#!/bin/bash
#
# Example script for running a basic NS2D simulation
#
# This script demonstrates a typical 2D turbulence simulation with:
# - Moderate resolution (512x512)
# - Stochastic OU forcing in wavenumber band [20, 30]
# - Constant-power injection at epsilon = 0.01
# - Standard output cadences
#
# Usage:
#   bash run_basic_simulation.sh
#
# For MPI parallel execution:
#   mpiexec -n 8 bash run_basic_simulation.sh

set -e  # Exit on error

# Simulation parameters
NX=512
NY=512
LX=6.283185307179586  # 2*pi
LY=6.283185307179586  # 2*pi

# Physics
NU=1e-4
ALPHA=0.05

# Forcing
FORCING_TYPE="stochastic"
STOCH_TYPE="ou"
KMIN=20.0
KMAX=30.0
EPS_TARGET=0.01
TAU_OU=0.3
EPS_SMOOTH=0.3

# Time integration
T_END=100.0
CFL_SAFETY=0.4

# Output
OUTDIR="output_basic_simulation"
TAG="basic"
SNAP_DT=2.0
SPECTRA_DT=0.5
SCALARS_DT=0.1

# Run simulation
python ../main.py \
    --Nx $NX \
    --Ny $NY \
    --Lx $LX \
    --Ly $LY \
    --nu $NU \
    --alpha $ALPHA \
    --forcing $FORCING_TYPE \
    --stoch_type $STOCH_TYPE \
    --kmin $KMIN \
    --kmax $KMAX \
    --eps_target $EPS_TARGET \
    --tau_ou $TAU_OU \
    --eps_smooth $EPS_SMOOTH \
    --t_end $T_END \
    --cfl_safety $CFL_SAFETY \
    --outdir $OUTDIR \
    --tag $TAG \
    --snap_dt $SNAP_DT \
    --spectra_dt $SPECTRA_DT \
    --scalars_dt $SCALARS_DT \
    --precision float64

echo ""
echo "Simulation complete! Output written to: $OUTDIR"
echo "To visualize results, see the post-processing examples in the documentation."
