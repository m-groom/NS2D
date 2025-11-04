#!/bin/bash
#
# Example script for running a basic NS2D simulation
#
# Usage:
#   bash run_simulation.sh
#
# For MPI parallel execution:
#   mpiexec -n 8 bash run_simulation.sh

set -e  # Exit on error

# Simulation parameters
NX=256
NY=256
LX=6.283185307179586  # 2*pi
LY=6.283185307179586  # 2*pi

# Physics
NU=5e-4
ALPHA=0.023

# Forcing
FORCING_TYPE="stochastic"
STOCH_TYPE="ou"
KMIN=8.0
KMAX=12.0
EPS_TARGET=0.001
POWER_MODE="constant"  # or "sigma" to bypass rescaling and use f_sigma directly
F_SIGMA=0.02           # used directly if POWER_MODE=="sigma"
TAU_OU=0.3
EPS_SMOOTH=0.0

# Time integration
T_END=500.0
CFL_SAFETY=0.4
CFL_MAX_DT=1e-2

# Output
OUTDIR="/scratch3/gro175/NS2D/striped"
SNAP_DT=0.1
SPECTRA_DT=0.2
SCALARS_DT=0.05
N_REALISATIONS=10

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
    --power_mode $POWER_MODE \
    --eps_target $EPS_TARGET \
    --f_sigma $F_SIGMA \
    --tau_ou $TAU_OU \
    --eps_smooth $EPS_SMOOTH \
    --t_end $T_END \
    --cfl_safety $CFL_SAFETY \
    --cfl_max_dt $CFL_MAX_DT \
    --outdir $OUTDIR \
    --snap_dt $SNAP_DT \
    --spectra_dt $SPECTRA_DT \
    --scalars_dt $SCALARS_DT \
    --n_realisations $N_REALISATIONS

echo ""
echo "Simulation complete! Output written to: $OUTDIR"
