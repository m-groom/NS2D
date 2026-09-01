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
NX=1024
NY=1024
LX=6.283185307179586  # 2*pi
LY=6.283185307179586  # 2*pi

# Physics
NU=9.3e-6
ALPHA=0.003

# Forcing
FORCING_TYPE="stochastic"
STOCH_TYPE="ou"
KMIN=56.0
KMAX=72.0
EPS_TARGET=1.71e-5
POWER_MODE="constant"  # or "sigma" to bypass rescaling and use f_sigma directly
F_SIGMA=0.02           # used directly if POWER_MODE=="sigma"
TAU_OU=0.3
EPS_SMOOTH=0.0

# Time integration
T_END=5000.0
CFL_SAFETY=0.4
CFL_MAX_DT=8e-3

# Output
OUTDIR="/scratch3/gro175/NS2D/striped/"
SNAP_DT=0.08
CHECKPOINT_DT=10.0
SPECTRA_DT=0.08
SCALARS_DT=0.08
N_REALISATIONS=6
SEED=1234

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
    --checkpoint_dt $CHECKPOINT_DT \
    --spectra_dt $SPECTRA_DT \
    --scalars_dt $SCALARS_DT \
    --n_realisations $N_REALISATIONS \
    --seed $SEED

echo ""
echo "Simulation complete! Output written to: $OUTDIR"
