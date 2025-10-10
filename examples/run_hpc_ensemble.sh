#!/bin/bash
#SBATCH --job-name=ns2d_ensemble
#SBATCH --account=OD-233742
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=ns2d_ensemble_%j.out
#SBATCH --error=ns2d_ensemble_%j.err
#
# Example SLURM script for running NS2D ensemble on HPC cluster
#
# This script runs 10 independent realizations of a 2D turbulence
# simulation at high resolution (2048x2048) using 128 MPI processes.
#
# Submit with: sbatch run_hpc_ensemble.sh

set -e

# Load required modules (adjust for your cluster)
# module load python/3.11
# module load openmpi/4.1.5
# module load hdf5/1.14.0

# Activate conda environment (if using conda)
# source activate ns2d

echo "==============================================="
echo "NS2D Ensemble Simulation"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of MPI tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "==============================================="

# Simulation parameters
NX=2048
NY=2048
NU=2e-5
ALPHA=0.05
N_REAL=10
T_END=500.0

# Forcing parameters
KMIN=30.0
KMAX=40.0
EPS_TARGET=0.02

# Output settings
OUTDIR="${SCRATCH}/ns2d_ensemble_run_${SLURM_JOB_ID}"
TAG="ensemble"

echo ""
echo "Output directory: $OUTDIR"
echo "Grid resolution: ${NX}x${NY}"
echo "Number of realizations: $N_REAL"
echo ""

# Run with MPI
srun python ../main.py \
    --Nx $NX \
    --Ny $NY \
    --nu $NU \
    --alpha $ALPHA \
    --forcing stochastic \
    --stoch_type ou \
    --kmin $KMIN \
    --kmax $KMAX \
    --eps_target $EPS_TARGET \
    --tau_ou 0.3 \
    --eps_smooth 0.3 \
    --t_end $T_END \
    --n_realisations $N_REAL \
    --seed 42 \
    --outdir $OUTDIR \
    --tag $TAG \
    --snap_dt 5.0 \
    --spectra_dt 1.0 \
    --scalars_dt 0.2 \
    --precision float64

echo ""
echo "==============================================="
echo "Simulation complete!"
echo "End time: $(date)"
echo "Output written to: $OUTDIR"
echo "==============================================="
