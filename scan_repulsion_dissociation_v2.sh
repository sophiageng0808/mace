#!/bin/bash
#SBATCH --job-name=repulsion_dissoc_scan
#SBATCH --output=outslurm/%x_%A_%a.out
#SBATCH --error=outslurm/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00

set -euo pipefail
mkdir -p outslurm

BASE_REPO="/scratch/sophiag/mace"
REPULSION_WT="/scratch/sophiag/mace_branches/exp-repulsion-v2"
VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source "$VENV_ACTIVATE"
export PYTHONPATH="$REPULSION_WT:${PYTHONPATH:-}"

cd "$REPULSION_WT"

python dissociation_scan_overfit100_repulsion_v2.py \
  --run_name "slurm_${SLURM_JOB_ID}"
