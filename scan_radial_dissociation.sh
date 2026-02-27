#!/bin/bash
#SBATCH --job-name=radial_dissoc_scan
#SBATCH --output=outslurm/%x_%j.out
#SBATCH --error=outslurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p outslurm

BASE_REPO="/h/400/sophiageng/mace"
RADIAL_WT="/h/400/sophiageng/mace_worktrees/radial_cosine"
VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source "$VENV_ACTIVATE"
export PYTHONPATH="$RADIAL_WT:${PYTHONPATH:-}"

cd "$RADIAL_WT"

python dissociation_scan_overfit100_radialbasis.py \
  --run_name "slurm_${SLURM_JOB_ID}"
