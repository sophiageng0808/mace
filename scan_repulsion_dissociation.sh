#!/bin/bash
#SBATCH --job-name=repulsion_dissoc_scan
#SBATCH --output=outslurm/%x_%j.out
#SBATCH --error=outslurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

set -euo pipefail
mkdir -p outslurm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}"

if [ -d "/scratch/$USER/mace" ]; then
  DEFAULT_BASE="/scratch/$USER/mace"
else
  DEFAULT_BASE="$SCRIPT_DIR"
fi

BASE_REPO="${BASE_REPO:-$DEFAULT_BASE}"
REPULSION_WT="${REPULSION_WT:-${SUBMIT_DIR:-$SCRIPT_DIR}}"
VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source "$VENV_ACTIVATE"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPULSION_WT:$REPULSION_WT/mace:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"

cd "$REPULSION_WT"

PYTHONPATH="$REPULSION_WT:$REPULSION_WT/mace:${PYTHONPATH:-}" \
python dissociation_scan_overfit100_repulsion.py \
  --run_name "slurm_${SLURM_JOB_ID}"
