#!/bin/bash
#SBATCH --job-name=mace_consistency
#SBATCH --output=outslurm/%x_%A_%a.out
#SBATCH --error=outslurm/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}"

if [ -d "/scratch/$USER/mace" ]; then
  DEFAULT_BASE="/scratch/$USER/mace"
else
  DEFAULT_BASE="$SCRIPT_DIR"
fi

BASE_REPO="${BASE_REPO:-$DEFAULT_BASE}"
VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

source "$VENV_ACTIVATE"
export PYTHONNOUSERSITE=1

SCRIPT="/scratch/$USER/mace_branches/exp-repulsion/model_consistency_checks.py"
MODELS="/scratch/$USER/mace_worktrees/jobs_repulsion/309421_2_repulsion_r12/repulsion_r12.model,/scratch/$USER/mace_worktrees/jobs_repulsion/309419_1_repulsion_zbl/repulsion_zbl.model,/scratch/$USER/mace_worktrees/jobs_repulsion/309413_3_repulsion_zbl_r12/repulsion_zbl_r12.model"
DATA="/scratch/$USER/mace/data/overfit100_E0sub.extxyz"
OUTDIR="/scratch/$USER/mace/outputs/consistency_checks/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTDIR"
CACHE_BASE="/scratch/$USER/.cache"
mkdir -p "$CACHE_BASE"

export MPLCONFIGDIR="$CACHE_BASE/matplotlib"
export XDG_CACHE_HOME="$CACHE_BASE"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"
export WANDB_DIR="$OUTDIR/wandb"
export WANDB_DATA_DIR="$OUTDIR/wandb"
mkdir -p "$WANDB_DIR"


python "$SCRIPT" \
  --models "$MODELS" \
  --data "$DATA" \
  --n-structures 1 \
  --scan-distances "0.30,0.25" \
  --hs "1e-4" \
  --preflight \
  --device cuda \
  --outdir "$OUTDIR/preflight"

python "$SCRIPT" \
  --models "$MODELS" \
  --data "$DATA" \
  --n-structures 0 \
  --scan-distances "0.30,0.25,0.20" \
  --hs "1e-4,3e-4,1e-3" \
  --device cuda \
  --wandb \
  --wandb-project "mace-consistency-checks" \
  --wandb-group "repulsion" \
  --outdir "$OUTDIR"
