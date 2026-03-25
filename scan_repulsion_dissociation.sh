#!/bin/bash
#SBATCH --job-name=repulsion_dissoc_scan
#SBATCH --output=outslurm/%x_%j.out
#SBATCH --error=outslurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

set -euo pipefail
mkdir -p outslurm

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MACE="${BASE_REPO:-/scratch/$USER/mace}"
WT="${REPULSION_WT:-${SLURM_SUBMIT_DIR:-$ROOT}}"
G="${GROUP_NAME:-r12_zbl}"

N=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N \
  NUMEXPR_NUM_THREADS=$N VECLIB_MAXIMUM_THREADS=$N BLIS_NUM_THREADS=$N

C="/scratch/$USER/.cache"
mkdir -p "$C/wandb" "$C/torch/kernels"
export WANDB_CACHE_DIR="$C/wandb" PYTORCH_KERNEL_CACHE_PATH="$C/torch/kernels"

source "$MACE/.macevenv/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_DISABLED="${WANDB_DISABLED:-false}"

cd "$WT"
RUN="${RUN_NAME:-slurm_${SLURM_JOB_ID:-manual}}"
python dissociation_scan_overfit100_repulsion.py --run_name "$RUN" --group "$G"
echo "outputs: $MACE/outputs/dissociation_scans_overfit100/$G/$RUN/"
