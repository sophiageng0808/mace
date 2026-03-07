#!/bin/bash
#SBATCH --job-name=plot_failed_zbl
#SBATCH --output=outslurm/%x_%j.out
#SBATCH --error=outslurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p outslurm

BASE_REPO="/scratch/sophiag/mace"
REPULSION_WT="/scratch/sophiag/mace_branches/exp-repulsion-v2"

FAILED_CSV="/scratch/sophiag/mace/outputs/dissociation_scans_overfit100/repulsion/repulsion_v2.3/zbl/failed_curves_metrics.csv"
MODEL_PATH="/scratch/sophiag/mace_worktrees/job_repulsion_v2/315222_1_repulsion_zbl/repulsion_zbl.model"
DATA_EXTXYZ="/scratch/sophiag/mace/data/overfit100_E0sub.extxyz"

OUT_DIR="/scratch/sophiag/mace/outputs/dissociation_scans_overfit100/repulsion/slurm_315359/plots_failed_zbl_with_bonds"

VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND="Agg"
export MPLCONFIGDIR="/tmp/mplconfig"

source "$VENV_ACTIVATE"
export PYTHONPATH="$REPULSION_WT:${PYTHONPATH:-}"

cd "$REPULSION_WT"

python plot_failed_zbl_curves_and_geometries.py \
  --failed_csv "$FAILED_CSV" \
  --model "$MODEL_PATH" \
  --data_extxyz "$DATA_EXTXYZ" \
  --steps 50 \
  --start 0.7 \
  --end 0.2 \
  --scale linear \
  --device cuda \
  --out_dir "$OUT_DIR"
