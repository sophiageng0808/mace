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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_REPO="${BASE_REPO:-/scratch/$USER/mace}"
REPULSION_WT="${REPULSION_WT:-$SCRIPT_DIR}"

FAILED_CSV="${FAILED_CSV:-$BASE_REPO/outputs/dissociation_scans_overfit100/repulsion/slurm_358187/zbl/failed_curves_metrics.csv}"
MODEL_PATH="${MODEL_PATH:-/scratch/$USER/mace_worktrees/jobs_repulsion/train4m_multiday_1_repulsion_zbl/repulsion_zbl.model}"
DATA_EXTXYZ="${DATA_EXTXYZ:-$BASE_REPO/data/train4M_split_25k/test.extxyz}"
OUT_DIR="${OUT_DIR:-$BASE_REPO/outputs/dissociation_scans_overfit100/repulsion/slurm_358187/zbl}"

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
export PYTHONPATH="$REPULSION_WT${PYTHONPATH:+:$PYTHONPATH}"

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
