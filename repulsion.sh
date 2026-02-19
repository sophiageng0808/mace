#!/bin/bash
#SBATCH --job-name=mace_repulsion_modes
#SBATCH --output=outslurm/%x_%A_%a.out
#SBATCH --error=outslurm/%x_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0

set -euo pipefail
mkdir -p outslurm

# -----------------------------
# Paths / env
# -----------------------------
BASE_REPO="/h/400/sophiageng/mace"
REPULSION_WT="/h/400/sophiageng/mace_worktrees/repulsion"
RUNS_ROOT="/h/400/sophiageng/mace_worktrees/jobs_repulsion"

VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

DATA="$BASE_REPO/data/overfit100_E0sub.extxyz"
E0S_JSON="$BASE_REPO/data/overfit100_E0zeros.json"
EKEY="REF_energy"
FKEY="REF_forces"
EPOCHS=200

# Threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate venv
source "$VENV_ACTIVATE"

export PYTHONPATH="$REPULSION_WT:${PYTHONPATH:-}"

# -----------------------------
# Task mapping
# 0 = baseline (no pair repulsion)
# 1 = zbl only (mode 1)
# 2 = r12 only (mode 2)
# 3 = zbl + r12 (mode 3)
# -----------------------------
TASK="${SLURM_ARRAY_TASK_ID}"

NAME=""
PAIR_FLAGS=""

case "$TASK" in
  0)
    NAME="baseline_norepulsion"
    PAIR_FLAGS=""
    ;;
  1)
    NAME="repulsion_zbl"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds zbl --pair_repulsion_mode 1"
    ;;
  2)
    NAME="repulsion_r12"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds r12 --pair_repulsion_mode 2 --r12_scale 1.0 --r12_cutoff 0.8"
    ;;
  3)
    NAME="repulsion_zbl_r12"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds zbl r12 --pair_repulsion_mode 3 --r12_scale 1.0 --r12_cutoff 0.8"
    ;;
  *)
    echo "Unknown task id: $TASK"
    exit 1
    ;;
esac

# -----------------------------
# Per-task run directory
# -----------------------------
mkdir -p "$RUNS_ROOT"
RUN_DIR="$RUNS_ROOT/${SLURM_JOB_ID}_${TASK}_${NAME}"
mkdir -p "$RUN_DIR"

cd "$RUN_DIR"
rm -rf "./checkpoints" "./logs" "./results" "./wandb" 2>/dev/null || true
mkdir -p "./checkpoints"

echo "============================================================"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "RUN_DIR=$RUN_DIR"
echo "REPULSION_WT=$REPULSION_WT"
echo "PYTHONPATH=$PYTHONPATH"
echo "RUN NAME=$NAME"
echo "PAIR_FLAGS=$PAIR_FLAGS"
echo "DATA=$DATA"
echo "============================================================"

python - <<'PY'
import inspect, os
import mace
import mace.tools.arg_parser as ap
import mace.tools.scripts_utils as su
print("mace from         :", inspect.getsourcefile(mace))
print("arg_parser from   :", inspect.getsourcefile(ap))
print("scripts_utils from:", inspect.getsourcefile(su))
print("PYTHONPATH        :", os.environ.get("PYTHONPATH"))
PY

python -m mace.cli.run_train --help | grep -E "pair_repulsion_kinds|pair_repulsion_mode|r12_scale|r12_cutoff|zbl_p|pair_repulsion_r_min"

python -m mace.cli.run_train \
  --name "$NAME" \
  --train_file "$DATA" \
  --valid_file "$DATA" \
  --test_file  "$DATA" \
  --energy_key "$EKEY" \
  --forces_key "$FKEY" \
  --E0s "$E0S_JSON" \
  --max_num_epochs "$EPOCHS" \
  --patience 50 \
  --seed 0 \
  --batch_size 10 \
  --default_dtype float64 \
  --device cpu \
  --work_dir "$RUN_DIR" \
  --checkpoints_dir "$RUN_DIR/checkpoints" \
  ${PAIR_FLAGS}

echo "[task $TASK] Done: $NAME"
