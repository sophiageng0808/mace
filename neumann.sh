#!/bin/bash
#SBATCH --job-name=mace_neumann_3modes
#SBATCH --output=outslurm/%x_%A_%a.out
#SBATCH --error=outslurm/%x_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0

set -euo pipefail
mkdir -p outslurm

# -----------------------------
# Paths / env
# -----------------------------
BASE_REPO="/h/400/sophiageng/mace"
NEUMANN_WT="/h/400/sophiageng/mace_worktrees/neumann"
RUNS_ROOT="/h/400/sophiageng/mace_worktrees/jobs_neumann"

VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

DATA="$BASE_REPO/data/overfit100_E0sub.extxyz"
E0S_JSON="$BASE_REPO/data/overfit100_E0zeros.json"
EKEY="REF_energy"
FKEY="REF_forces"

EPOCHS=200
PATIENCE=50
SEED=0

BATCH_SIZE=10
DEFAULT_DTYPE="float64"

# Threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Disable wandb (optional)
export WANDB_MODE=disabled
export WANDB_SILENT=true

# Activate venv
source "$VENV_ACTIVATE"

# Ensure correct code import
export PYTHONPATH="$NEUMANN_WT:${PYTHONPATH:-}"

# -----------------------------
# Task mapping
# -----------------------------
TASK="${SLURM_ARRAY_TASK_ID}"

NAME=""
RADIAL_TYPE=""
NEUMANN_EPS="1e-3"
RADIAL_FLAGS=()

case "$TASK" in
  0)
    NAME="baseline_bessel"
    RADIAL_TYPE="bessel"
    ;;
  1)
    NAME="radial_neumann"
    RADIAL_TYPE="neumann"
    ;;
  2)
    NAME="radial_neumann_reg"
    RADIAL_TYPE="neumann_reg"
    RADIAL_FLAGS+=(--neumann_eps "$NEUMANN_EPS")
    ;;
  *)
    echo "Unknown task id: $TASK"
    exit 1
    ;;
esac

# -----------------------------
# Run directory
# -----------------------------
mkdir -p "$RUNS_ROOT"
RUN_TAG="${SLURM_JOB_ID}_${TASK}_${NAME}"
RUN_DIR="$RUNS_ROOT/$RUN_TAG"
mkdir -p "$RUN_DIR"

cd "$RUN_DIR"
rm -rf ./checkpoints ./logs ./results ./wandb 2>/dev/null || true
mkdir -p ./checkpoints
export MACE_CHECKPOINT_DIR="$RUN_DIR/checkpoints"

# -----------------------------
# Detect checkpoint + num_workers flags
# -----------------------------
SAVE_EVERY_FLAG=()
NUM_WORKERS_FLAG=()

HELP_TEXT="$(mace_run_train --help 2>&1 || true)"
if echo "$HELP_TEXT" | grep -q -- "--save_every"; then
  SAVE_EVERY_FLAG+=(--save_every 10)
elif echo "$HELP_TEXT" | grep -q -- "--checkpoint_interval"; then
  SAVE_EVERY_FLAG+=(--checkpoint_interval 10)
elif echo "$HELP_TEXT" | grep -q -- "--save_interval"; then
  SAVE_EVERY_FLAG+=(--save_interval 10)
fi

if echo "$HELP_TEXT" | grep -q -- "--num_workers"; then
  NUM_WORKERS_FLAG+=(--num_workers 0)
fi

echo "============================================================"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "RUN_DIR=$RUN_DIR"
echo "NEUMANN_WT=$NEUMANN_WT"
echo "PYTHONPATH=$PYTHONPATH"
echo "RUN NAME=$NAME"
echo "RADIAL_TYPE=$RADIAL_TYPE"
echo "NEUMANN_EPS=$NEUMANN_EPS"
echo "MACE_CHECKPOINT_DIR=$MACE_CHECKPOINT_DIR"
echo "SAVE_EVERY_FLAG=${SAVE_EVERY_FLAG[*]:-<none-found>}"
echo "NUM_WORKERS_FLAG=${NUM_WORKERS_FLAG[*]:-<none-found>}"
echo "DEFAULT_DTYPE=$DEFAULT_DTYPE  BATCH_SIZE=$BATCH_SIZE"
echo "============================================================"

python - <<'PY'
import os, inspect
import mace
import mace.modules.radial as r
print("mace from           :", inspect.getsourcefile(mace))
print("radial from         :", inspect.getsourcefile(r))
print("PYTHONPATH          :", os.environ.get("PYTHONPATH"))
print("MACE_CHECKPOINT_DIR :", os.environ.get("MACE_CHECKPOINT_DIR"))
PY

# -----------------------------
# Train
# -----------------------------
mace_run_train \
  --name "$NAME" \
  --train_file "$DATA" \
  --valid_file "$DATA" \
  --test_file  "$DATA" \
  --energy_key "$EKEY" \
  --forces_key "$FKEY" \
  --E0s "$E0S_JSON" \
  --max_num_epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --pin_memory False \
  --default_dtype "$DEFAULT_DTYPE" \
  --device cpu \
  --radial_type "$RADIAL_TYPE" \
  "${RADIAL_FLAGS[@]}" \
  "${SAVE_EVERY_FLAG[@]}" \
  "${NUM_WORKERS_FLAG[@]}"

echo "[task $TASK] Done: $NAME"
