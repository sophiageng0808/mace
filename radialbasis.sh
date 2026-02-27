#!/bin/bash
#SBATCH --job-name=mace_radialbasis_cos_modes
#SBATCH --array=2-3
#SBATCH --output=outslurm/%x_%A.out
#SBATCH --error=outslurm/%x_%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p outslurm

# -----------------------------
# Paths / env
# -----------------------------
BASE_REPO="/h/400/sophiageng/mace"
RADIAL_WT="/h/400/sophiageng/mace_worktrees/radial_cosine"
RUNS_ROOT="/h/400/sophiageng/mace_worktrees/jobs_radialbasis"

VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

DATA="$BASE_REPO/data/overfit100_E0sub.extxyz"
EKEY="REF_energy"
FKEY="REF_forces"
E0S_JSON="$BASE_REPO/data/overfit100_E0zeros.json"
EPOCHS=400
BATCH_SIZE=1
NUM_CHANNELS=32

# Threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Activate venv
source "$VENV_ACTIVATE"

TASK="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "$TASK" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with: sbatch --array=2-3 radialbasis.sh"
  exit 1
fi

# -----------------------------
# Task mapping
# -----------------------------
CODE_DIR=""
MODE=""
NAME=""

case "$TASK" in
  0)
    CODE_DIR="$BASE_REPO"
    MODE=""
    NAME="main_baseline"
    ;;
  1)
    CODE_DIR="$RADIAL_WT"
    MODE="sine"
    NAME="radial_sine"
    ;;
  2)
    CODE_DIR="$RADIAL_WT"
    MODE="cos"
    NAME="radial_cos"
    ;;
  3)
    CODE_DIR="$RADIAL_WT"
    MODE="cosm1_over_r"
    NAME="radial_cosm1_over_r"
    ;;
  4)
    CODE_DIR="$RADIAL_WT"
    MODE="cos_over_r"
    NAME="radial_cos_over_r"
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

  # Ensure correct code import
  export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

  # Set/unset radial mode
  if [[ -n "$MODE" ]]; then
    export MACE_BESSEL_MODE="$MODE"
  else
    unset MACE_BESSEL_MODE
  fi

  # -----------------------------
  # Fresh run area
  # -----------------------------
  cd "$RUN_DIR"
  rm -rf "./checkpoints" "./logs" "./results" "./wandb" 2>/dev/null || true
  mkdir -p "./checkpoints"
  export MACE_CHECKPOINT_DIR="$RUN_DIR/checkpoints"

  # Detect checkpoint save flag
  SAVE_EVERY_FLAG=""
  HELP_TEXT="$(mace_run_train --help 2>&1 || true)"
  if echo "$HELP_TEXT" | grep -q -- "--save_every"; then
    SAVE_EVERY_FLAG="--save_every 10"
  elif echo "$HELP_TEXT" | grep -q -- "--checkpoint_interval"; then
    SAVE_EVERY_FLAG="--checkpoint_interval 10"
  elif echo "$HELP_TEXT" | grep -q -- "--save_interval"; then
    SAVE_EVERY_FLAG="--save_interval 10"
  fi

  echo "============================================================"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
  echo "RUN_DIR=$RUN_DIR"
  echo "CODE_DIR=$CODE_DIR"
  echo "RUN NAME=$NAME"
  echo "MACE_BESSEL_MODE=${MACE_BESSEL_MODE-<unset>}"
  echo "CHECKPOINT_DIR=$RUN_DIR/checkpoints"
  echo "SAVE_EVERY_FLAG=${SAVE_EVERY_FLAG:-<none-found>}"
  echo "============================================================"

  python - <<'PY'
import os, inspect
import mace
import mace.modules.radial as r
print("MACE_BESSEL_MODE =", os.environ.get("MACE_BESSEL_MODE"))
print("mace from        =", inspect.getsourcefile(mace))
print("radial from      =", inspect.getsourcefile(r))
print("MACE_CHECKPOINT_DIR =", os.environ.get("MACE_CHECKPOINT_DIR"))
PY

  # -----------------------------
  # Train
  # -----------------------------
  # shellcheck disable=SC2086
  mace_run_train \
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
    --batch_size "$BATCH_SIZE" \
    --num_channels "$NUM_CHANNELS" \
    --num_workers 0 \
    --pin_memory False \
    --default_dtype float32 \
    --device cuda \
    --wandb \
    --wandb_name "$NAME" \
    ${SAVE_EVERY_FLAG}

echo "[task $TASK] Done: $NAME"
