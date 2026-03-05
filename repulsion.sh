#!/bin/bash
#SBATCH --job-name=mace_repulsion_modes
#SBATCH --output=outslurm/%x_%A_%a.out
#SBATCH --error=outslurm/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

set -euo pipefail
mkdir -p outslurm

# ============================================================
# Repulsion sweep (4 tasks)
# 0 = baseline (no pair repulsion)
# 1 = ZBL only
# 2 = r^-12 only
# 3 = ZBL + r^-12
# ============================================================

# -----------------------------
# Paths / env
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}"

if [ -d "/scratch/$USER/mace" ]; then
  DEFAULT_BASE="/scratch/$USER/mace"
else
  DEFAULT_BASE="$SCRIPT_DIR"
fi

BASE_REPO="${BASE_REPO:-$DEFAULT_BASE}"
REPULSION_WT="${REPULSION_WT:-${SUBMIT_DIR:-$SCRIPT_DIR}}"
RUNS_ROOT="${RUNS_ROOT:-/scratch/$USER/mace_worktrees/jobs_repulsion}"

VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

DATA="$BASE_REPO/data/overfit100_E0sub.extxyz"
E0S_JSON="$BASE_REPO/data/overfit100_E0zeros.json"
EKEY="REF_energy"
FKEY="REF_forces"

EPOCHS=400
R_MAX=4.0
RADIAL_MLP="[32,32]"
PATIENCE=120
SEED=0
LR=0.003
SCHEDULER_PATIENCE=10
LR_FACTOR=0.5
ZBL_SCALE=0.1

BATCH_SIZE=1
DEFAULT_DTYPE="float32"
NUM_CHANNELS=32
NUM_INTERACTIONS=2
CORRELATION=2
MAX_ELL=2
MAX_L=1

# Threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Activate venv
source "$VENV_ACTIVATE"

# Ensure correct code import
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPULSION_WT:$REPULSION_WT/mace:${PYTHONPATH:-}"

# -----------------------------
# Task mapping (array id)
# -----------------------------
TASK="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "$TASK" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with: sbatch --array=0-3 repulsion.sh"
  exit 1
fi

NAME=""
PAIR_FLAGS=""

case "$TASK" in
  0)
    NAME="baseline_norepulsion"
    ;;
  1)
    NAME="repulsion_zbl"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds zbl --pair_repulsion_mode 1 --zbl_scale $ZBL_SCALE"
    ;;
  2)
    NAME="repulsion_r12"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds r12 --pair_repulsion_mode 2 --r12_scale 1.0 --r12_cutoff 0.8"
    ;;
  3)
    NAME="repulsion_zbl_r12"
    # IMPORTANT: your argparse choices require repeated flag
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds zbl --pair_repulsion_kinds r12 --pair_repulsion_mode 3 --zbl_scale $ZBL_SCALE --r12_scale 1.0 --r12_cutoff 0.8"
    ;;
  *)
    echo "Unknown task id: $TASK (expected 0-3)"
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

  # Avoid permission errors for matplotlib/fontconfig caches on shared systems
  export MPLCONFIGDIR="$RUN_DIR/.mplconfig"
  export XDG_CACHE_HOME="$RUN_DIR/.cache"
  export FC_CACHEDIR="$XDG_CACHE_HOME/fontconfig"
  mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FC_CACHEDIR"

  # W&B: force offline on compute nodes (can override via env)
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export WANDB_DISABLED="${WANDB_DISABLED:-false}"
  # W&B auth: source env file or load key file; keep offline if missing
  WANDB_ENV_FILE="${WANDB_ENV_FILE:-/home/$USER/.wandb_env}"
  WANDB_KEY_FILE="${WANDB_KEY_FILE:-/scratch/$USER/.wandb_api_key}"
  if [ -f "$WANDB_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$WANDB_ENV_FILE"
  fi
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
  fi
  if [ -z "${WANDB_API_KEY:-}" ]; then
    if [ "${WANDB_MODE}" != "offline" ]; then
      export WANDB_MODE="offline"
    fi
    export WANDB_DISABLED="false"
  fi

  # W&B destination (set these in env or defaults here)
  export WANDB_PROJECT="${WANDB_PROJECT:-mace}"
  export WANDB_ENTITY="${WANDB_ENTITY:-}"
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[wandb] WANDB_API_KEY not found; logging disabled (offline)."
  else
    echo "[wandb] WANDB_PROJECT=${WANDB_PROJECT} WANDB_ENTITY=${WANDB_ENTITY:-<default>}"
  fi

  # -----------------------------
  # Detect checkpoint + num_workers flags
  # -----------------------------
  SAVE_EVERY_FLAG=""
  NUM_WORKERS_FLAG=""

  HELP_TEXT="$(PYTHONPATH="$REPULSION_WT:$REPULSION_WT/mace:${PYTHONPATH:-}" python -m mace.cli.run_train --help 2>&1 || true)"
  if echo "$HELP_TEXT" | grep -q -- "--save_every"; then
    SAVE_EVERY_FLAG="--save_every 10"
  elif echo "$HELP_TEXT" | grep -q -- "--checkpoint_interval"; then
    SAVE_EVERY_FLAG="--checkpoint_interval 10"
  elif echo "$HELP_TEXT" | grep -q -- "--save_interval"; then
    SAVE_EVERY_FLAG="--save_interval 10"
  fi

  if echo "$HELP_TEXT" | grep -q -- "--num_workers"; then
    NUM_WORKERS_FLAG="--num_workers 0"
  fi

  # Drop repulsion args if this mace_run_train doesn't support them

  # -----------------------------
  # Debug: verify what code + what memory limit you got
  # -----------------------------
  echo "============================================================"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
  echo "RUN_DIR=$RUN_DIR"
  echo "REPULSION_WT=$REPULSION_WT"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "RUN NAME=$NAME"
  echo "PAIR_FLAGS=${PAIR_FLAGS:-<none>}"
  echo "DATA=$DATA"
  echo "E0S_JSON=$E0S_JSON"
  echo "MACE_CHECKPOINT_DIR=$MACE_CHECKPOINT_DIR"
  echo "SAVE_EVERY_FLAG=${SAVE_EVERY_FLAG:-<none-found>}"
  echo "NUM_WORKERS_FLAG=${NUM_WORKERS_FLAG:-<none-found>}"
  echo "DEFAULT_DTYPE=$DEFAULT_DTYPE  NUM_CHANNELS=$NUM_CHANNELS  BATCH_SIZE=$BATCH_SIZE"
  echo "---- scontrol memory debug ----"
  scontrol show job "$SLURM_JOB_ID" | egrep -i "JobId=|ReqMem|MinMemory|AllocTRES|TRES|NumTasks|CPUs/Task|Mem|Partition" || true
  echo "---- cgroup memory.max ----"
  ( cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true )
  echo "============================================================"

  python - <<'PY'
import os, inspect
import mace
import mace.tools.arg_parser as ap
import mace.modules.radial as r
print("mace from       :", inspect.getsourcefile(mace))
print("arg_parser from :", inspect.getsourcefile(ap))
print("radial from     :", inspect.getsourcefile(r))
print("PYTHONPATH      :", os.environ.get("PYTHONPATH"))
PY

  PYTHONPATH="$REPULSION_WT:$REPULSION_WT/mace:${PYTHONPATH:-}" python -m mace.cli.run_train --help | grep -E "pair_repulsion|pair_repulsion_kinds|pair_repulsion_mode|r12_scale|r12_cutoff|zbl_p|zbl_scale|pair_repulsion_r_min" || true

  # -----------------------------
  # Train
  # -----------------------------
  PYTHONPATH="$REPULSION_WT:$REPULSION_WT/mace:${PYTHONPATH:-}" python -m mace.cli.run_train \
    --name "$NAME" \
    --train_file "$DATA" \
    --valid_file "$DATA" \
    --test_file  "$DATA" \
    --energy_key "$EKEY" \
    --forces_key "$FKEY" \
    --E0s "$E0S_JSON" \
    --r_max "$R_MAX" \
    --max_num_epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --lr "$LR" \
    --scheduler_patience "$SCHEDULER_PATIENCE" \
    --lr_factor "$LR_FACTOR" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --pin_memory False \
    --default_dtype "$DEFAULT_DTYPE" \
    --num_channels "$NUM_CHANNELS" \
    --num_interactions "$NUM_INTERACTIONS" \
    --correlation "$CORRELATION" \
    --max_ell "$MAX_ELL" \
    --max_L "$MAX_L" \
    --radial_MLP "$RADIAL_MLP" \
    --device cuda \
    --wandb \
    --wandb_name "$NAME" \
    --wandb_project "${WANDB_PROJECT}" \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    $SAVE_EVERY_FLAG \
    $NUM_WORKERS_FLAG \
    $PAIR_FLAGS

echo "[task $TASK] Done: $NAME"