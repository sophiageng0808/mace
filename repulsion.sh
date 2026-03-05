#!/bin/bash
#SBATCH --job-name=mace_repulsion_modes
#SBATCH --output=outslurm/%x_%A_%a.out
#SBATCH --error=outslurm/%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --array=1-3
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4

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
BASE_REPO="/scratch/sophiag/mace"
REPULSION_WT="/scratch/sophiag/mace_branches/exp-repulsion-v2"
RUNS_ROOT="/scratch/sophiag/mace_worktrees/job_repulsion_v2"

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
LR=0.001
SCHEDULER_PATIENCE=10
LR_FACTOR=0.5
ZBL_SCALE=0.1
PAIR_REP_SCALE=0.3
PAIR_REP_TRAINABLE="True"
EMA_DECAY=0.999

BATCH_SIZE=1
DEFAULT_DTYPE="float32"
# Moderate capacity to train properly without OOM
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
export WANDB_MODE="offline"

# Activate venv
source "$VENV_ACTIVATE"

# Ensure correct code import
export PYTHONPATH="$REPULSION_WT:${PYTHONPATH:-}"

# -----------------------------
# Task mapping (array id)
# -----------------------------
TASK="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "$TASK" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with: sbatch --array=3 repulsion.sh"
  exit 1
fi

NAME=""
PAIR_FLAGS=""
PAIR_REPULSION_FLAGS=""
PAIR_EMB_FLAGS="--pair_repulsion_embedding --pair_repulsion_symmetric_pair_feat True --pair_repulsion_gate cosine --pair_repulsion_r_on 0.60 --pair_repulsion_r_cut 1.20 --pair_repulsion_alpha_reg 1e-3 --pair_repulsion_alpha_min 0.1 --pair_repulsion_alpha_max 10.0"
# Tuned embedding-conditioned repulsion:
# - Disable gate (avoid mid-range suppression)
# - Narrow alpha clamp around 1.0
# - Reduce alpha MLP capacity + stronger regularization
PAIR_EMB_TUNED="--pair_repulsion_embedding --pair_repulsion_symmetric_pair_feat True --pair_repulsion_gate none --pair_repulsion_alpha_hidden_dim 16 --pair_repulsion_alpha_reg 1e-2 --pair_repulsion_alpha_min 0.8 --pair_repulsion_alpha_max 1.2"


case "$TASK" in
  0)
    NAME="baseline_norepulsion"
    ;;
  1)
    NAME="repulsion_zbl"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds zbl --pair_repulsion_mode 1 --zbl_scale $ZBL_SCALE $PAIR_EMB_TUNED"
    ;;
  2)
    NAME="repulsion_r12"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds r12 --pair_repulsion_mode 2 --r12_scale 1.0 --r12_cutoff 0.8 $PAIR_EMB_TUNED"
    ;;
  3)
    NAME="repulsion_zbl_r12"
    # IMPORTANT: your argparse choices require repeated flag
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds zbl --pair_repulsion_kinds r12 --pair_repulsion_mode 3 --zbl_scale $ZBL_SCALE --r12_scale 1.0 --r12_cutoff 0.8 $PAIR_EMB_TUNED"
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

  # -----------------------------
  # Detect checkpoint + num_workers flags
  # -----------------------------
  SAVE_EVERY_FLAG=""
  NUM_WORKERS_FLAG=""

  HELP_TEXT="$(mace_run_train --help 2>&1 || true)"
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

  if echo "$HELP_TEXT" | grep -q -- "--pair_repulsion_scale"; then
    PAIR_REPULSION_FLAGS="$PAIR_REPULSION_FLAGS --pair_repulsion_scale $PAIR_REP_SCALE"
  fi
  if echo "$HELP_TEXT" | grep -q -- "--pair_repulsion_trainable"; then
    PAIR_REPULSION_FLAGS="$PAIR_REPULSION_FLAGS --pair_repulsion_trainable $PAIR_REP_TRAINABLE"
  fi
  if [ -n "$PAIR_REPULSION_FLAGS" ]; then
    PAIR_FLAGS="$PAIR_FLAGS $PAIR_REPULSION_FLAGS"
  fi

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

  mace_run_train --help | grep -E "pair_repulsion|pair_repulsion_kinds|pair_repulsion_mode|r12_scale|r12_cutoff|zbl_p|zbl_scale|pair_repulsion_r_min|foundation_model" || true

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
    --ema \
    --ema_decay "$EMA_DECAY" \
    --wandb \
    --wandb_name "$NAME" \
    $SAVE_EVERY_FLAG \
    $NUM_WORKERS_FLAG \
    $PAIR_FLAGS

echo "[task $TASK] Done: $NAME"