#!/bin/bash
#SBATCH --job-name=mace_repulsion
#SBATCH --output=outslurm/%x_%A_%a.out
#SBATCH --error=outslurm/%x_%A_%a.err
#SBATCH --time=9:00:00
#SBATCH --array=1-2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

set -euo pipefail
mkdir -p outslurm

# ============================================================
# Repulsion sweep (2 tasks): ZBL only, r^-12 only
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

# train4M H5 data
H5_ROOT="/scratch/$USER/mace/data/train4M_h5"
TRAIN_DATA="$H5_ROOT/train"
VALID_DATA="$H5_ROOT/val"
TEST_DATA="$H5_ROOT/test"
E0S_MODE="$H5_ROOT/E0s.json"
MAX_SAMPLES_PER_EPOCH=20000
MAX_VALID_SAMPLES="${MAX_VALID_SAMPLES:-20000}"
ATOMIC_NUMS="[$(python3 -c "import json; d=json.load(open('$H5_ROOT/E0s.json')); print(','.join(str(k) for k in sorted(int(x) for x in d.keys())))" 2>/dev/null)]"
EKEY="REF_energy"
FKEY="REF_forces"
VALID_FRACTION=0.0

EPOCHS=200
R_MAX=5.0
RADIAL_MLP="[64,64]"
PATIENCE=25
SEED=0
LR=0.01
SCHEDULER_PATIENCE=25
LR_FACTOR=0.8
ZBL_SCALE=1.0
R12_SCALE=1.0
R12_CUTOFF=0.8

BATCH_SIZE=64
VALID_BATCH_SIZE=64
DEFAULT_DTYPE="float32"
NUM_CHANNELS=32
NUM_INTERACTIONS=2
CORRELATION=3
MAX_ELL=3
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

# Use scratch for caches (home may be read-only on compute nodes)
CACHE_BASE="/scratch/$USER/.cache"
mkdir -p "$CACHE_BASE/wandb" "$CACHE_BASE/torch/kernels"
export WANDB_CACHE_DIR="$CACHE_BASE/wandb"
export PYTORCH_KERNEL_CACHE_PATH="$CACHE_BASE/torch/kernels"

source "$VENV_ACTIVATE"

export PYTHONNOUSERSITE=1
# Repo root contains the ``mace`` package (editable layout).
export PYTHONPATH="$REPULSION_WT${PYTHONPATH:+:$PYTHONPATH}"

# -----------------------------
# Task mapping (array id)
# -----------------------------
TASK="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "$TASK" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with: sbatch --array=1-2 repulsion_2.sh"
  exit 1
fi

NAME=""
PAIR_FLAGS=""

case "$TASK" in
  1)
    NAME="repulsion_zbl"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds zbl --zbl_scale $ZBL_SCALE"
    ;;
  2)
    NAME="repulsion_r12"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds r12 --r12_scale $R12_SCALE --r12_cutoff $R12_CUTOFF"
    ;;
  *)
    echo "Unknown task id: $TASK (expected 1-2)"
    exit 1
    ;;
esac

  # -----------------------------
  # Run directory
  # -----------------------------
  RUN_GROUP="${RUN_GROUP:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-job}}}"
  RESTART_INDEX="${RESTART_INDEX:-1}"
  mkdir -p "$RUNS_ROOT"
  case "$TASK" in
    1) RUN_SUFFIX="zbl${ZBL_SCALE}" ;;
    2) RUN_SUFFIX="r12${R12_SCALE}" ;;
  esac
  PARAMS_TAG="lr${LR}_ep${EPOCHS}"
  [ -n "${MAX_SAMPLES_PER_EPOCH}" ] && PARAMS_TAG="${PARAMS_TAG}_samp${MAX_SAMPLES_PER_EPOCH}"
  [ -n "${MAX_VALID_SAMPLES}" ] && PARAMS_TAG="${PARAMS_TAG}_val${MAX_VALID_SAMPLES}"
  RUN_TAG="train4M_${PARAMS_TAG}_${RUN_SUFFIX}"
  RUN_DIR="$RUNS_ROOT/$RUN_TAG"
  mkdir -p "$RUN_DIR"

  cd "$RUN_DIR"
  # Only wipe on first run when no checkpoints exist; preserve checkpoints for restarts
  HAS_CHECKPOINTS=false
  if [ -d ./checkpoints ] && ls ./checkpoints/*.pt 1>/dev/null 2>&1; then
    HAS_CHECKPOINTS=true
  fi
  if [ "$RESTART_INDEX" -eq 1 ] && [ "$HAS_CHECKPOINTS" = false ]; then
    rm -rf ./checkpoints ./logs ./results ./wandb 2>/dev/null || true
  fi
  mkdir -p ./checkpoints ./logs ./results
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
  if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE}" != "offline" ]; then
    export WANDB_MODE="offline"
  fi

  export WANDB_PROJECT="${WANDB_PROJECT:-mace}"
  export WANDB_ENTITY="${WANDB_ENTITY:-}"

  SAVE_EVERY_FLAG=""
  NUM_WORKERS_FLAG="--num_workers 0"

  HELP_TEXT="$(python -m mace.cli.run_train --help 2>&1 || true)"
  if echo "$HELP_TEXT" | grep -q -- "--save_every"; then
    SAVE_EVERY_FLAG="--save_every 10"
  elif echo "$HELP_TEXT" | grep -q -- "--checkpoint_interval"; then
    SAVE_EVERY_FLAG="--checkpoint_interval 10"
  elif echo "$HELP_TEXT" | grep -q -- "--save_interval"; then
    SAVE_EVERY_FLAG="--save_interval 10"
  fi

  if ! echo "$HELP_TEXT" | grep -q -- "--num_workers"; then
    NUM_WORKERS_FLAG=""
  fi

  echo "RUN_DIR=$RUN_DIR task=$TASK name=$NAME"

  RUN_CMD="python -m mace.cli.run_train"
  $RUN_CMD \
    --name "$NAME" \
    --work_dir "$RUN_DIR" \
    --checkpoints_dir "$RUN_DIR/checkpoints" \
    --train_file "$TRAIN_DATA" \
    --valid_file "$VALID_DATA" \
    --test_file  "$TEST_DATA" \
    --energy_key "$EKEY" \
    --forces_key "$FKEY" \
    --valid_fraction "$VALID_FRACTION" \
    --E0s "$E0S_MODE" \
    ${ATOMIC_NUMS:+--atomic_numbers "$ATOMIC_NUMS"} \
    ${MAX_SAMPLES_PER_EPOCH:+--max_samples_per_epoch "$MAX_SAMPLES_PER_EPOCH"} \
    --log_dir "$RUN_DIR/logs" \
    --results_dir "$RUN_DIR/results" \
    --r_max "$R_MAX" \
    --max_num_epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --lr "$LR" \
    --scheduler_patience "$SCHEDULER_PATIENCE" \
    --lr_factor "$LR_FACTOR" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --valid_batch_size "$VALID_BATCH_SIZE" \
    --max_valid_samples "$MAX_VALID_SAMPLES" \
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
    $( { [ "$RESTART_INDEX" -gt 1 ] || [ "$HAS_CHECKPOINTS" = true ]; } && echo "--restart_latest" ) \
    $SAVE_EVERY_FLAG \
    $NUM_WORKERS_FLAG \
    $PAIR_FLAGS

echo "done task=$TASK $NAME"

  RESTARTS_LEFT="${RESTARTS_LEFT:-1}"
  if [ "$RESTARTS_LEFT" -gt 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    NEXT_RESTARTS_LEFT=$((RESTARTS_LEFT - 1))
    NEXT_RESTART_INDEX=$((RESTART_INDEX + 1))
    echo "resubmit RESTART_INDEX=$NEXT_RESTART_INDEX RESTARTS_LEFT=$NEXT_RESTARTS_LEFT"
    sbatch --array="$TASK" --export=ALL,RUN_GROUP="$RUN_GROUP",RESTART_INDEX="$NEXT_RESTART_INDEX",RESTARTS_LEFT="$NEXT_RESTARTS_LEFT" "$0"
  fi