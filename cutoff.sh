#!/bin/bash
#SBATCH --job-name=mace_cutoff
#SBATCH --output=outslurm/%x_%A.out
#SBATCH --error=outslurm/%x_%A.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1


set -euo pipefail

mkdir -p outslurm

# -----------------------------
# Paths / env
# -----------------------------
BASE_REPO="/h/400/sophiageng/mace"                      # clean reference clone (main)
CUTOFF_CODE_DIR="/h/400/sophiageng/mace_worktrees/cutoff" # worktree holding cutoff plumbing
RUNS_ROOT="/h/400/sophiageng/mace_worktrees/jobs_cutoff"  # final outputs (network FS)
VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

DATA="$BASE_REPO/data/overfit100_E0sub.extxyz"
EKEY="REF_energy"
FKEY="REF_forces"
E0S_JSON="$BASE_REPO/data/overfit100_E0zeros.json"
EPOCHS=400
BATCH_SIZE=4

# Threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Activate venv
source "$VENV_ACTIVATE"

# -----------------------------
# Helper: retry wrappers for flaky FS
# -----------------------------
retry() {
  # retry <n> <cmd...>
  local n="$1"; shift
  local i=1
  until "$@"; do
    if (( i >= n )); then
      return 1
    fi
    echo "[retry $i/$n] command failed: $*" >&2
    sleep $((2*i))
    i=$((i+1))
  done
}

# -----------------------------
# Choose a *local* working directory to avoid NFS stale handles
# -----------------------------
LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/${USER}/slurm_${SLURM_JOB_ID}}"
mkdir -p "$LOCAL_SCRATCH"
mkdir -p "$RUNS_ROOT"

for TASK in 0 1 2; do
  # -----------------------------
  # Task mapping
  # -----------------------------
  NAME=""
  CUTOFF_KIND=""
  CUTOFF_ENV_N=""
  CUTOFF_ENV_EPS=""

  case "$TASK" in
    0)
      NAME="cutoff_polynomial"
      CUTOFF_KIND="polynomial"
      CUTOFF_ENV_N="1"
      CUTOFF_ENV_EPS="1e-3"
      ;;
    1)
      NAME="cutoff_cosine"
      CUTOFF_KIND="cosine"
      CUTOFF_ENV_N="1"
      CUTOFF_ENV_EPS="1e-3"
      ;;
    2)
      NAME="cutoff_poly_over_rn"
      CUTOFF_KIND="poly_over_rn"
      CUTOFF_ENV_N="2"
      CUTOFF_ENV_EPS="1e-3"
      ;;
    *)
      echo "Unknown task id: $TASK"
      exit 1
      ;;
  esac

  RUN_TAG="${SLURM_JOB_ID}_${TASK}_${NAME}"
  LOCAL_RUN_DIR="$LOCAL_SCRATCH/$RUN_TAG"
  FINAL_RUN_DIR="$RUNS_ROOT/$RUN_TAG"

  mkdir -p "$LOCAL_RUN_DIR"

  sync_back() {
    mkdir -p "$FINAL_RUN_DIR"
    rsync -a --partial --inplace "$LOCAL_RUN_DIR"/ "$FINAL_RUN_DIR"/ 2>/dev/null || true
  }
  trap sync_back EXIT

  retry 3 cd "$LOCAL_RUN_DIR"

  export PYTHONPATH="$CUTOFF_CODE_DIR:${PYTHONPATH:-}"

  # Set cutoff env vars
  export MACE_CUTOFF_KIND="$CUTOFF_KIND"
  export MACE_CUTOFF_ENV_N="$CUTOFF_ENV_N"
  export MACE_CUTOFF_ENV_EPS="$CUTOFF_ENV_EPS"

  rm -rf "./checkpoints" "./logs" "./results" "./wandb" 2>/dev/null || true
  mkdir -p "./checkpoints" "./logs" "./results"

  export MACE_CHECKPOINT_DIR="$LOCAL_RUN_DIR/checkpoints"

  SAVE_EVERY_FLAG=""
  HELP_TEXT="$(mace_run_train --help 2>&1 || true)"
  if echo "$HELP_TEXT" | grep -q -- "--save_every"; then
    SAVE_EVERY_FLAG="--save_every 10"
  elif echo "$HELP_TEXT" | grep -q -- "--checkpoint_interval"; then
    SAVE_EVERY_FLAG="--checkpoint_interval 10"
  elif echo "$HELP_TEXT" | grep -q -- "--save_interval"; then
    SAVE_EVERY_FLAG="--save_interval 10"
  else
    echo "WARNING: Could not find a checkpoint-every-N-epochs flag in mace_run_train --help."
  fi

  echo "============================================================"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
  echo "LOCAL_RUN_DIR=$LOCAL_RUN_DIR"
  echo "FINAL_RUN_DIR=$FINAL_RUN_DIR"
  echo "CODE_DIR=$CUTOFF_CODE_DIR"
  echo "RUN NAME=$NAME"
  echo "MACE_CUTOFF_KIND=$MACE_CUTOFF_KIND"
  echo "MACE_CUTOFF_ENV_N=$MACE_CUTOFF_ENV_N"
  echo "MACE_CUTOFF_ENV_EPS=$MACE_CUTOFF_ENV_EPS"
  echo "MACE_CHECKPOINT_DIR=$MACE_CHECKPOINT_DIR"
  echo "SAVE_EVERY_FLAG=${SAVE_EVERY_FLAG:-<none-found>}"
  echo "CPUS=$SLURM_CPUS_PER_TASK  MEM=${SLURM_MEM_PER_NODE:-<unknown>}"
  echo "============================================================"

  python - <<'PY'
import os, inspect
import mace
import mace.tools.model_script_utils as msu
print("PWD                  =", os.getcwd())
print("PYTHONPATH           =", os.environ.get("PYTHONPATH"))
print("mace from            =", inspect.getsourcefile(mace))
print("model_script_utils   =", inspect.getsourcefile(msu))
print("ENV cutoff kind      =", os.environ.get("MACE_CUTOFF_KIND"))
print("ENV env_n            =", os.environ.get("MACE_CUTOFF_ENV_N"))
print("ENV env_eps          =", os.environ.get("MACE_CUTOFF_ENV_EPS"))
print("MACE_CHECKPOINT_DIR  =", os.environ.get("MACE_CHECKPOINT_DIR"))
PY

  # -----------------------------
  # Train (GPU) in LOCAL_RUN_DIR
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
    --default_dtype float64 \
    --device cuda \
    --wandb \
    --wandb_name "$NAME" \
    ${SAVE_EVERY_FLAG}

  sync_back
  echo "[task $TASK] Done: $NAME"
done