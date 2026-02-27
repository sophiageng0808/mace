#!/bin/bash
#SBATCH --job-name=mace_softcore_cutoff
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
BASE_REPO="/h/400/sophiageng/mace"
SOFTCAP_WT="/h/400/sophiageng/mace_worktrees/softcap"
RUNS_ROOT="/h/400/sophiageng/mace_worktrees/jobs_softcap"

VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

DATA="$BASE_REPO/data/overfit100_E0sub.extxyz"
E0S_JSON="$BASE_REPO/data/overfit100_E0zeros.json"
EKEY="REF_energy"
FKEY="REF_forces"

EPOCHS=400
PATIENCE=50
SEED=0

BATCH_SIZE=10
DEFAULT_DTYPE="float32"

# Threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Activate venv
source "$VENV_ACTIVATE"

# Ensure correct code import
export PYTHONPATH="$SOFTCAP_WT:${PYTHONPATH:-}"

for TASK in 0 1 2 3; do
  # -----------------------------
  # Task mapping
  # -----------------------------
  NAME=""
  SOFTCAP_KIND=""
  SOFTCAP_EPS="1e-3"
  SOFTCAP_CONCAT="True"
  SOFTCAP_APPLY_ENV="True"

  case "$TASK" in
    0)
      NAME="softcap_none"
      SOFTCAP_KIND="none"
      ;;
    1)
      NAME="softcap_inv_softplus"
      SOFTCAP_KIND="inv_softplus"
      ;;
    2)
      NAME="softcap_quadratic"
      SOFTCAP_KIND="quadratic"
      ;;
    3)
      NAME="softcap_log_radial"
      SOFTCAP_KIND="log_radial"
      ;;
    *)
      echo "Unknown task id: $TASK"
      exit 1
      ;;
  esac

  SOFTCAP_FLAGS="--softcap_kind ${SOFTCAP_KIND} --softcap_eps ${SOFTCAP_EPS} --softcap_concat ${SOFTCAP_CONCAT} --softcap_apply_env ${SOFTCAP_APPLY_ENV}"

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

  echo "============================================================"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
  echo "RUN_DIR=$RUN_DIR"
  echo "SOFTCAP_WT=$SOFTCAP_WT"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "RUN NAME=$NAME"
  echo "SOFTCAP_KIND=$SOFTCAP_KIND"
  echo "SOFTCAP_EPS=$SOFTCAP_EPS"
  echo "SOFTCAP_CONCAT=$SOFTCAP_CONCAT"
  echo "SOFTCAP_APPLY_ENV=$SOFTCAP_APPLY_ENV"
  echo "MACE_CHECKPOINT_DIR=$MACE_CHECKPOINT_DIR"
  echo "SAVE_EVERY_FLAG=${SAVE_EVERY_FLAG:-<none-found>}"
  echo "NUM_WORKERS_FLAG=${NUM_WORKERS_FLAG:-<none-found>}"
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
    --device cuda \
    --wandb \
    --wandb_name "$NAME" \
    $SOFTCAP_FLAGS \
    $SAVE_EVERY_FLAG \
    $NUM_WORKERS_FLAG

  echo "[task $TASK] Done: $NAME"
done
