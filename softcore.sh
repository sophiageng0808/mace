#!/bin/bash
#SBATCH --job-name=mace_softcore_cutoff
#SBATCH --output=outslurm/%x_%A_%a.out
#SBATCH --error=outslurm/%x_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
mkdir -p outslurm

# -----------------------------
# Paths / env
# -----------------------------
BASE_REPO="/h/400/sophiageng/mace"
SOFTCORE_WT="/h/400/sophiageng/mace_worktrees/softcore"
RUNS_ROOT="/h/400/sophiageng/mace_worktrees/jobs_softcore"

VENV_ACTIVATE="$BASE_REPO/.macevenv/bin/activate"

DATA="$BASE_REPO/data/overfit100_E0sub.extxyz"
EKEY="REF_energy"
FKEY="REF_forces"
E0S_JSON="$BASE_REPO/data/overfit100_E0zeros.json"
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

# -----------------------------
# Task mapping
# -----------------------------
TASK="${SLURM_ARRAY_TASK_ID}"

NAME=""
CUTOFF_KIND=""
SOFTCORE_P=""
SOFTCORE_EPS=""

case "$TASK" in
  0)
    NAME="cutoff_polynomial"
    CUTOFF_KIND="polynomial"
    SOFTCORE_P="6"
    SOFTCORE_EPS="1e-3"
    ;;
  1)
    NAME="cutoff_softcore_p6_eps1e-3"
    CUTOFF_KIND="softcore"
    SOFTCORE_P="6"
    SOFTCORE_EPS="1e-3"
    ;;
  2)
    NAME="cutoff_softcore_p6_eps1e-2"
    CUTOFF_KIND="softcore"
    SOFTCORE_P="6"
    SOFTCORE_EPS="1e-2"
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
export PYTHONPATH="$SOFTCORE_WT:${PYTHONPATH:-}"

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
echo "CODE_DIR=$SOFTCORE_WT"
echo "RUN NAME=$NAME"
echo "CUTOFF_KIND=$CUTOFF_KIND"
echo "SOFTCORE_P=$SOFTCORE_P"
echo "SOFTCORE_EPS=$SOFTCORE_EPS"
echo "CHECKPOINT_DIR=$RUN_DIR/checkpoints"
echo "SAVE_EVERY_FLAG=${SAVE_EVERY_FLAG:-<none-found>}"
echo "============================================================"

python - <<'PY'
import os, inspect
import mace
import mace.modules.radial as r
print("mace from           =", inspect.getsourcefile(mace))
print("radial from         =", inspect.getsourcefile(r))
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
  --batch_size 10 \
  --default_dtype float64 \
  --device cpu \
  --cutoff_kind "$CUTOFF_KIND" \
  --softcore_cutoff_p "$SOFTCORE_P" \
  --softcore_cutoff_eps "$SOFTCORE_EPS" \
  ${SAVE_EVERY_FLAG}

echo "[task $TASK] Done: $NAME"
