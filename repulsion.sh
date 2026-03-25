#!/bin/bash
#SBATCH --job-name=mace_repulsion
#SBATCH --output=outslurm/%x_%A_%a.out
#SBATCH --error=outslurm/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=1-2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

set -euo pipefail
mkdir -p outslurm

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MACE="${BASE_REPO:-/scratch/$USER/mace}"
WT="${REPULSION_WT:-${SLURM_SUBMIT_DIR:-$ROOT}}"
RUNS="${RUNS_ROOT:-/scratch/$USER/mace_worktrees/jobs_repulsion}"
H5="/scratch/$USER/mace/data/train4M_h5"
ATOMIC_NUMS="[$(python3 -c "import json; d=json.load(open('$H5/E0s.json')); print(','.join(str(k) for k in sorted(map(int, d))))" 2>/dev/null)]"

EPOCHS="${EPOCHS:-200}"
MAX_SAMPLES_PER_EPOCH="${MAX_SAMPLES_PER_EPOCH:-20000}"
ZBL_SCALE="${ZBL_SCALE:-1.0}"
R12_SCALE="${R12_SCALE:-1.0}"
R12_CUTOFF="${R12_CUTOFF:-0.8}"

TASK="${SLURM_ARRAY_TASK_ID:-}"
[ -n "$TASK" ] || { echo "Use: sbatch --array=1-2 $0"; exit 1; }

case "$TASK" in
  1)
    NAME="repulsion_zbl_${ZBL_SCALE}"
    RUN_DIR_TAG="zbl_${ZBL_SCALE}"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds zbl --zbl_scale $ZBL_SCALE"
    ;;
  2)
    NAME="repulsion_r12_${R12_SCALE}"
    RUN_DIR_TAG="r12_${R12_SCALE}"
    PAIR_FLAGS="--pair_repulsion --pair_repulsion_kinds r12 --r12_scale $R12_SCALE --r12_cutoff $R12_CUTOFF"
    ;;
  *) echo "Unknown SLURM_ARRAY_TASK_ID=$TASK (use 1 or 2)"; exit 1 ;;
esac

for v in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS \
         VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS TORCH_NUM_THREADS; do
  export "$v=${SLURM_CPUS_PER_TASK:-4}"
done
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

mkdir -p "/scratch/$USER/.cache/wandb" "/scratch/$USER/.cache/torch/kernels"
export WANDB_CACHE_DIR="/scratch/$USER/.cache/wandb" PYTORCH_KERNEL_CACHE_PATH="/scratch/$USER/.cache/torch/kernels"

source "$MACE/.macevenv/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE="${WANDB_MODE:-offline}"
[ -f "${WANDB_ENV_FILE:-$HOME/.wandb_env}" ] && source "${WANDB_ENV_FILE:-$HOME/.wandb_env}"
[ -n "${WANDB_API_KEY:-}" ] || [ ! -f "/scratch/$USER/.wandb_api_key" ] || export WANDB_API_KEY="$(</scratch/$USER/.wandb_api_key)"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE=offline

RUN_GROUP="${RUN_GROUP:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-job}}}"
RESTART_INDEX="${RESTART_INDEX:-1}"
RUN_DIR="$RUNS/$RUN_DIR_TAG"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

HAS_CKPT=false
if [ -d ./checkpoints ] && ls ./checkpoints/*.pt >/dev/null 2>&1; then
  HAS_CKPT=true
fi
if [ "$RESTART_INDEX" -eq 1 ] && [ "$HAS_CKPT" = false ]; then
  rm -rf ./checkpoints ./logs ./results ./wandb 2>/dev/null || true
fi
mkdir -p ./checkpoints ./logs ./results
export MPLCONFIGDIR="$RUN_DIR/.mplconfig" XDG_CACHE_HOME="$RUN_DIR/.cache" FC_CACHEDIR="$RUN_DIR/.cache/fontconfig"
mkdir -p "$MPLCONFIGDIR" "$FC_CACHEDIR"

RL=()
{ [ "$RESTART_INDEX" -gt 1 ] || [ "$HAS_CKPT" = true ]; } && RL=(--restart_latest)

echo "RUN_DIR=$RUN_DIR task=$TASK $RUN_DIR_TAG ($NAME)"

python "$WT/mace/cli/run_train.py" \
  --name="$NAME" \
  --train_file="$H5/train" \
  --valid_file="$H5/val" \
  --valid_fraction=0.0 \
  --test_file="$H5/test" \
  --E0s="$H5/E0s.json" \
  --model="${MODEL:-MACE}" \
  --max_num_epochs="$EPOCHS" \
  --max_samples_per_epoch="$MAX_SAMPLES_PER_EPOCH" \
  ${ATOMIC_NUMS:+--atomic_numbers "$ATOMIC_NUMS"} \
  --patience=30 \
  --scheduler_patience=10 \
  --seed=0 \
  --default_dtype=float32 \
  --device=cuda \
  --wandb \
  --wandb_name="$NAME" \
  --wandb_project="$WANDB_PROJECT" \
  ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
  "${RL[@]}" \
  $PAIR_FLAGS

echo "done task=$TASK $NAME"

RESTARTS_LEFT="${RESTARTS_LEFT:-1}"
if [ "$RESTARTS_LEFT" -gt 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  sbatch --array="$TASK" --export=ALL,RUN_GROUP="$RUN_GROUP",RESTART_INDEX="$((RESTART_INDEX + 1))",RESTARTS_LEFT="$((RESTARTS_LEFT - 1))" "$0"
fi
