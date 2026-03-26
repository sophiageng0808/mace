#!/bin/bash
#SBATCH --job-name=mace_preprocess_h5
#SBATCH --output=outslurm/%x_%j.out
#SBATCH --error=outslurm/%x_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=compute

set -euo pipefail
mkdir -p outslurm

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACE="${MACE_REPO:-${BASE_REPO:-/scratch/${USER}/mace}}"
WT="${MACE_WORKTREE:-$ROOT}"

# Same as repulsion.sh H5=...
H5_DATA="${H5_DATA:-/scratch/${USER}/mace/data/train4M_h5}"
DATA_SPLIT_ROOT="${DATA_SPLIT_ROOT:-/scratch/${USER}/mace/data/train4M_split_full}"

TRAIN_FILE="${TRAIN_FILE:-}"
VALID_FILE="${VALID_FILE:-}"
TEST_FILE="${TEST_FILE:-}"
H5_PREFIX="${H5_PREFIX:-/scratch/${USER}/mace/data/train4M_h5_preprocessed/}"

if [[ -z "$TRAIN_FILE" && -f "$DATA_SPLIT_ROOT/train.extxyz" ]]; then
  TRAIN_FILE="$DATA_SPLIT_ROOT/train.extxyz"
fi
if [[ -z "$VALID_FILE" && -f "$DATA_SPLIT_ROOT/val.extxyz" ]]; then
  VALID_FILE="$DATA_SPLIT_ROOT/val.extxyz"
fi
if [[ -z "$TEST_FILE" && -f "$DATA_SPLIT_ROOT/test.extxyz" ]]; then
  TEST_FILE="$DATA_SPLIT_ROOT/test.extxyz"
fi

VALID_FRACTION="${VALID_FRACTION:-0.0}"
SEED="${SEED:-123}"

R_MAX="${R_MAX:-5.0}"

if [[ -z "${ATOMIC_NUMBERS:-}" ]]; then
  if [[ -f "$H5_DATA/E0s.json" ]]; then
    ATOMIC_NUMBERS="[$(python3 -c "import json; d=json.load(open('$H5_DATA/E0s.json')); print(','.join(str(k) for k in sorted(map(int, d))))")]"
  else
    ATOMIC_NUMBERS="[1, 6, 7, 8, 9, 15, 16, 17, 35, 53]"
  fi
fi

if [[ -z "${E0S:-}" ]]; then
  if [[ -f "$H5_DATA/E0s.json" ]]; then
    E0S="$H5_DATA/E0s.json"
  else
    E0S="average"
  fi
fi

NUM_PROCESS="${NUM_PROCESS:-100}"

COMPUTE_STATISTICS="${COMPUTE_STATISTICS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ -z "$TRAIN_FILE" ]]; then
  echo "error: set TRAIN_FILE or place train.extxyz under DATA_SPLIT_ROOT=$DATA_SPLIT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "error: TRAIN_FILE does not exist: $TRAIN_FILE" >&2
  exit 1
fi

PREPROCESS_PY="${WT}/mace/cli/preprocess_data.py"
if [[ ! -f "$PREPROCESS_PY" ]]; then
  PREPROCESS_PY="${MACE}/mace/cli/preprocess_data.py"
fi
if [[ ! -f "$PREPROCESS_PY" ]]; then
  echo "error: could not find preprocess_data.py under WT=$WT or MACE=$MACE" >&2
  exit 1
fi

_h5p="${H5_PREFIX%/}"
_h5d="${H5_DATA%/}"
if [[ "$_h5p" == "$_h5d" && -z "${ALLOW_OVERWRITE_H5_DATA:-}" ]]; then
  echo "error: H5_PREFIX equals H5_DATA ($_h5d); refusing to overwrite repulsion dataset." >&2
  echo "  Use default H5_PREFIX (train4M_h5_preprocessed), another directory, or ALLOW_OVERWRITE_H5_DATA=1" >&2
  exit 1
fi

mkdir -p "${H5_PREFIX}"

for v in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS \
         VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS TORCH_NUM_THREADS; do
  export "$v=${SLURM_CPUS_PER_TASK:-32}"
done

if [[ -f "${MACE}/.macevenv/bin/activate" ]]; then

  source "${MACE}/.macevenv/bin/activate"
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${WT}${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  python "$PREPROCESS_PY"
  --train_file="$TRAIN_FILE"
  --valid_fraction="$VALID_FRACTION"
  --r_max="$R_MAX"
  --h5_prefix="$H5_PREFIX"
  --atomic_numbers="$ATOMIC_NUMBERS"
  --E0s="$E0S"
  --seed="$SEED"
  --num_process="$NUM_PROCESS"
  --batch_size="$BATCH_SIZE"
)

if [[ -n "$VALID_FILE" ]]; then
  CMD+=(--valid_file="$VALID_FILE")
fi
if [[ -n "$TEST_FILE" ]]; then
  if [[ ! -f "$TEST_FILE" ]]; then
    echo "error: TEST_FILE does not exist: $TEST_FILE" >&2
    exit 1
  fi
  CMD+=(--test_file="$TEST_FILE")
fi
if [[ "$COMPUTE_STATISTICS" == "1" || "$COMPUTE_STATISTICS" == "true" ]]; then
  CMD+=(--compute_statistics)
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
