#!/bin/bash
#SBATCH --job-name=plot_failed_zbl
#SBATCH --output=outslurm/%x_%j.out
#SBATCH --error=outslurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p outslurm

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MACE="${BASE_REPO:-/scratch/$USER/mace}"
RUN="${DISSOC_RUN:-$MACE/outputs/dissociation_scans_overfit100/repulsion/slurm_358187/zbl}"

FAILED_CSV="${FAILED_CSV:-$RUN/failed_curves_metrics.csv}"
MODEL_PATH="${MODEL_PATH:-/scratch/$USER/mace_worktrees/jobs_repulsion/train4m_multiday_1_repulsion_zbl/repulsion_zbl.model}"
H5_DIR="${H5_DIR:-$MACE/data/train4M_h5/test}"
OUT_DIR="${OUT_DIR:-$RUN}"

N=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N \
  NUMEXPR_NUM_THREADS=$N VECLIB_MAXIMUM_THREADS=$N BLIS_NUM_THREADS=$N TORCH_NUM_THREADS=$N
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig

source "$MACE/.macevenv/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

python plot_failed_zbl_curves_and_geometries.py \
  --failed_csv "$FAILED_CSV" --model "$MODEL_PATH" --h5_dir "$H5_DIR" \
  --out_dir "$OUT_DIR" --device cuda
