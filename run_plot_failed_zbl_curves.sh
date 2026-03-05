#!/usr/bin/env bash
#SBATCH --job-name=plot_failed_zbl
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=/scratch/sophiag/mace/outputs/dissociation_scans_overfit100/repulsion/slurm_313057/zbl/plot_failed_zbl_%j.out
#SBATCH --error=/scratch/sophiag/mace/outputs/dissociation_scans_overfit100/repulsion/slurm_313057/zbl/plot_failed_zbl_%j.err

set -euo pipefail

export MPLCONFIGDIR=/scratch/sophiag/mace/.mplconfig
mkdir -p "$MPLCONFIGDIR"

source /scratch/sophiag/mace/.macevenv/bin/activate
cd /scratch/sophiag/mace_branches/exp-repulsion-v2

python plot_failed_zbl_curves.py \
  --n 5 \
  --steps 50 \
  --failed_csv "/scratch/sophiag/mace/outputs/dissociation_scans_overfit100/repulsion/slurm_313057/zbl/failed_curves_metrics.csv"
