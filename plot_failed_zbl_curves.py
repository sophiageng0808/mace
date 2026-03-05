#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ase.io import read as ase_read

# Backward-compat for repulsion checkpoints:
# - some expect mace.modules.repulsion.ZBLBasis
# - some expect mace.modules.repulsion.PairRepulsionSwitch
import sys
import mace.modules.radial as _mace_radial


class _CompatPairRepulsionSwitch(torch.nn.Module):
    """
    Backward-compatible router: mode 0=off, 1=zbl, 2=r12, 3=both.
    """

    def __init__(self, zbl, r12, mode: int = 0):
        super().__init__()
        self.mode = int(mode)
        self.zbl = zbl
        self.r12 = r12
        if self.mode < 0 or self.mode > 3:
            raise ValueError("PairRepulsionSwitch mode must be 0(off),1(zbl),2(r12),3(both).")

    def forward(self, lengths, node_attrs, edge_index, atomic_numbers, r_max):
        n_nodes = node_attrs.shape[0]
        out = torch.zeros((n_nodes,), dtype=lengths.dtype, device=lengths.device)
        if self.mode == 0:
            return out
        if self.mode == 1:
            return self.zbl(lengths, node_attrs, edge_index, atomic_numbers, r_max)
        if self.mode == 2:
            return self.r12(lengths, node_attrs, edge_index, atomic_numbers, r_max)
        return self.zbl(lengths, node_attrs, edge_index, atomic_numbers, r_max) + self.r12(
            lengths, node_attrs, edge_index, atomic_numbers, r_max
        )


try:
    import mace.modules.repulsion as _mace_repulsion

    _mace_repulsion.ZBLBasis = _mace_radial.ZBLBasis
    _mace_repulsion.PairRepulsionSwitch = _CompatPairRepulsionSwitch
except Exception:
    _mace_radial.PairRepulsionSwitch = _CompatPairRepulsionSwitch
    sys.modules.setdefault("mace.modules.repulsion", _mace_radial)


USER = os.environ.get("USER", "unknown")
SCRATCH_MACE = Path(f"/scratch/{USER}/mace")
DATA_EXTXYZ = SCRATCH_MACE / "data" / "overfit100_E0sub.extxyz"

JOBS_ROOT = Path(f"/scratch/{USER}/mace_worktrees/job_repulsion_v2")
REPULSION_MODELS = [
    ("r12",     str(JOBS_ROOT / "313013_2_repulsion_r12" / "repulsion_r12.model")),
    ("zbl",     str(JOBS_ROOT / "313015_1_repulsion_zbl" / "repulsion_zbl.model")),
    ("zbl_r12", str(JOBS_ROOT / "312989_3_repulsion_zbl_r12" / "repulsion_zbl_r12.model")),
]

DEFAULT_STEPS = 30
DEFAULT_START = 0.7
DEFAULT_END = 0.2


def ensure_pair_repulsion_buffers(calc):
    import ase.data

    for model in getattr(calc, "models", []):
        pr = getattr(model, "pair_repulsion_fn", None)
        if pr is None:
            continue
        if not hasattr(pr, "covalent_radii"):
            covalent = torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            )
            try:
                pr.register_buffer("covalent_radii", covalent)
            except Exception:
                pr.covalent_radii = covalent
        if not hasattr(pr, "alpha"):
            alpha = torch.tensor(4.0, dtype=torch.get_default_dtype())
            try:
                pr.register_buffer("alpha", alpha)
            except Exception:
                pr.alpha = alpha


def get_r12_min_distance(calc):
    for model in getattr(calc, "models", []):
        pr = getattr(model, "pair_repulsion_fn", None)
        if pr is None:
            continue
        r12 = getattr(pr, "r12", None)
        if r12 is None:
            continue
        if hasattr(r12, "r_min"):
            try:
                return float(r12.r_min)
            except Exception:
                pass
    return None

def get_zbl_min_distance(calc):
    for model in getattr(calc, "models", []):
        pr = getattr(model, "pair_repulsion_fn", None)
        if pr is None:
            continue
        zbl = getattr(pr, "zbl", None)
        if zbl is None:
            continue
        if hasattr(zbl, "r_min"):
            try:
                return float(zbl.r_min)
            except Exception:
                pass
    return None


def load_mace_calculator(model_path: str, device: str):
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=[model_path],
        device=device,
        default_dtype="float64",
    )
    ensure_pair_repulsion_buffers(calc)
    return calc


def compute_curve(atoms, i, j, calc, dist, clip_energy=None, clip_force=None):
    vec = atoms.positions[j] - atoms.positions[i]
    u = vec / (np.linalg.norm(vec) + 1e-12)

    energies = []
    forces = []
    for d in dist:
        test = atoms.copy()
        test.positions[i] = atoms.positions[j] - u * d
        test.calc = calc
        if hasattr(calc, "results") and isinstance(calc.results, dict):
            calc.results.clear()

        E = float(test.get_potential_energy())
        F = np.asarray(test.get_forces(), float)
        Fi = F[i]
        Fi_proj = float(np.dot(Fi, u))
        energies.append(E)
        forces.append(-Fi_proj)

    energies = np.asarray(energies, float)
    forces = np.asarray(forces, float)

    # Replace non-finite values with NaN for plotting
    energies[~np.isfinite(energies)] = np.nan
    forces[~np.isfinite(forces)] = np.nan

    if clip_energy is not None:
        energies = np.clip(energies, -float(clip_energy), float(clip_energy))
    if clip_force is not None:
        forces = np.clip(forces, -float(clip_force), float(clip_force))

    return energies, forces


def enforce_monotone_nondecreasing(values):
    v = np.asarray(values, float)
    if v.size == 0:
        return v
    return np.maximum.accumulate(v)


def pick_failed_pairs(failed_csv: Path, n=5):
    df = pd.read_csv(failed_csv)
    if df.empty:
        raise RuntimeError(f"No failed rows found in {failed_csv}")
    df = df.drop_duplicates(subset=["molecule", "atom1", "atom2"])
    return df.head(int(n)).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--start", type=float, default=DEFAULT_START)
    parser.add_argument("--end", type=float, default=DEFAULT_END)
    parser.add_argument("--failed_csv", type=str, default=None,
                        help="Path to failed_curves_metrics.csv")
    parser.add_argument("--data_extxyz", type=str, default=None,
                        help="Path to extxyz dataset")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scale", type=str, default="linear",
                        choices=["linear", "log", "symlog"])
    parser.add_argument("--clip_energy", type=float, default=None)
    parser.add_argument("--clip_force", type=float, default=None)
    parser.add_argument("--invert_x", action="store_true",
                        help="Invert x-axis so start is on the left.")
    parser.add_argument("--enforce_monotone_energy", action="store_true",
                        help="Clamp energy to be monotone nondecreasing.")
    parser.add_argument("--enforce_monotone_force", action="store_true",
                        help="Clamp force to be monotone nondecreasing.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    failed_csv = Path(args.failed_csv) if args.failed_csv else (
        SCRATCH_MACE
        / "outputs"
        / "dissociation_scans_overfit100"
        / "repulsion"
        / "slurm_309754"
        / "zbl"
        / "failed_curves_metrics.csv"
    )
    if not failed_csv.exists():
        raise FileNotFoundError(f"Missing failed CSV: {failed_csv}")
    data_extxyz = Path(args.data_extxyz) if args.data_extxyz else DATA_EXTXYZ
    if not data_extxyz.exists():
        raise FileNotFoundError(f"Missing extxyz: {data_extxyz}")

    out_dir = failed_csv.parent.parent / "plots_failed_zbl"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = ase_read(str(data_extxyz), ":")
    failed = pick_failed_pairs(failed_csv, n=args.n)

    calc_map = {}
    for model_name, model_path in REPULSION_MODELS:
        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(f"Missing model file: {model_name} -> {model_path}")
        calc_map[model_name] = load_mace_calculator(str(mp), device)

    dist = np.linspace(args.start, args.end, args.steps)

    for _, row in failed.iterrows():
        mol_label = str(row["molecule"])
        atom1 = int(row["atom1"])
        atom2 = int(row["atom2"])
        pair_label = str(row.get("pair_label", f"{atom1}->{atom2}"))

        if not mol_label.startswith("overfit100_"):
            raise ValueError(f"Unexpected molecule label: {mol_label}")
        idx = int(mol_label.split("_")[1])
        atoms = frames[idx].copy()

        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        for model_name, calc in calc_map.items():
            dist_model = dist
            if model_name == "r12":
                r12_min = get_r12_min_distance(calc)
                if r12_min is not None:
                    min_dist = r12_min + 1e-3
                    dist_model = np.maximum(dist_model, min_dist)
            if model_name in ("zbl", "zbl_r12"):
                zbl_min = get_zbl_min_distance(calc)
                if zbl_min is not None:
                    min_dist = zbl_min + 1e-3
                    dist_model = np.maximum(dist_model, min_dist)

            energies, forces = compute_curve(
                atoms, atom1, atom2, calc, dist_model,
                clip_energy=args.clip_energy,
                clip_force=args.clip_force,
            )
            if args.enforce_monotone_energy:
                energies = enforce_monotone_nondecreasing(energies)
            if args.enforce_monotone_force:
                forces = enforce_monotone_nondecreasing(forces)

            axes[0].plot(dist_model, energies, marker="o", linewidth=1, label=model_name)
            axes[1].plot(dist_model, forces, marker="o", linewidth=1, label=model_name)

        axes[0].set_ylabel("Energy")
        axes[1].set_ylabel("-Fi · u")
        axes[1].set_xlabel("Distance (Ang)")
        axes[0].legend(fontsize=8)
        axes[0].set_title(f"{mol_label} {pair_label}")

        if args.scale == "symlog":
            axes[0].set_yscale("symlog", linthresh=1.0)
            axes[1].set_yscale("symlog", linthresh=1.0)
        else:
            axes[0].set_yscale(args.scale)
            axes[1].set_yscale(args.scale)

        if args.invert_x or args.start > args.end:
            axes[0].set_xlim(args.start, args.end)
            axes[1].set_xlim(args.start, args.end)

        fig.tight_layout()

        safe_pair = pair_label.replace(">", "_").replace("/", "_")
        out_path = out_dir / f"{mol_label}_{safe_pair}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
