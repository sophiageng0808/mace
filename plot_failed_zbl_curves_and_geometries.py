#!/usr/bin/env python3
"""
Plot failed ZBL dissociation curves and geometries for train4m scan results.
Adapted from exp-repulsion for train4m molecule labels (train4m_N = index N in data extxyz).
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ase.io import read as ase_read, write as ase_write
from ase.neighborlist import neighbor_list
import ase.data

import mace.modules.repulsion  # noqa: F401


USER = os.environ.get("USER", "unknown")
SCRATCH_MACE = Path(f"/scratch/{USER}/mace")
# train4m scan uses train.extxyz by default; override with --data_extxyz if scan used test.extxyz
DATA_EXTXYZ_DEFAULT = SCRATCH_MACE / "data" / "train4M_split_25k" / "train.extxyz"

DEFAULT_STEPS = 50
DEFAULT_START = 0.7
DEFAULT_END = 0.2


def ensure_pair_repulsion_buffers(calc):
    for model in getattr(calc, "models", []):
        pr = getattr(model, "pair_repulsion_fn", None)
        if pr is None:
            continue
        if not hasattr(pr, "covalent_radii"):
            covalent = torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            )
            pr.register_buffer("covalent_radii", covalent)
        if not hasattr(pr, "alpha"):
            alpha = torch.tensor(4.0, dtype=torch.get_default_dtype())
            pr.register_buffer("alpha", alpha)


def load_mace_calculator(model_path: str, device: str):
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=[model_path],
        device=device,
        default_dtype="float64",
    )
    ensure_pair_repulsion_buffers(calc)
    return calc


def compute_curve(atoms, i, j, calc, dist):
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

    energies[~np.isfinite(energies)] = np.nan
    forces[~np.isfinite(forces)] = np.nan

    return energies, forces


def plot_curve(dist, energies, forces, title, out_path, scale):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(dist, energies, marker="o", linewidth=1)
    axes[1].plot(dist, forces, marker="o", linewidth=1)
    axes[0].set_ylabel("Energy")
    axes[1].set_ylabel("-Fi · u")
    axes[1].set_xlabel("Distance (Ang)")
    axes[0].set_title(title)

    if scale == "symlog":
        axes[0].set_yscale("symlog", linthresh=1.0)
        axes[1].set_yscale("symlog", linthresh=1.0)
    else:
        axes[0].set_yscale(scale)
        axes[1].set_yscale(scale)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _set_equal_3d(ax, xyz):
    xyz = np.asarray(xyz, float)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    r = 0.55 * span
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def _bond_pairs(atoms, scale=1.2, min_radius=0.2):
    zs = np.asarray(atoms.numbers, int)
    radii = np.asarray(ase.data.covalent_radii, float)[zs]
    radii = np.clip(radii, min_radius, None)
    cutoffs = scale * radii
    i, j = neighbor_list("ij", atoms, cutoffs)
    if i.size == 0:
        return []
    mask = i < j
    return list(zip(i[mask], j[mask]))


def plot_geometry(atoms_start, atoms_end, i, j, title, out_path):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, atoms, label in [(ax1, atoms_start, "start"), (ax2, atoms_end, "end")]:
        pos = np.asarray(atoms.positions, float)
        zs = np.asarray(atoms.numbers, int)
        colors = ase.data.colors.jmol_colors[zs]
        for bi, bj in _bond_pairs(atoms):
            ax.plot(
                [pos[bi, 0], pos[bj, 0]],
                [pos[bi, 1], pos[bj, 1]],
                [pos[bi, 2], pos[bj, 2]],
                color="gray",
                linewidth=0.9,
                alpha=0.9,
            )
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=60, c=colors, edgecolor="k", linewidth=0.3)
        ax.scatter(
            pos[i, 0], pos[i, 1], pos[i, 2],
            s=120, facecolor="none", edgecolor="red", linewidth=1.5
        )
        ax.scatter(
            pos[j, 0], pos[j, 1], pos[j, 2],
            s=120, facecolor="none", edgecolor="blue", linewidth=1.5
        )
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            [pos[i, 2], pos[j, 2]],
            color="dimgray",
            linewidth=1.4,
        )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        _set_equal_3d(ax, pos)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _mol_label_to_idx(mol_label: str) -> int:
    """Parse train4m_N or overfit100_N -> index N."""
    if mol_label.startswith("train4m_"):
        return int(mol_label.split("_")[1])
    if mol_label.startswith("overfit100_"):
        return int(mol_label.split("_")[1])
    raise ValueError(f"Unexpected molecule label: {mol_label} (expected train4m_N or overfit100_N)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot failed ZBL curves and geometries for train4m dissociation scan."
    )
    parser.add_argument("--failed_csv", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_extxyz", type=str, default=str(DATA_EXTXYZ_DEFAULT))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--start", type=float, default=DEFAULT_START)
    parser.add_argument("--end", type=float, default=DEFAULT_END)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scale", type=str, default="linear",
                        choices=["linear", "log", "symlog"])
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    failed_csv = Path(args.failed_csv)
    model_path = Path(args.model)
    data_extxyz = Path(args.data_extxyz)

    if not failed_csv.exists():
        raise FileNotFoundError(f"Missing failed CSV: {failed_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not data_extxyz.exists():
        raise FileNotFoundError(f"Missing extxyz: {data_extxyz}")

    out_dir = Path(args.out_dir) if args.out_dir else failed_csv.parent.parent / "plots_failed_zbl"
    curves_dir = out_dir / "curves"
    geom_dir = out_dir / "geometries"
    curves_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(failed_csv)
    if df.empty:
        raise RuntimeError(f"No rows found in {failed_csv}")
    if "failed" in df.columns:
        df = df[df["failed"] == True]
    df = df.drop_duplicates(subset=["molecule", "atom1", "atom2"]).reset_index(drop=True)

    frames = ase_read(str(data_extxyz), ":")
    calc = load_mace_calculator(str(model_path), device)

    dist = np.linspace(args.start, args.end, int(args.steps))

    for _, row in df.iterrows():
        mol_label = str(row["molecule"])
        atom1 = int(row["atom1"])
        atom2 = int(row["atom2"])
        pair_label = str(row.get("pair_label", f"{atom1}->{atom2}"))

        idx = _mol_label_to_idx(mol_label)
        if idx >= len(frames):
            print(f"skip {mol_label}: idx {idx} >= nframes {len(frames)}")
            continue
        atoms = frames[idx].copy()

        n_atoms = len(atoms)
        if atom1 < 0 or atom1 >= n_atoms or atom2 < 0 or atom2 >= n_atoms:
            print(f"skip {mol_label}: bad atom indices (n={n_atoms})")
            continue

        vec = atoms.positions[atom2] - atoms.positions[atom1]
        u = vec / (np.linalg.norm(vec) + 1e-12)
        atoms_start = atoms.copy()
        atoms_end = atoms.copy()
        atoms_start.positions[atom1] = atoms.positions[atom2] - u * float(args.start)
        atoms_end.positions[atom1] = atoms.positions[atom2] - u * float(args.end)

        energies, forces = compute_curve(atoms, atom1, atom2, calc, dist)

        safe_pair = pair_label.replace(">", "_").replace("/", "_")
        curve_path = curves_dir / f"{mol_label}_{safe_pair}_curve.png"
        geom_path = geom_dir / f"{mol_label}_{safe_pair}_geom.png"
        steps_xyz_path = geom_dir / f"{mol_label}_{safe_pair}_steps.xyz"

        plot_curve(
            dist,
            energies,
            forces,
            title=f"{mol_label} {pair_label}",
            out_path=curve_path,
            scale=args.scale,
        )
        plot_geometry(
            atoms_start,
            atoms_end,
            atom1,
            atom2,
            title=f"{mol_label} {pair_label}",
            out_path=geom_path,
        )

        step_atoms = []
        for d in dist:
            step = atoms.copy()
            step.positions[atom1] = atoms.positions[atom2] - u * float(d)
            step_atoms.append(step)
        ase_write(str(steps_xyz_path), step_atoms, format="xyz")

    print(f"wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
