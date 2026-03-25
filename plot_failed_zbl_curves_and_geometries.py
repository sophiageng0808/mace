#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import write as ase_write
from ase.neighborlist import neighbor_list
import ase.data
import h5py

from mace.calculators import MACECalculator

USER = os.environ.get("USER", "unknown")
SCRATCH_MACE = Path(f"/scratch/{USER}/mace")
DATA_H5_DEFAULT = SCRATCH_MACE / "data" / "train4M_h5" / "test"

DEFAULT_STEPS = 50
DEFAULT_START = 0.7
DEFAULT_END = 0.2


def _unpack_h5_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value


def load_frames_from_h5(h5_dir: Union[str, Path]) -> List[Atoms]:
    """Load ASE Atoms from a MACE preprocessed HDF5 directory (sharded *.h5 / *.hdf5)."""
    h5_path = Path(h5_dir)
    if not h5_path.is_dir():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")
    files = sorted(h5_path.glob("*.h5")) + sorted(h5_path.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 files in {h5_dir}")

    frames = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            for batch_key in sorted(k for k in f.keys() if k.startswith("config_batch_")):
                grp = f[batch_key]
                for config_key in sorted(grp.keys(), key=lambda x: int(x.split("_")[-1])):
                    sub = grp[config_key]
                    numbers = np.asarray(sub["atomic_numbers"][()])
                    positions = np.asarray(sub["positions"][()])
                    cell = _unpack_h5_value(sub["cell"][()])
                    pbc = _unpack_h5_value(sub["pbc"][()])
                    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
                    for k, v in sub["properties"].items():
                        val = _unpack_h5_value(v[()])
                        if val is not None:
                            arr = np.asarray(val)
                            if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
                                atoms.info[k] = float(arr) if arr.size else val
                            elif arr.ndim == 2:
                                atoms.arrays[k] = arr
                    if "total_charge" in atoms.info and "charge" not in atoms.info:
                        atoms.info["charge"] = atoms.info["total_charge"]
                    if "total_spin" in atoms.info and "spin" not in atoms.info:
                        atoms.info["spin"] = atoms.info["total_spin"]
                    frames.append(atoms)
    return frames


def mol_label_to_index(mol_label: str) -> int:
    """Parse train4m_N or overfit100_N -> N (same indexing as dissociation_scan_overfit100_repulsion)."""
    for prefix in ("train4m_", "overfit100_"):
        if mol_label.startswith(prefix):
            return int(mol_label[len(prefix) :])
    raise ValueError(f"Unexpected molecule label: {mol_label} (expected train4m_N or overfit100_N)")


def load_mace_calculator(model_path: str, device: str) -> MACECalculator:
    calc = MACECalculator(
        model_paths=[model_path],
        device=device,
        default_dtype="float64",
    )
    return calc


def compute_curve(atoms, i, j, calc, dist):
    vec = atoms.positions[j] - atoms.positions[i]
    u = vec / (np.linalg.norm(vec) + 1e-12)
    energies, forces = [], []
    for d in dist:
        test = atoms.copy()
        test.positions[i] = atoms.positions[j] - u * d
        test.calc = calc
        if hasattr(calc, "results") and isinstance(calc.results, dict):
            calc.results.clear()
        energies.append(float(test.get_potential_energy()))
        f = np.asarray(test.get_forces(), float)
        forces.append(-float(np.dot(f[i], u)))
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
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins)) or 1.0
    r = 0.55 * span
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def _bond_pairs(atoms, scale=1.2, min_radius=0.2):
    zs = np.asarray(atoms.numbers, int)
    radii = np.clip(np.asarray(ase.data.covalent_radii, float)[zs], min_radius, None)
    i, j = neighbor_list("ij", atoms, scale * radii)
    if i.size == 0:
        return []
    m = i < j
    return list(zip(i[m], j[m]))


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
        ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], s=120, facecolor="none", edgecolor="red", linewidth=1.5)
        ax.scatter(pos[j, 0], pos[j, 1], pos[j, 2], s=120, facecolor="none", edgecolor="blue", linewidth=1.5)
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


def _truthy_failed(value) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in ("true", "1", "yes")


def read_failed_rows(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    if "failed" in fields:
        rows = [r for r in rows if _truthy_failed(r.get("failed"))]
    seen = set()
    out = []
    for row in rows:
        key = (row["molecule"], row["atom1"], row["atom2"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description="Plot failed ZBL curves from scan CSV using H5 structures.")
    p.add_argument("--failed_csv", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--h5_dir", type=str, default=str(DATA_H5_DEFAULT), help="MACE HDF5 shard directory")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--start", type=float, default=DEFAULT_START)
    p.add_argument("--end", type=float, default=DEFAULT_END)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--scale", type=str, default="linear", choices=["linear", "log", "symlog"])
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    failed_csv = Path(args.failed_csv)
    model_path = Path(args.model)
    h5_dir = Path(args.h5_dir)

    for path, name in [(failed_csv, "failed CSV"), (model_path, "model"), (h5_dir, "H5 directory")]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    out_dir = Path(args.out_dir) if args.out_dir else failed_csv.parent.parent / "plots_failed_zbl"
    curves_dir = out_dir / "curves"
    geom_dir = out_dir / "geometries"
    curves_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)

    rows = read_failed_rows(failed_csv)
    if not rows:
        raise RuntimeError(f"No failed rows in {failed_csv} (check 'failed' column or use unfiltered CSV)")

    frames = load_frames_from_h5(h5_dir)
    calc = load_mace_calculator(str(model_path), device)
    dist = np.linspace(args.start, args.end, int(args.steps))

    for row in rows:
        mol_label = str(row["molecule"])
        atom1, atom2 = int(row["atom1"]), int(row["atom2"])
        pair_label = str(row.get("pair_label") or f"{atom1}->{atom2}")

        idx = mol_label_to_index(mol_label)
        if idx >= len(frames):
            print(f"skip {mol_label}: idx {idx} >= nframes {len(frames)}")
            continue
        atoms = frames[idx].copy()
        n_atoms = len(atoms)
        if not (0 <= atom1 < n_atoms and 0 <= atom2 < n_atoms):
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

        plot_curve(dist, energies, forces, f"{mol_label} {pair_label}", curve_path, args.scale)
        plot_geometry(atoms_start, atoms_end, atom1, atom2, f"{mol_label} {pair_label}", geom_path)

        step_atoms = []
        for d in dist:
            step = atoms.copy()
            step.positions[atom1] = atoms.positions[atom2] - u * float(d)
            step_atoms.append(step)
        ase_write(str(steps_xyz_path), step_atoms, format="xyz")

    print(f"wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
