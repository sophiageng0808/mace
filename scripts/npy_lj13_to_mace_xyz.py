#!/usr/bin/env python3
"""Convert flat LJ-style coordinates to extended XYZ for MACE training.

Input: either
  - a plain ``.npy`` (N, n*3) positions only — energies/forces are computed here
    with the same analytic LJ+oscillator as in this repo (numpy); or
  - a labeled ``.npz`` from ``adjoint_samplers/scripts/label_lj_npy_mace.py``
    with keys ``positions``, ``energies``, ``forces`` (and optional ``source_npy``).

Infers n_atoms = n_columns // 3. Species: sidecar ``<stem>.chemistry.json`` next to
the anchor .npy path, ``--chemistry-json``, ``--chemistry-npy``, ``MACE_AS_Z``, or ``--Z``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write


def load_atomic_numbers(
    chemistry_anchor: Path, n_atoms: int, chemistry_json: Path | None, z_single: int | None
) -> list[int]:
    """Resolve per-atom Z: explicit path > sidecar next to anchor > env MACE_AS_Z > --Z."""
    paths_try: list[Path] = []
    if chemistry_json is not None:
        paths_try.append(chemistry_json)
    stem = chemistry_anchor.name
    if stem.endswith(".npy"):
        stem = stem[: -len(".npy")]
    elif stem.endswith(".npz"):
        stem = stem[: -len(".npz")]
    paths_try.append(chemistry_anchor.parent / f"{stem}.chemistry.json")

    for p in paths_try:
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                meta = json.load(f)
            if "atomic_numbers" in meta:
                zs = [int(z) for z in meta["atomic_numbers"]]
                if len(zs) != n_atoms:
                    raise SystemExit(
                        f"{p}: atomic_numbers length {len(zs)} != n_atoms {n_atoms}"
                    )
                return zs
            if "Z" in meta:
                return [int(meta["Z"])] * n_atoms
            raise SystemExit(f"{p}: expected 'Z' or 'atomic_numbers'")

    env_z = os.environ.get("MACE_AS_Z")
    if env_z is not None:
        return [int(env_z)] * n_atoms
    if z_single is not None:
        return [int(z_single)] * n_atoms

    raise SystemExit(
        "Set species for MACE: add a sidecar <stem>.chemistry.json with "
        '{"Z": <int>} or {"atomic_numbers": [...]}, or pass --chemistry-json, '
        "--chemistry-npy (for .npz inputs), --Z, or export MACE_AS_Z."
    )


def load_positions_and_labels(
    input_path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Path]:
    """Returns flat positions (N, dim), optional energies (N,), forces (N, dim), chemistry anchor."""
    if input_path.suffix.lower() == ".npz":
        with np.load(input_path) as z:
            if "positions" not in z or "energies" not in z or "forces" not in z:
                raise SystemExit(
                    f"{input_path}: labeled .npz must contain positions, energies, forces"
                )
            pos = np.asarray(z["positions"], dtype=np.float64)
            energies = np.asarray(z["energies"], dtype=np.float64)
            forces = np.asarray(z["forces"], dtype=np.float64)
            src = str(z["source_npy"].item()) if "source_npy" in z else ""
        anchor = Path(src) if src else input_path
        return pos, energies, forces, anchor

    raw = np.load(input_path)
    if raw.ndim != 2 or raw.shape[1] % 3 != 0:
        raise SystemExit(f"Expected 2D array with n*3 columns, got {raw.shape}")
    return raw.astype(np.float64), None, None, input_path


def lj_osc_energy_forces(
    pos: np.ndarray, eps: float = 1.0, rm: float = 1.0, osc_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """pos: (N, n_atom, 3) -> energies (N,), forces (N, n_atom, 3)."""
    n = pos.shape[1]
    diff = pos[:, :, np.newaxis, :] - pos[:, np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist = np.maximum(dist, 1e-12)
    inv = 1.0 / dist
    mask = ~np.eye(n, dtype=bool)
    t6 = (rm * inv) ** 6
    t12 = t6 * t6
    pair_e = eps * (t12 - 2.0 * t6) * mask
    elj = 0.5 * pair_e.sum(axis=(1, 2))
    de_dr = eps * (-12.0 * rm**12 * inv**13 + 12.0 * rm**6 * inv**7) * mask
    u = diff / dist[..., np.newaxis]
    forces = np.zeros_like(pos)
    iu, ju = np.triu_indices(n, k=1)
    for ii, jj in zip(iu, ju):
        c = de_dr[:, ii, jj, np.newaxis] * u[:, ii, jj, :]
        forces[:, ii] -= c
        forces[:, jj] += c
    rmean = pos.mean(axis=1, keepdims=True)
    xc = pos - rmean
    eosc = 0.5 * osc_scale * np.sum(xc * xc, axis=(1, 2))
    fosc = -osc_scale * xc
    return elj + eosc, forces + fosc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_path", type=str, help=".npy positions or labeled .npz")
    p.add_argument("output_xyz", type=str)
    p.add_argument(
        "--n-particles",
        type=int,
        default=None,
        help="Override atom count; default: infer as n_columns // 3",
    )
    p.add_argument(
        "--Z",
        type=int,
        default=None,
        metavar="Z",
        help="Homonuclear atomic number (used if no chemistry json / MACE_AS_Z)",
    )
    p.add_argument(
        "--chemistry-json",
        type=str,
        default=None,
        help="JSON with 'Z' or 'atomic_numbers'",
    )
    p.add_argument(
        "--chemistry-npy",
        type=str,
        default=None,
        help="Original .npy path for sidecar chemistry.json when input is .npz",
    )
    p.add_argument("--box", type=float, default=40.0, help="Cubic cell side (Å)")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    input_path = Path(args.input_path)
    raw_flat, energies_npz, forces_npz, chemistry_anchor = load_positions_and_labels(
        input_path
    )
    if args.chemistry_npy is not None:
        chemistry_anchor = Path(args.chemistry_npy)

    if raw_flat.ndim != 2 or raw_flat.shape[1] % 3 != 0:
        raise SystemExit(f"Expected (N, n*3), got {raw_flat.shape}")

    n_particles = args.n_particles or (raw_flat.shape[1] // 3)
    if raw_flat.shape[1] != n_particles * 3:
        raise SystemExit(
            f"Columns {raw_flat.shape[1]} not divisible into {n_particles} atoms * 3"
        )

    chem_path = Path(args.chemistry_json) if args.chemistry_json else None
    atomic_numbers = load_atomic_numbers(
        chemistry_anchor, n_particles, chem_path, args.Z
    )
    symbols = [chemical_symbols[z] for z in atomic_numbers]

    if args.max_frames is not None:
        raw_flat = raw_flat[: args.max_frames]
        if energies_npz is not None:
            energies_npz = energies_npz[: args.max_frames]
            forces_npz = forces_npz[: args.max_frames]
    n_frames = raw_flat.shape[0]
    pos = raw_flat.reshape(n_frames, n_particles, 3)

    shift = np.full(3, 0.5 * args.box)
    pos = pos - pos.mean(axis=1, keepdims=True) + shift

    if energies_npz is not None:
        energies = energies_npz.astype(np.float64)
        forces = forces_npz.reshape(n_frames, n_particles, 3).astype(np.float64)
    else:
        energies, forces = lj_osc_energy_forces(pos)

    cell = np.diag(np.full(3, args.box))

    all_atoms = []
    for idx in range(n_frames):
        atoms = Atoms(numbers=atomic_numbers, positions=pos[idx], cell=cell, pbc=False)
        atoms.info["REF_energy"] = float(energies[idx])
        atoms.arrays["REF_forces"] = forces[idx].astype(np.float64)
        atoms.info["config_type"] = "Default"
        all_atoms.append(atoms)
    write(args.output_xyz, all_atoms, format="extxyz")

    src = "npz labels" if energies_npz is not None else "numpy LJ+osc"
    print(
        f"Wrote {n_frames} frames to {args.output_xyz} "
        f"(n_atoms={n_particles}, energy/forces={src}, symbols={symbols[:3]}{'...' if len(symbols) > 3 else ''})"
    )


if __name__ == "__main__":
    main()
