#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from ase.io import read as ase_read


def _parse_float_list(text: str) -> List[float]:
    if text is None or text == "":
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_pair(text: str) -> Optional[Tuple[int, int]]:
    if text is None or text == "":
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("pair must be 'i,j'")
    return int(parts[0]), int(parts[1])


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv_row(path: Path, header: List[str], row: List) -> None:
    _ensure_parent(path)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def _load_frames(extxyz_path: str) -> List:
    frames = ase_read(extxyz_path, ":")
    if not isinstance(frames, list):
        frames = [frames]
    return frames


def _load_mace_calculator(model_path: str, device: str):
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=[model_path],
        device=device,
        default_dtype="float64",
    )
    _ensure_pair_repulsion_buffers(calc)
    return calc


def _ensure_pair_repulsion_buffers(calc):
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


def _get_energy_forces(calc, atoms) -> Tuple[float, np.ndarray]:
    test = atoms.copy()
    test.info = dict(getattr(test, "info", {}) or {})
    test.calc = calc
    if hasattr(calc, "results") and isinstance(calc.results, dict):
        calc.results.clear()
    energy = float(test.get_potential_energy())
    forces = np.asarray(test.get_forces(), float)
    return energy, forces


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    # Uniform random rotation via quaternion
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    q = np.array([q1, q2, q3, q4], float)
    q1, q2, q3, q4 = q
    return np.array(
        [
            [1 - 2 * (q3**2 + q4**2), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2**2 + q4**2), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2**2 + q3**2)],
        ],
        float,
    )


def _pick_pair(atoms, pair_override: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    nat = len(atoms)
    if nat < 2:
        raise ValueError("Need at least 2 atoms for pair selection")
    if pair_override is not None:
        i, j = pair_override
        if i < 0 or j < 0 or i >= nat or j >= nat or i == j:
            raise ValueError("Invalid pair indices")
        return i, j
    positions = np.asarray(atoms.positions, float)
    dmat = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    dmat += np.eye(nat) * 1e9
    i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
    return int(i), int(j)


def _fd_force_check(
    calc,
    atoms,
    atom_index: int,
    direction: np.ndarray,
    hs: Iterable[float],
) -> Dict[float, Dict[str, float]]:
    direction = np.asarray(direction, float)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    base_E, base_F = _get_energy_forces(calc, atoms)
    base_comp = float(np.dot(base_F[atom_index], direction))
    out = {}
    for h in hs:
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions[atom_index] = plus.positions[atom_index] + direction * h
        minus.positions[atom_index] = minus.positions[atom_index] - direction * h
        e_plus, _ = _get_energy_forces(calc, plus)
        e_minus, _ = _get_energy_forces(calc, minus)
        f_fd = -(e_plus - e_minus) / (2.0 * h)
        denom = max(abs(base_comp), abs(f_fd), 1e-12)
        rel_err = abs(f_fd - base_comp) / denom
        out[float(h)] = {
            "E": float(base_E),
            "F_model": float(base_comp),
            "F_fd": float(f_fd),
            "rel_err": float(rel_err),
        }
    return out


def _translation_invariance(calc, atoms, shift: np.ndarray) -> Dict[str, float]:
    base_E, base_F = _get_energy_forces(calc, atoms)
    moved = atoms.copy()
    moved.positions = moved.positions + shift[None, :]
    moved_E, moved_F = _get_energy_forces(calc, moved)
    e_diff = abs(moved_E - base_E)
    f_diff = float(np.max(np.abs(moved_F - base_F)))
    return {"energy_diff": float(e_diff), "force_max_diff": f_diff}


def _rotation_invariance(calc, atoms, rot: np.ndarray) -> Dict[str, float]:
    base_E, base_F = _get_energy_forces(calc, atoms)
    moved = atoms.copy()
    moved.positions = np.dot(moved.positions, rot.T)
    moved_E, moved_F = _get_energy_forces(calc, moved)
    e_diff = abs(moved_E - base_E)
    rotated_F = np.dot(base_F, rot.T)
    f_diff = float(np.max(np.abs(moved_F - rotated_F)))
    return {"energy_diff": float(e_diff), "force_max_diff": f_diff}


def _permutation_invariance(calc, atoms, rng: np.random.Generator) -> Optional[Dict[str, float]]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    unique, counts = np.unique(symbols, return_counts=True)
    candidates = [u for u, c in zip(unique, counts) if c >= 2]
    if not candidates:
        return None
    sym = rng.choice(candidates)
    idx = np.where(symbols == sym)[0]
    i, j = rng.choice(idx, size=2, replace=False)
    base_E, base_F = _get_energy_forces(calc, atoms)
    moved = atoms.copy()
    tmp = moved.positions[i].copy()
    moved.positions[i] = moved.positions[j]
    moved.positions[j] = tmp
    moved_E, moved_F = _get_energy_forces(calc, moved)
    e_diff = abs(moved_E - base_E)
    # Permute forces back to compare
    perm_F = moved_F.copy()
    perm_F[i], perm_F[j] = perm_F[j].copy(), perm_F[i].copy()
    f_diff = float(np.max(np.abs(perm_F - base_F)))
    return {"energy_diff": float(e_diff), "force_max_diff": f_diff}


def _loop_work(calc, atoms, i: int, j: int, delta: float) -> float:
    work = 0.0
    cur = atoms.copy()

    for step in range(4):
        _, forces = _get_energy_forces(calc, cur)
        if step == 0:
            dr = np.array([delta, 0.0, 0.0])
            work += float(np.dot(forces[i], dr))
            cur.positions[i] = cur.positions[i] + dr
        elif step == 1:
            dr = np.array([0.0, delta, 0.0])
            work += float(np.dot(forces[j], dr))
            cur.positions[j] = cur.positions[j] + dr
        elif step == 2:
            dr = np.array([-delta, 0.0, 0.0])
            work += float(np.dot(forces[i], dr))
            cur.positions[i] = cur.positions[i] + dr
        else:
            dr = np.array([0.0, -delta, 0.0])
            work += float(np.dot(forces[j], dr))
            cur.positions[j] = cur.positions[j] + dr
    return float(work)


def _enforce_bond_length(positions: np.ndarray, i: int, j: int, target: float) -> np.ndarray:
    pos = np.asarray(positions, float).copy()
    midpoint = 0.5 * (pos[i] + pos[j])
    vec = pos[j] - pos[i]
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return pos
    u = vec / n
    pos[i] = midpoint - 0.5 * target * u
    pos[j] = midpoint + 0.5 * target * u
    return pos


def _jiggle_fixed_bond(atoms, i: int, j: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    pos = np.asarray(atoms.positions, float)
    d0 = float(np.linalg.norm(pos[j] - pos[i]))
    noise = rng.normal(scale=scale, size=pos.shape)
    pos_jiggle = pos + noise
    pos_jiggle = _enforce_bond_length(pos_jiggle, i, j, d0)
    return pos_jiggle


def _scan_along_direction(
    calc,
    atoms,
    i: int,
    j: int,
    distances: List[float],
    direction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction, float)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    energies = []
    forces_proj = []
    for d in distances:
        test = atoms.copy()
        test.positions[i] = test.positions[j] - direction * d
        e, f = _get_energy_forces(calc, test)
        energies.append(e)
        forces_proj.append(float(np.dot(f[i], direction)))
    return np.asarray(energies, float), np.asarray(forces_proj, float)


def _zbl_potential(z1: int, z2: int, r: float) -> float:
    # ZBL screening length (Angstrom) and potential (eV) with standard coefficients
    # Using common parameterization: phi(x) = sum a_i * exp(-b_i * x)
    a = np.array([0.1818, 0.5099, 0.2802, 0.02817], float)
    b = np.array([3.2, 0.9423, 0.4029, 0.2016], float)
    a0 = 0.529177  # Bohr in Angstrom
    a_screen = 0.88534 * a0 / (z1**0.23 + z2**0.23)
    x = r / a_screen
    phi = float(np.sum(a * np.exp(-b * x)))
    # Coulomb prefactor in eV*Angstrom
    return (14.399645 * z1 * z2 / r) * phi


def _short_range_baseline(atoms, i: int, j: int, distances: List[float]) -> Dict[str, float]:
    z1 = int(atoms[i].number)
    z2 = int(atoms[j].number)
    zbl_vals = []
    for d in distances:
        if d <= 1e-6:
            zbl_vals.append(np.inf)
        else:
            zbl_vals.append(_zbl_potential(z1, z2, d))
    zbl_vals = np.asarray(zbl_vals, float)
    monotone = _monotone_repulsion(distances, zbl_vals)
    return {"zbl_monotone": monotone}


def _monotone_repulsion(distances: List[float], energies: np.ndarray) -> bool:
    if len(distances) < 2:
        return True
    if distances[1] > distances[0]:
        return bool(np.all(np.diff(energies) <= 1e-8))
    return bool(np.all(np.diff(energies) >= -1e-8))


def _find_ref_energy_forces(atoms, energy_key: Optional[str], forces_key: Optional[str]):
    ref_E = None
    ref_F = None
    info = dict(getattr(atoms, "info", {}) or {})
    arrays = dict(getattr(atoms, "arrays", {}) or {})

    if energy_key is not None:
        ref_E = info.get(energy_key, None)
    else:
        for key in ("energy", "E", "ref_energy", "REF_energy"):
            if key in info:
                ref_E = info[key]
                break

    if forces_key is not None:
        ref_F = arrays.get(forces_key, None)
    else:
        for key in ("forces", "F", "ref_forces", "REF_forces"):
            if key in arrays:
                ref_F = arrays[key]
                break
    if ref_E is not None:
        ref_E = float(ref_E)
    if ref_F is not None:
        ref_F = np.asarray(ref_F, float)
    return ref_E, ref_F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model paths or glob")
    parser.add_argument("--data", type=str, default=None, help="extxyz file with structures")
    parser.add_argument("--ref-scan", type=str, default=None, help="extxyz with reference scan frames")
    parser.add_argument("--ref-energy-key", type=str, default=None)
    parser.add_argument("--ref-forces-key", type=str, default=None)
    parser.add_argument("--n-structures", type=int, default=6)
    parser.add_argument("--short-range-max", type=float, default=0.35)
    parser.add_argument("--pair", type=str, default=None, help="Override pair indices i,j")
    parser.add_argument("--hs", type=str, default="1e-4,3e-4,1e-3")
    parser.add_argument("--scan-distances", type=str, default="0.30,0.25,0.20")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Default: auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--loop-delta", type=float, default=1e-3)
    parser.add_argument("--jiggle", type=float, default=0.02)
    parser.add_argument("--translate-mag", type=float, default=0.5)
    parser.add_argument("--off-axis-angle-deg", type=float, default=10.0)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="mace-consistency-checks")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="offline")
    parser.add_argument("--preflight", action="store_true", help="Run a quick compile/run check and exit")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    models_arg = args.models
    model_paths: List[str] = []
    for part in [p.strip() for p in models_arg.split(",") if p.strip()]:
        if "*" in part or "?" in part or "[" in part:
            model_paths.extend([str(p) for p in Path().glob(part)])
        else:
            model_paths.append(part)
    model_paths = [str(Path(p).resolve()) for p in model_paths]
    if len(model_paths) == 0:
        raise FileNotFoundError("No model files matched --models")

    data_path = args.data
    if data_path is None:
        user = os.environ.get("USER", "unknown")
        default_path = Path(f"/scratch/{user}/mace/data/overfit100_E0sub.extxyz")
        if default_path.exists():
            data_path = str(default_path)
    if data_path is None:
        raise FileNotFoundError("Provide --data extxyz for structures")

    frames = _load_frames(data_path)
    pair_override = _parse_int_pair(args.pair)
    hs = _parse_float_list(args.hs)
    scan_distances = _parse_float_list(args.scan_distances)
    if len(hs) == 0:
        raise ValueError("--hs must contain at least one value")
    if len(scan_distances) < 2:
        raise ValueError("Need at least 2 scan distances")

    outdir = Path(args.outdir) if args.outdir else Path("./model_consistency_checks")
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Select structures with short-range contacts if possible
    if args.n_structures <= 0:
        selected = list(enumerate(frames))
    else:
        selected = []
        selected_idx = set()
        for idx, at in enumerate(frames):
            pos = np.asarray(at.positions, float)
            if len(at) < 2:
                continue
            dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            dmat += np.eye(len(at)) * 1e9
            if np.min(dmat) <= args.short_range_max:
                selected.append((idx, at))
                selected_idx.add(idx)
        if len(selected) < args.n_structures:
            for idx, at in enumerate(frames):
                if idx in selected_idx:
                    continue
                selected.append((idx, at))
                selected_idx.add(idx)
                if len(selected) >= args.n_structures:
                    break
        else:
            selected = selected[: args.n_structures]
    selected_count = len(selected)

    ref_frames = _load_frames(args.ref_scan) if args.ref_scan else None

    use_wandb = bool(args.wandb)
    wandb_run = None
    if use_wandb:
        import wandb

        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        os.environ.setdefault("WANDB_DIR", str(outdir / "wandb"))
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            config={
                "models": model_paths,
                "data": data_path,
                "ref_scan": args.ref_scan,
                "n_structures": selected_count,
                "short_range_max": args.short_range_max,
                "pair": args.pair,
                "hs": hs,
                "scan_distances": scan_distances,
                "device": device,
                "seed": args.seed,
                "loop_delta": args.loop_delta,
                "jiggle": args.jiggle,
                "translate_mag": args.translate_mag,
                "off_axis_angle_deg": args.off_axis_angle_deg,
                "outdir": str(outdir),
            },
        )

    fd_csv = outdir / "fd_force_checks.csv"
    inv_csv = outdir / "invariance_checks.csv"
    loop_csv = outdir / "loop_work_checks.csv"
    ood_csv = outdir / "ood_jiggle_checks.csv"
    approach_csv = outdir / "approach_direction_checks.csv"
    ref_csv = outdir / "reference_compare.csv"
    summary_csv = outdir / "summary_metrics.csv"

    for model_path in model_paths:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        calc = _load_mace_calculator(model_path, device)
        model_name = Path(model_path).stem

        if args.preflight:
            frame_idx, atoms = selected[0]
            i, j = _pick_pair(atoms, pair_override)
            _get_energy_forces(calc, atoms)
            _fd_force_check(calc, atoms, i, np.array([1.0, 0.0, 0.0], float), [hs[0]])
            print(f"[OK] Preflight succeeded for model {model_name} on frame {frame_idx}")
            if use_wandb and wandb_run is not None:
                wandb_run.finish()
            return

        acc = {
            "fd_rel_err": [],
            "trans_energy_diff": [],
            "trans_force_diff": [],
            "rot_energy_diff": [],
            "rot_force_diff": [],
            "perm_energy_diff": [],
            "perm_force_diff": [],
            "loop_work": [],
            "jiggle_energy_std": [],
            "jiggle_force_std": [],
            "axis_monotone": [],
            "off_monotone": [],
        }

        for frame_idx, atoms in selected:
            i, j = _pick_pair(atoms, pair_override)
            direction = np.array([1.0, 0.0, 0.0], float)

            # 1A FD force check
            fd = _fd_force_check(calc, atoms, i, direction, hs)
            for h, stats in fd.items():
                acc["fd_rel_err"].append(stats["rel_err"])
                _write_csv_row(
                    fd_csv,
                    ["model", "frame", "atom", "h", "energy", "F_model", "F_fd", "rel_err"],
                    [model_name, frame_idx, i, h, stats["E"], stats["F_model"], stats["F_fd"], stats["rel_err"]],
                )

            # 1B invariances
            shift = rng.normal(size=3)
            shift = shift / (np.linalg.norm(shift) + 1e-12) * args.translate_mag
            trans = _translation_invariance(calc, atoms, shift)
            rot = _rotation_invariance(calc, atoms, _random_rotation_matrix(rng))
            perm = _permutation_invariance(calc, atoms, rng)
            acc["trans_energy_diff"].append(trans["energy_diff"])
            acc["trans_force_diff"].append(trans["force_max_diff"])
            acc["rot_energy_diff"].append(rot["energy_diff"])
            acc["rot_force_diff"].append(rot["force_max_diff"])
            _write_csv_row(
                inv_csv,
                ["model", "frame", "test", "energy_diff", "force_max_diff"],
                [model_name, frame_idx, "translation", trans["energy_diff"], trans["force_max_diff"]],
            )
            _write_csv_row(
                inv_csv,
                ["model", "frame", "test", "energy_diff", "force_max_diff"],
                [model_name, frame_idx, "rotation", rot["energy_diff"], rot["force_max_diff"]],
            )
            if perm is not None:
                acc["perm_energy_diff"].append(perm["energy_diff"])
                acc["perm_force_diff"].append(perm["force_max_diff"])
                _write_csv_row(
                    inv_csv,
                    ["model", "frame", "test", "energy_diff", "force_max_diff"],
                    [model_name, frame_idx, "permutation", perm["energy_diff"], perm["force_max_diff"]],
                )

            # 1C loop work
            loop_work = _loop_work(calc, atoms, i, j, args.loop_delta)
            acc["loop_work"].append(loop_work)
            _write_csv_row(
                loop_csv,
                ["model", "frame", "pair", "loop_delta", "loop_work"],
                [model_name, frame_idx, f"{i},{j}", args.loop_delta, loop_work],
            )

            # 2A OOD jiggle with fixed bond length
            jiggle_energies = []
            jiggle_forces = []
            for _ in range(5):
                jiggle_pos = _jiggle_fixed_bond(atoms, i, j, rng, args.jiggle)
                jiggle_atoms = atoms.copy()
                jiggle_atoms.positions = jiggle_pos
                e, f = _get_energy_forces(calc, jiggle_atoms)
                u = jiggle_atoms.positions[j] - jiggle_atoms.positions[i]
                u = u / (np.linalg.norm(u) + 1e-12)
                fproj = float(np.dot(f[i], u))
                jiggle_energies.append(e)
                jiggle_forces.append(fproj)
            _write_csv_row(
                ood_csv,
                ["model", "frame", "energy_std", "force_proj_std"],
                [model_name, frame_idx, float(np.std(jiggle_energies)), float(np.std(jiggle_forces))],
            )
            acc["jiggle_energy_std"].append(float(np.std(jiggle_energies)))
            acc["jiggle_force_std"].append(float(np.std(jiggle_forces)))

            # 2B approach directions
            bond_vec = atoms.positions[j] - atoms.positions[i]
            bond_u = bond_vec / (np.linalg.norm(bond_vec) + 1e-12)
            # off-axis direction: rotate by small angle around random perpendicular
            rand_vec = rng.normal(size=3)
            perp = np.cross(bond_u, rand_vec)
            if np.linalg.norm(perp) < 1e-6:
                perp = np.array([0.0, 1.0, 0.0])
            perp = perp / (np.linalg.norm(perp) + 1e-12)
            angle = math.radians(args.off_axis_angle_deg)
            off_u = math.cos(angle) * bond_u + math.sin(angle) * perp
            e_axis, f_axis = _scan_along_direction(calc, atoms, i, j, scan_distances, bond_u)
            e_off, f_off = _scan_along_direction(calc, atoms, i, j, scan_distances, off_u)
            axis_mono = _monotone_repulsion(scan_distances, e_axis)
            off_mono = _monotone_repulsion(scan_distances, e_off)
            acc["axis_monotone"].append(int(axis_mono))
            acc["off_monotone"].append(int(off_mono))
            _write_csv_row(
                approach_csv,
                ["model", "frame", "direction", "energy_monotone", "force_proj_min", "force_proj_max"],
                [model_name, frame_idx, "axis", axis_mono, float(np.min(f_axis)), float(np.max(f_axis))],
            )
            _write_csv_row(
                approach_csv,
                ["model", "frame", "direction", "energy_monotone", "force_proj_min", "force_proj_max"],
                [model_name, frame_idx, "off_axis", off_mono, float(np.min(f_off)), float(np.max(f_off))],
            )

            # 3B analytic baseline
            baseline = _short_range_baseline(atoms, i, j, scan_distances)
            _write_csv_row(
                approach_csv,
                ["model", "frame", "direction", "energy_monotone", "force_proj_min", "force_proj_max"],
                [model_name, frame_idx, "zbl_baseline", baseline["zbl_monotone"], "", ""],
            )

            # 3A reference comparisons (optional)
            if ref_frames is not None:
                for ridx, ref_atoms in enumerate(ref_frames):
                    ref_E, ref_F = _find_ref_energy_forces(
                        ref_atoms, args.ref_energy_key, args.ref_forces_key
                    )
                    if ref_E is None or ref_F is None:
                        continue
                    e_model, f_model = _get_energy_forces(calc, ref_atoms)
                    bond = ref_atoms.positions[j] - ref_atoms.positions[i]
                    u = bond / (np.linalg.norm(bond) + 1e-12)
                    fproj_model = float(np.dot(f_model[i], u))
                    fproj_ref = float(np.dot(ref_F[i], u))
                    _write_csv_row(
                        ref_csv,
                        ["model", "ref_frame", "energy_model", "energy_ref", "force_proj_model", "force_proj_ref"],
                        [model_name, ridx, e_model, ref_E, fproj_model, fproj_ref],
                    )

        def _mean(x):
            return float(np.mean(x)) if len(x) else np.nan

        _write_csv_row(
            summary_csv,
            [
                "model",
                "fd_rel_err_mean",
                "trans_energy_diff_mean",
                "trans_force_diff_mean",
                "rot_energy_diff_mean",
                "rot_force_diff_mean",
                "perm_energy_diff_mean",
                "perm_force_diff_mean",
                "loop_work_mean",
                "jiggle_energy_std_mean",
                "jiggle_force_std_mean",
                "axis_monotone_rate",
                "off_monotone_rate",
            ],
            [
                model_name,
                _mean(acc["fd_rel_err"]),
                _mean(acc["trans_energy_diff"]),
                _mean(acc["trans_force_diff"]),
                _mean(acc["rot_energy_diff"]),
                _mean(acc["rot_force_diff"]),
                _mean(acc["perm_energy_diff"]),
                _mean(acc["perm_force_diff"]),
                _mean(acc["loop_work"]),
                _mean(acc["jiggle_energy_std"]),
                _mean(acc["jiggle_force_std"]),
                _mean(acc["axis_monotone"]),
                _mean(acc["off_monotone"]),
            ],
        )

        if use_wandb and wandb_run is not None:
            wandb.log(
                {
                    f"{model_name}/fd_rel_err_mean": _mean(acc["fd_rel_err"]),
                    f"{model_name}/trans_energy_diff_mean": _mean(acc["trans_energy_diff"]),
                    f"{model_name}/trans_force_diff_mean": _mean(acc["trans_force_diff"]),
                    f"{model_name}/rot_energy_diff_mean": _mean(acc["rot_energy_diff"]),
                    f"{model_name}/rot_force_diff_mean": _mean(acc["rot_force_diff"]),
                    f"{model_name}/perm_energy_diff_mean": _mean(acc["perm_energy_diff"]),
                    f"{model_name}/perm_force_diff_mean": _mean(acc["perm_force_diff"]),
                    f"{model_name}/loop_work_mean": _mean(acc["loop_work"]),
                    f"{model_name}/jiggle_energy_std_mean": _mean(acc["jiggle_energy_std"]),
                    f"{model_name}/jiggle_force_std_mean": _mean(acc["jiggle_force_std"]),
                    f"{model_name}/axis_monotone_rate": _mean(acc["axis_monotone"]),
                    f"{model_name}/off_monotone_rate": _mean(acc["off_monotone"]),
                }
            )

    if use_wandb and wandb_run is not None and not args.preflight:
        import wandb

        art = wandb.Artifact("consistency_check_outputs", type="results")
        for path in [fd_csv, inv_csv, loop_csv, ood_csv, approach_csv, ref_csv, summary_csv]:
            if path.exists():
                art.add_file(str(path), name=path.name)
        wandb_run.log_artifact(art)
        wandb_run.finish()

    print(f"[DONE] Wrote checks to: {outdir}")


if __name__ == "__main__":
    main()
