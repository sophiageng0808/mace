#!/usr/bin/env python3
import os
import csv
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

import wandb
import mace.modules.repulsion

# -----------------------------
# Config (dataset + scan)
# -----------------------------
USER = os.environ.get("USER", "unknown")
SCRATCH_MACE = Path(f"/scratch/{USER}/mace")
DATA_H5 = os.environ.get(
    "DATA_H5",
    str(SCRATCH_MACE / "data" / "train4M_h5" / "test"),
)
E0S_FILE = Path(
    os.environ.get("E0S_FILE", str(SCRATCH_MACE / "data" / "train4M_h5" / "E0s.json"))
)
N_STRUCTURES_DEFAULT = 1000
SEED_DEFAULT = 0

STEPS_DEFAULT = 50
START_DISTANCE_DEFAULT = 0.7
END_DISTANCE_DEFAULT = 0.2
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"

REL_DROP_TOL = 0.01  # monotonicity tolerance as fraction of curve span
CLAMP_THRESHOLD = 9.9e5

RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", f"/scratch/{USER}/mace_worktrees/jobs_repulsion"))
REPULSION_MODELS = [
    (
        "r12 scale 1.0",
        str(
            RUNS_ROOT
            / "r12_1.0"
            / "checkpoints"
            / "repulsion_r12_1.0_run-0_epoch-123.model"
        ),
    ),
    (
        "r12 scale 0.1",
        str(
            RUNS_ROOT
            / "r12_0.1"
            / "checkpoints"
            / "repulsion_r12_0.1_run-0_epoch-104.model"
        ),
    ),
    (
        "zbl scale 1.0",
        str(
            RUNS_ROOT
            / "zbl_1.0_preprocessed"
            / "checkpoints"
            / "repulsion_zbl_1.0_preprocessed_run-0_epoch-6.model"
        ),
    ),
    (
        "zbl scale 0.1",
        str(
            RUNS_ROOT
            / "zbl_0.1_preprocessed"
            / "checkpoints"
            / "repulsion_zbl_0.1_preprocessed_run-0_epoch-6.model"
        ),
    ),
]

# -----------------------------
# CSV headers
# -----------------------------
CSV_HEADER = [
    "molecule", "atom1", "atom2", "model", "group",
    "energy_monotone_shorter", "force_monotone_shorter", "force_abs_monotone_shorter",
    "min_energy", "max_energy", "max_force",
    "max_net_force_norm", "mean_net_force_norm",
    "max_net_torque_norm", "mean_net_torque_norm",
    "max_abs_dEdd_mismatch", "mean_abs_dEdd_mismatch",
    "max_rel_dEdd_mismatch", "mean_rel_dEdd_mismatch",
    "mean_force_cos_angle", "min_force_cos_angle",
    "n_nan_energy", "n_inf_energy", "n_nan_force", "n_inf_force",
    "has_nonfinite", "has_nan_only",
    "atom1_type", "atom2_type", "pair_label", "failed", "d_fail",
    "charge", "spin",
]

SUMMARY_HEADER = [
    "group", "model",
    "n_curves", "n_failed", "fail_rate",
    "energy_fail_rate", "force_fail_rate",
    "nonfinite_rate", "nan_only_rate",
    "mean_max_abs_dEdd_mismatch", "mean_mean_abs_dEdd_mismatch",
    "mean_max_rel_dEdd_mismatch", "mean_mean_rel_dEdd_mismatch",
    "mean_mean_net_force_norm", "mean_mean_net_torque_norm",
    "mean_mean_force_cos_angle", "mean_min_force_cos_angle",
]

def _unpack_h5_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value


def _scalar_property_to_info_float(arr: np.ndarray):
    """Return float for info dict, or None to omit (non-numeric, empty string, NaN/inf)."""
    if arr.size == 0:
        return None
    if not (arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1)):
        return None
    if np.issubdtype(arr.dtype, np.number):
        x = float(arr.reshape(-1)[0])
        return x if np.isfinite(x) else None
    raw = arr.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    s = str(raw).strip()
    if s == "" or s.lower() == "none":
        return None
    return None


def load_frames_from_h5(h5_dir: str) -> list:
    """Load structures from MACE sharded HDF5 directory (e.g. train4M_h5/test)."""
    import h5py
    from ase import Atoms

    h5_path = Path(h5_dir)
    if not h5_path.is_dir():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")
    h5_files = sorted(h5_path.glob("*.h5")) + sorted(h5_path.glob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5/.hdf5 files in {h5_dir}")

    frames = []
    for fpath in h5_files:
        with h5py.File(fpath, "r") as f:
            batch_keys = sorted(k for k in f.keys() if k.startswith("config_batch_"))
            for batch_key in batch_keys:
                grp = f[batch_key]
                config_keys = sorted(grp.keys(), key=lambda x: int(x.split("_")[-1]))
                for config_key in config_keys:
                    sub = grp[config_key]
                    numbers = np.asarray(sub["atomic_numbers"][()])
                    positions = np.asarray(sub["positions"][()])
                    cell = _unpack_h5_value(sub["cell"][()])
                    pbc = _unpack_h5_value(sub["pbc"][()])
                    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
                    for k, v in sub["properties"].items():
                        val = _unpack_h5_value(v[()])
                        if val is None:
                            continue
                        arr = np.asarray(val)
                        if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
                            xf = _scalar_property_to_info_float(arr)
                            if xf is not None:
                                atoms.info[k] = xf
                        elif arr.ndim == 2:
                            atoms.arrays[k] = arr
                    if "total_charge" in atoms.info and "charge" not in atoms.info:
                        atoms.info["charge"] = atoms.info["total_charge"]
                    if "total_spin" in atoms.info and "spin" not in atoms.info:
                        atoms.info["spin"] = atoms.info["total_spin"]
                    frames.append(atoms)
    return frames

def ensure_csv(path: Path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(header)

def write_csv(path: Path, row):
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(row)

def monotone_nondecreasing(values, rel_drop_tol=REL_DROP_TOL):
    v = np.asarray(values, float)
    if v.size < 3:
        return True
    vmin, vmax = float(np.min(v)), float(np.max(v))
    span = vmax - vmin
    if span <= 0:
        return True
    dv = np.diff(v)
    return not np.any(dv < -rel_drop_tol * span)

def first_failure_index(values, rel_drop_tol=REL_DROP_TOL):
    v = np.asarray(values, float)
    if v.size < 2:
        return None
    vmin, vmax = float(np.min(v)), float(np.max(v))
    span = vmax - vmin
    if span <= 0:
        return None
    dv = np.diff(v)
    bad = np.where(dv < -rel_drop_tol * span)[0]
    if bad.size == 0:
        return None
    return int(bad[0] + 1)


def _violations_involve_only_clamp(values, rel_drop_tol=REL_DROP_TOL, clamp_threshold=CLAMP_THRESHOLD):
    """True if all monotonicity violations in values involve at least one clamped point (|v|>=clamp_threshold)."""
    v = np.asarray(values, float)
    if v.size < 2:
        return True
    vmin, vmax = float(np.min(v)), float(np.max(v))
    span = vmax - vmin
    if span <= 0:
        return True
    dv = np.diff(v)
    bad = np.where(dv < -rel_drop_tol * span)[0]
    if bad.size == 0:
        return True
    for i in bad:
        a, b = float(v[i]), float(v[i + 1])
        # Violation is clamp-induced only if at least one value is at clamp boundary
        if abs(a) < clamp_threshold and abs(b) < clamp_threshold:
            return False  # real monotonicity failure, not clamp artifact
    return True

def preserve_info(atoms):
    atoms.info = dict(getattr(atoms, "info", {}) or {})

def get_charge_spin(atoms):
    info = dict(getattr(atoms, "info", {}) or {})
    charge = info.get("charge", None)
    spin = info.get("spin", None)
    if charge is not None:
        charge = int(charge)
    if spin is not None:
        spin = int(spin)
    return charge, spin

def load_mace_calculator(model_path: str, device: str):
    from mace.calculators import MACECalculator
    calc = MACECalculator(
        model_paths=[model_path],
        device=device,
        default_dtype="float64",
    )
    return calc


def _pair_repulsion_r_min(calc, submodule: str):
    """Return r_min from the pair-repulsion r12 or zbl block (submodule name) if present."""
    for model in getattr(calc, "models", []):
        pr = getattr(model, "pair_repulsion_fn", None)
        if pr is None:
            continue
        sub = getattr(pr, submodule, None)
        if sub is None:
            continue
        if hasattr(sub, "r_min"):
            return float(sub.r_min)
    return None


def _distance_grid_with_floor(dist, r_min, eps=1e-3):
    """Raise distances below r_min + eps when the grid would otherwise violate the floor."""
    if r_min is None:
        return dist
    min_dist = r_min + eps
    if np.min(dist) > min_dist:
        return dist
    return np.maximum(dist, min_dist)

def run_scan(
    atoms, i, j, calc,
    model_name: str,
    group_name: str,
    mol_label: str,
    dist: np.ndarray,
    all_csv: Path,
    failed_csv: Path,
):
    vec = atoms.positions[j] - atoms.positions[i]
    u = vec / (np.linalg.norm(vec) + 1e-12)

    raw_e, raw_f, raw_f_abs = [], [], []
    net_force_norms, net_torque_norms = [], []
    fi_dot_u, cos_angles = [], []
    n_nan_energy = 0
    n_inf_energy = 0
    n_nan_force = 0
    n_inf_force = 0

    charge, spin = get_charge_spin(atoms)

    for d in dist:
        test = atoms.copy()
        test.positions[i] = atoms.positions[j] - u * d
        preserve_info(test)
        test.calc = calc

        if hasattr(calc, "results") and isinstance(calc.results, dict):
            calc.results.clear()

        E = float(test.get_potential_energy())
        F = np.asarray(test.get_forces(), float)
        if not np.isfinite(E):
            if np.isnan(E):
                n_nan_energy += 1
            else:
                n_inf_energy += 1
        finite_mask = np.isfinite(F)
        if not np.all(finite_mask):
            n_nan_force += int(np.isnan(F).sum())
            n_inf_force += int(np.isinf(F).sum())

        Fi = F[i]
        Fi_proj = float(np.dot(Fi, u))
        Fi_norm = float(np.linalg.norm(Fi))
        cos_angle = Fi_proj / (Fi_norm + 1e-12)
        cos_angles.append(cos_angle)

        netF = np.sum(F, axis=0)
        net_force_norms.append(float(np.linalg.norm(netF)))

        com = np.asarray(test.get_center_of_mass(), float)
        r_rel = np.asarray(test.positions, float) - com[None, :]
        tau = np.sum(np.cross(r_rel, F), axis=0)
        net_torque_norms.append(float(np.linalg.norm(tau)))

        fi_dot_u.append(Fi_proj)
        raw_e.append(E)
        raw_f.append(-Fi_proj)
        raw_f_abs.append(abs(Fi_proj))

    raw_e = np.asarray(raw_e, float)
    raw_f = np.asarray(raw_f, float)
    raw_f_abs = np.asarray(raw_f_abs, float)
    net_force_norms = np.asarray(net_force_norms, float)
    net_torque_norms = np.asarray(net_torque_norms, float)
    fi_dot_u = np.asarray(fi_dot_u, float)
    cos_angles = np.asarray(cos_angles, float)

    denom = (dist[2:] - dist[:-2])
    dEdd_fd = (raw_e[2:] - raw_e[:-2]) / denom
    dEdd_force = fi_dot_u[1:-1]
    mismatch = dEdd_fd - dEdd_force
    abs_mismatch = np.abs(mismatch)
    rel_mismatch = abs_mismatch / (np.abs(dEdd_fd) + 1e-12)

    max_abs_mis = float(np.max(abs_mismatch)) if abs_mismatch.size else np.nan
    mean_abs_mis = float(np.mean(abs_mismatch)) if abs_mismatch.size else np.nan
    max_rel_mis = float(np.max(rel_mismatch)) if rel_mismatch.size else np.nan
    mean_rel_mis = float(np.mean(rel_mismatch)) if rel_mismatch.size else np.nan

    atom1 = atoms[i].symbol
    atom2 = atoms[j].symbol
    pair_label = f"{atom1}{i}->{atom2}{j}"

    e_mono = monotone_nondecreasing(raw_e)
    f_mono = monotone_nondecreasing(raw_f)
    f_abs_mono = monotone_nondecreasing(raw_f_abs)
    # Failure is defined only from force (-Fi·u) monotonicity

    min_e, max_e = float(np.min(raw_e)), float(np.max(raw_e))
    hit_clamp = (min_e <= -CLAMP_THRESHOLD) or (max_e >= CLAMP_THRESHOLD)
    if hit_clamp and failed:
        f_clamp_only = _violations_involve_only_clamp(raw_f)
        if f_clamp_only:
            failed = False
    has_nonfinite = (n_nan_energy + n_inf_energy + n_nan_force + n_inf_force) > 0
    has_nan_only = (n_nan_energy + n_nan_force) > 0 and (n_inf_energy + n_inf_force) == 0

    f_fail_idx = first_failure_index(raw_f)
    
    d_fail = float(dist[f_fail_idx]) if f_fail_idx is not None else np.nan

    row = [
        mol_label, i, j, model_name, group_name,
        bool(e_mono), bool(f_mono), bool(f_abs_mono),
        float(np.min(raw_e)), float(np.max(raw_e)), float(np.max(raw_f)),
        float(np.max(net_force_norms)), float(np.mean(net_force_norms)),
        float(np.max(net_torque_norms)), float(np.mean(net_torque_norms)),
        max_abs_mis, mean_abs_mis, max_rel_mis, mean_rel_mis,
        float(np.mean(cos_angles)), float(np.min(cos_angles)),
        int(n_nan_energy), int(n_inf_energy), int(n_nan_force), int(n_inf_force),
        bool(has_nonfinite), bool(has_nan_only),
        atom1, atom2, pair_label, bool(failed), d_fail,
        charge, spin,
    ]

    ensure_csv(all_csv, CSV_HEADER)
    ensure_csv(failed_csv, CSV_HEADER)
    write_csv(all_csv, row)
    if failed:
        write_csv(failed_csv, row)

    return dict(
        failed=bool(failed),
        energy_failed=bool(not e_mono),
        force_failed=bool(not f_mono),
        has_nonfinite=bool(has_nonfinite),
        has_nan_only=bool(has_nan_only),
        max_abs_dEdd_mismatch=max_abs_mis,
        mean_abs_dEdd_mismatch=mean_abs_mis,
        max_rel_dEdd_mismatch=max_rel_mis,
        mean_rel_dEdd_mismatch=mean_rel_mis,
        mean_net_force_norm=float(np.mean(net_force_norms)),
        mean_net_torque_norm=float(np.mean(net_torque_norms)),
        mean_force_cos_angle=float(np.mean(cos_angles)),
        min_force_cos_angle=float(np.min(cos_angles)),
    )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_structures",
        type=int,
        default=N_STRUCTURES_DEFAULT,
        help="Max structures to scan. None = full dataset (use --n_structures 0 or omit to mean all)",
    )
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    parser.add_argument("--start_distance", type=float, default=START_DISTANCE_DEFAULT)
    parser.add_argument("--end_distance", type=float, default=END_DISTANCE_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Default: auto")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run folder name. Default: timestamp.")
    parser.add_argument("--group", type=str, default="repulsion_cosine_gate",
                        help="Output group name (e.g. repulsion, repulsion_smooth, repulsion_cosine_gate).")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = args.device or DEVICE_DEFAULT

    repulsion_root = Path(__file__).resolve().parent
    os.chdir(repulsion_root)

    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    group_name = args.group

    OUTPUT_ROOT = SCRATCH_MACE / "outputs" / "dissociation_scans_overfit100"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    group_dir = OUTPUT_ROOT / group_name / run_id
    group_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_DISABLED", "false")
        os.environ.setdefault("WANDB_DIR", str(group_dir / "wandb"))
        wandb.init(
            project="dmlip-dissociation-scan-overfit100",
            name=f"overfit100_scan_{run_id}",
            config={"group": group_name, "run_id": run_id, "device": device},
        )

    if not E0S_FILE.is_file():
        raise FileNotFoundError(f"Missing E0S_FILE: {E0S_FILE}")
    h5_dir = Path(DATA_H5)
    if not h5_dir.is_dir():
        raise FileNotFoundError(f"Missing H5 directory: {DATA_H5}")

    frames = load_frames_from_h5(DATA_H5)
    ds_len = len(frames)
    n_to_run = ds_len if (args.n_structures is None or args.n_structures <= 0) else min(args.n_structures, ds_len)
    print(f"H5 n={ds_len} scan n={n_to_run}")

    dist = np.linspace(args.start_distance, args.end_distance, args.steps)
    rng = np.random.default_rng(args.seed)

    SUMMARY_CSV_PATH = group_dir / "model_summary_metrics.csv"
    ensure_csv(SUMMARY_CSV_PATH, SUMMARY_HEADER)

    # Load calculators
    calc_map = {}
    for model_name, model_path in REPULSION_MODELS:
        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(f"[{group_name}] Missing model file: {model_name} -> {model_path}")
        calc_map[model_name] = load_mace_calculator(str(mp), device)

    # Per-model CSV outputs
    per_model_csv = {}
    for model_name in calc_map.keys():
        model_dir = group_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        all_csv = model_dir / "all_curves_metrics.csv"
        failed_csv = model_dir / "failed_curves_metrics.csv"
        ensure_csv(all_csv, CSV_HEADER)
        ensure_csv(failed_csv, CSV_HEADER)
        per_model_csv[model_name] = (all_csv, failed_csv)

    per_model_acc = {
        model_name: {
            "n": 0,
            "n_failed": 0,
            "n_energy_failed": 0,
            "n_force_failed": 0,
            "n_nonfinite": 0,
            "n_nan_only": 0,
            "max_abs_mis": [],
            "mean_abs_mis": [],
            "max_rel_mis": [],
            "mean_rel_mis": [],
            "mean_net_force": [],
            "mean_net_torque": [],
            "mean_cos": [],
            "min_cos": [],
        }
        for model_name in calc_map.keys()
    }

    for idx in range(n_to_run):
        at = frames[int(idx)].copy()
        preserve_info(at)
        mol_label = f"train4m_{idx}"

        nat = len(at)
        if nat < 2:
            continue

        i = int(rng.integers(0, nat))
        dist_to_all = np.linalg.norm(at.positions - at.positions[i], axis=1)
        dist_to_all[i] = np.inf
        j = int(np.argmin(dist_to_all))

        for model_name, calc in calc_map.items():
            dist_model = dist
            if "r12" in model_name:
                dist_model = _distance_grid_with_floor(dist_model, _pair_repulsion_r_min(calc, "r12"))
            if "zbl" in model_name:
                dist_model = _distance_grid_with_floor(dist_model, _pair_repulsion_r_min(calc, "zbl"))
            metrics = run_scan(
                at, i, j, calc,
                model_name=model_name,
                group_name=group_name,
                mol_label=mol_label,
                dist=dist_model,
                all_csv=per_model_csv[model_name][0],
                failed_csv=per_model_csv[model_name][1],
            )

            acc = per_model_acc[model_name]
            acc["n"] += 1
            acc["n_failed"] += int(metrics["failed"])
            acc["n_energy_failed"] += int(metrics["energy_failed"])
            acc["n_force_failed"] += int(metrics["force_failed"])
            acc["n_nonfinite"] += int(metrics["has_nonfinite"])
            acc["n_nan_only"] += int(metrics["has_nan_only"])
            acc["max_abs_mis"].append(metrics["max_abs_dEdd_mismatch"])
            acc["mean_abs_mis"].append(metrics["mean_abs_dEdd_mismatch"])
            acc["max_rel_mis"].append(metrics["max_rel_dEdd_mismatch"])
            acc["mean_rel_mis"].append(metrics["mean_rel_dEdd_mismatch"])
            acc["mean_net_force"].append(metrics["mean_net_force_norm"])
            acc["mean_net_torque"].append(metrics["mean_net_torque_norm"])
            acc["mean_cos"].append(metrics["mean_force_cos_angle"])
            acc["min_cos"].append(metrics["min_force_cos_angle"])

    for model_name, acc in per_model_acc.items():
        n = max(acc["n"], 1)
        n_failed = acc["n_failed"]
        row = [
            group_name, model_name,
            int(acc["n"]), int(n_failed), float(n_failed / n),
            float(acc["n_energy_failed"] / n), float(acc["n_force_failed"] / n),
            float(acc["n_nonfinite"] / n), float(acc["n_nan_only"] / n),
            float(np.nanmean(acc["max_abs_mis"])), float(np.nanmean(acc["mean_abs_mis"])),
            float(np.nanmean(acc["max_rel_mis"])), float(np.nanmean(acc["mean_rel_mis"])),
            float(np.nanmean(acc["mean_net_force"])), float(np.nanmean(acc["mean_net_torque"])),
            float(np.nanmean(acc["mean_cos"])), float(np.nanmean(acc["min_cos"])),
        ]
        write_csv(SUMMARY_CSV_PATH, row)

    if use_wandb:
        art = wandb.Artifact(f"scan_outputs_overfit100_{group_name}_{run_id}", type="results")
        for model_name, (all_csv, failed_csv) in per_model_csv.items():
            art.add_file(str(all_csv), name=f"{model_name}/all_curves_metrics.csv")
            art.add_file(str(failed_csv), name=f"{model_name}/failed_curves_metrics.csv")
        art.add_file(str(SUMMARY_CSV_PATH))
        wandb.run.log_artifact(art)

    print(f"done {group_name} -> {group_dir}  summary {SUMMARY_CSV_PATH}")

    if use_wandb:
        wandb.finish()
