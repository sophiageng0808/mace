#!/usr/bin/env python3
import os
import csv
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import wandb
from ase.io import read as ase_read

# -----------------------------
# Config (dataset + scan)
# -----------------------------
DATA_EXTXYZ = "/h/400/sophiageng/mace/data/overfit100_E0sub.extxyz"
N_STRUCTURES_DEFAULT = 100

STEPS_DEFAULT = 50
START_DISTANCE_DEFAULT = 0.7
END_DISTANCE_DEFAULT = 0.2
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"

REL_DROP_TOL = 0.01  # monotonicity tolerance as fraction of curve span

# -----------------------------
# Radial-basis models (ABSOLUTE PATHS)
# -----------------------------
RADIAL_MODELS = [
    # ("main_baseline", "/h/400/sophiageng/mace_worktrees/jobs_radialbasis/baseline/main_baseline.model"),
    # ("sine", "/h/400/sophiageng/mace_worktrees/jobs_radialbasis/sine/radial_sine.model"),
    # ("cos", "/h/400/sophiageng/mace_worktrees/jobs_radialbasis/cos/radial_cos.model"),
    ("cosm1_over_r", "/h/400/sophiageng/mace_worktrees/jobs_radialbasis/360491_3_radial_cosm1_over_r/radial_cosm1_over_r.model"),
    # ("cos_over_r", "/h/400/sophiageng/mace_worktrees/jobs_radialbasis/360409_4_radial_cos_over_r/radial_cos_over_r.model"),
    # ("cos", "/h/400/sophiageng/mace_worktrees/jobs_radialbasis/360493_2_radial_cos/radial_cos.model"),
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
    "atom1_type", "atom2_type", "pair_label", "failed", "d_fail",
    "charge", "spin",
]

SUMMARY_HEADER = [
    "group", "model",
    "n_curves", "n_failed", "fail_rate",
    "energy_fail_rate", "force_fail_rate",
    "mean_max_abs_dEdd_mismatch", "mean_mean_abs_dEdd_mismatch",
    "mean_max_rel_dEdd_mismatch", "mean_mean_rel_dEdd_mismatch",
    "mean_mean_net_force_norm", "mean_mean_net_torque_norm",
    "mean_mean_force_cos_angle", "mean_min_force_cos_angle",
]

# -----------------------------
# Helpers
# -----------------------------
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
    return MACECalculator(
        model_paths=[model_path],
        device=device,
        default_dtype="float64",
    )

def eval_signature(calc, atoms):
    test = atoms.copy()
    preserve_info(test)
    test.calc = calc
    if hasattr(calc, "results") and isinstance(calc.results, dict):
        calc.results.clear()
    E = float(test.get_potential_energy())
    F = np.asarray(test.get_forces(), float)
    return E, float(np.linalg.norm(F)), float(np.max(np.abs(F)))

def run_scan(
    atoms, i, j, calc,
    model_name: str,
    group_name: str,
    mol_label: str,
    dist: np.ndarray,
    all_csv: Path,
    failed_csv: Path,
    use_wandb: bool,
):
    vec = atoms.positions[j] - atoms.positions[i]
    u = vec / (np.linalg.norm(vec) + 1e-12)

    raw_e, raw_f, raw_f_abs = [], [], []
    net_force_norms, net_torque_norms = [], []
    fi_dot_u, cos_angles = [], []

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
    failed = (not e_mono) or (not f_mono)

    e_fail_idx = first_failure_index(raw_e)
    f_fail_idx = first_failure_index(raw_f)
    fail_candidates = [idx for idx in (e_fail_idx, f_fail_idx) if idx is not None]
    d_fail = float(dist[min(fail_candidates)]) if fail_candidates else np.nan

    row = [
        mol_label, i, j, model_name, group_name,
        bool(e_mono), bool(f_mono), bool(f_abs_mono),
        float(np.min(raw_e)), float(np.max(raw_e)), float(np.max(raw_f)),
        float(np.max(net_force_norms)), float(np.mean(net_force_norms)),
        float(np.max(net_torque_norms)), float(np.mean(net_torque_norms)),
        max_abs_mis, mean_abs_mis, max_rel_mis, mean_rel_mis,
        float(np.mean(cos_angles)), float(np.min(cos_angles)),
        atom1, atom2, pair_label, bool(failed), d_fail,
        charge, spin,
    ]

    ensure_csv(all_csv, CSV_HEADER)
    ensure_csv(failed_csv, CSV_HEADER)
    write_csv(all_csv, row)
    if failed:
        write_csv(failed_csv, row)

    if failed and use_wandb:
        wandb.log({f"{group_name}/{model_name}/failed_curve": 1})

    return dict(
        failed=bool(failed),
        energy_failed=bool(not e_mono),
        force_failed=bool(not f_mono),
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
    parser.add_argument("--n_structures", type=int, default=N_STRUCTURES_DEFAULT)
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    parser.add_argument("--start_distance", type=float, default=START_DISTANCE_DEFAULT)
    parser.add_argument("--end_distance", type=float, default=END_DISTANCE_DEFAULT)
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Default: auto")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run folder name. Default: timestamp.")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = args.device or DEVICE_DEFAULT

    radial_root = Path(__file__).resolve().parent
    os.chdir(radial_root)

    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    group_name = "radial_basis"

    OUTPUT_ROOT = Path("/h/400/sophiageng/mace/outputs") / "dissociation_scans_overfit100"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="dmlip-dissociation-scan-overfit100",
            name=f"overfit100_scan_{run_id}",
            config={
                "groups": [group_name],
                "steps": args.steps,
                "start_distance": args.start_distance,
                "end_distance": args.end_distance,
                "device": device,
                "data_extxyz": DATA_EXTXYZ,
                "n_structures": args.n_structures,
                "run_id": run_id,
            },
            settings=wandb.Settings(start_method="thread"),
        )

    if not os.path.exists(DATA_EXTXYZ):
        raise FileNotFoundError(f"Missing extxyz: {DATA_EXTXYZ}")

    frames = ase_read(DATA_EXTXYZ, ":")
    ds_len = len(frames)
    n_to_run = ds_len if args.n_structures is None else min(int(args.n_structures), ds_len)
    print(f"[OK] Loaded extxyz {DATA_EXTXYZ} with len={ds_len}; running n={n_to_run}")

    dist = np.linspace(args.start_distance, args.end_distance, args.steps)

    # Group-level outputs (per-run folder)
    group_dir = OUTPUT_ROOT / group_name / run_id
    group_dir.mkdir(parents=True, exist_ok=True)

    SUMMARY_CSV_PATH = group_dir / "model_summary_metrics.csv"
    ensure_csv(SUMMARY_CSV_PATH, SUMMARY_HEADER)

    # Load calculators
    calc_map = {}
    for model_name, model_path in RADIAL_MODELS:
        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(f"[{group_name}] Missing model file: {model_name} -> {model_path}")
        calc_map[model_name] = load_mace_calculator(str(mp), device)
        print(f"[OK] [{group_name}] Loaded {model_name} from {model_path}")

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

    # Fingerprint sanity check
    probe = frames[0].copy()
    preserve_info(probe)
    print(f"\n[FINGERPRINT] [{group_name}] energy + ||F|| + max|F| on frame0:")
    for name, calc in calc_map.items():
        E, Fn, Fmax = eval_signature(calc, probe)
        print(f"  {name:22s}  E={E:+.8f}  ||F||={Fn:.6f}  max|F|={Fmax:.6f}")

    print("\n[INFO] Starting scans...\n")

    per_model_acc = {
        model_name: {
            "n": 0,
            "n_failed": 0,
            "n_energy_failed": 0,
            "n_force_failed": 0,
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
        mol_label = f"overfit100_{idx}"

        nat = len(at)
        if nat < 2:
            continue

        for i in range(nat):
            dist_to_all = np.linalg.norm(at.positions - at.positions[i], axis=1)
            dist_to_all[i] = np.inf
            j = int(np.argmin(dist_to_all))

            for model_name, calc in calc_map.items():
                metrics = run_scan(
                    at, i, j, calc,
                    model_name=model_name,
                    group_name=group_name,
                    mol_label=mol_label,
                    dist=dist,
                    all_csv=per_model_csv[model_name][0],
                    failed_csv=per_model_csv[model_name][1],
                    use_wandb=use_wandb,
                )

                acc = per_model_acc[model_name]
                acc["n"] += 1
                acc["n_failed"] += int(metrics["failed"])
                acc["n_energy_failed"] += int(metrics["energy_failed"])
                acc["n_force_failed"] += int(metrics["force_failed"])
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

    print(f"[DONE] Group {group_name} finished.")
    print(f"  Per-model CSV root: {group_dir}")
    print(f"  Summary CSV:     {SUMMARY_CSV_PATH}\n")

    global_summary_csv = group_dir / "all_groups_model_summary.csv"
    pd.DataFrame([{
        "run_id": run_id,
        "group": group_name,
        "model": name,
        "n_curves": int(per_model_acc[name]["n"]),
        "n_failed": int(per_model_acc[name]["n_failed"]),
        "fail_rate": float(per_model_acc[name]["n_failed"] / max(per_model_acc[name]["n"], 1)),
    } for name in per_model_acc.keys()]).to_csv(global_summary_csv, index=False)

    print("[DONE] Finished radial_basis group.")
    print("Run outputs root:", OUTPUT_ROOT)
    print("Global model summary CSV:", global_summary_csv)

    if use_wandb:
        if global_summary_csv.exists():
            wandb.log({"all_groups_model_summary": wandb.Table(dataframe=pd.read_csv(global_summary_csv))})
        wandb.finish()
