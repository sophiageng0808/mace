#!/usr/bin/env python3
"""Load a training checkpoint (.pt) and save a full MACE module (.model) for calculators / LAMMPS.

Expects the same MACE code as training (this branch: pair repulsion) on PYTHONPATH.
Uses logs/*_debug.log in the run directory to recover training Namespace (same idea as run_train).
"""
from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from mace import tools
from mace.data import KeySpecification, update_keyspec_from_kwargs
from mace.tools.model_script_utils import configure_model
from mace.tools.scripts_utils import dict_to_array, get_atomic_energies
from mace.tools.utils import AtomicNumberTable


def _extract_namespace_inner(line: str) -> str:
    start = line.index("Namespace(") + len("Namespace(")
    depth = 1
    i = start
    while i < len(line) and depth:
        if line[i] == "(":
            depth += 1
        elif line[i] == ")":
            depth -= 1
        i += 1
    return line[start : i - 1]


def _split_top_level(s: str) -> list[str]:
    fields: list[str] = []
    buf: list[str] = []
    depth = 0
    for c in s:
        if c == "," and depth == 0:
            w = "".join(buf).strip()
            if w:
                fields.append(w)
            buf = []
        else:
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            buf.append(c)
    if buf:
        w = "".join(buf).strip()
        if w:
            fields.append(w)
    return fields


def _parse_value(v: str):
    v = v.strip()
    if v == "None":
        return None
    if v == "True":
        return True
    if v == "False":
        return False
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        return v[1:-1]
    if v.startswith("["):
        return ast.literal_eval(v)
    if v.startswith("{"):
        return ast.literal_eval(v)
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_training_args_from_debug_log(log_path: Path) -> SimpleNamespace:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    line = None
    for ln in text.splitlines():
        if "Configuration: Namespace(" in ln:
            line = ln
            break
    if line is None:
        raise FileNotFoundError(f"No 'Configuration: Namespace(' line in {log_path}")

    inner = _extract_namespace_inner(line)
    raw: dict = {}
    for field in _split_top_level(inner):
        if "=" not in field:
            continue
        key, val = field.split("=", 1)
        key = key.strip()
        if key == "key_specification":
            continue
        if val.strip().startswith("KeySpecification"):
            continue
        raw[key] = _parse_value(val)

    parser = tools.build_default_arg_parser()
    train_file = raw.get("train_file")
    if isinstance(train_file, list):
        train_file = train_file[0]
    e0s = raw.get("E0s")
    if not train_file or not e0s:
        raise ValueError(f"Debug log {log_path} missing train_file or E0s in Namespace")
    init_argv = [
        "--name",
        str(raw.get("name", "_")),
        "--train_file",
        str(train_file),
        "--E0s",
        str(e0s),
    ]
    args = parser.parse_args(init_argv)
    for k, v in raw.items():
        if hasattr(args, k):
            setattr(args, k, v)

    args, _ = tools.check_args(args)
    if getattr(args, "foundation_model", None) is None:
        args.multiheads_finetuning = False

    # run_train sets these on args before configure_model (standard energy+forces MACE)
    args.compute_energy = True
    args.compute_forces = getattr(args, "compute_forces", True)
    args.compute_dipole = getattr(args, "compute_dipole", False)
    args.compute_polarizability = getattr(args, "compute_polarizability", False)

    args.key_specification = KeySpecification()
    update_keyspec_from_kwargs(args.key_specification, vars(args))
    return args


def _find_debug_log(run_dir: Path, checkpoint_stem: str) -> Path:
    """checkpoint_stem like repulsion_zbl_0.1_run-0_epoch-50 -> repulsion_zbl_0.1_run-0_debug.log"""
    import re

    m = re.match(r"^(?P<tag>.+)_epoch-\d+(?:_swa)?$", checkpoint_stem)
    tag = m.group("tag") if m else checkpoint_stem
    cand = run_dir / "logs" / f"{tag}_debug.log"
    if cand.is_file():
        return cand
    logs = sorted((run_dir / "logs").glob("*_debug.log"))
    if len(logs) == 1:
        return logs[0]
    raise FileNotFoundError(
        f"Could not find debug log for {checkpoint_stem!r} under {run_dir / 'logs'}"
    )


def export_checkpoint(
    checkpoint_path: Path,
    run_dir: Path | None,
    out_path: Path | None,
    device: str = "cpu",
) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    run_dir = (run_dir or checkpoint_path.parent.parent).resolve()
    os.chdir(run_dir)
    args = parse_training_args_from_debug_log(_find_debug_log(run_dir, checkpoint_path.stem))

    args.device = device
    tools.set_default_dtype(args.default_dtype)
    tools.init_device(args.device)

    zs = ast.literal_eval(args.atomic_numbers)
    z_table = AtomicNumberTable(zs)
    heads = ["Default"]
    atomic_energies_dict = {
        "Default": get_atomic_energies(args.E0s, None, z_table),
    }
    atomic_energies = dict_to_array(atomic_energies_dict, heads)

    args.mean = 0.0
    args.std = [1.0] * len(heads)

    model, _ = configure_model(
        args,
        train_loader=None,
        atomic_energies=atomic_energies,
        model_foundation=None,
        heads=heads,
        z_table=z_table,
        head_configs=None,
    )
    model = model.to(torch.device(device))

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict mismatch: missing={missing} unexpected={unexpected}")

    if out_path is None:
        out_path = checkpoint_path.with_suffix(".model")
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out_path)
    logging.info("Wrote %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="Path to *_epoch-*.pt checkpoint")
    p.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Training run directory (default: parent of checkpoints/)",
    )
    p.add_argument("-o", "--output", type=Path, default=None, help="Output .model path")
    p.add_argument("--device", default="cpu", help="map_location / model device (default: cpu)")
    args = p.parse_args()
    out = export_checkpoint(args.checkpoint, args.run_dir, args.output, device=args.device)
    print(out)


if __name__ == "__main__":
    main()
