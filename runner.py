#!/usr/bin/env python3
"""
QTCL Experiment Orchestrator.

Generates all (dataset, backbone, head, seed) combinations from config.yaml,
applies CLI filters, checks resumability, and executes sequentially or in parallel.

Usage:
    python runner.py --config config.yaml
    python runner.py --config config.yaml --dataset split_cifar10 --backbone resnet18 --dry-run
    python runner.py --config config.yaml --study ablation --parallel 4
    python runner.py --config config.yaml --export-commands
"""

import argparse
import csv
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

_csv_lock = Lock()

_log_handler = logging.StreamHandler(sys.stdout)
if hasattr(_log_handler.stream, "reconfigure"):
    _log_handler.stream.reconfigure(encoding="utf-8")
_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])
logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    run_id:   str
    dataset:  str
    backbone: str
    head:     str
    seed:     int
    study:    str = "main"
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("datasets", "backbones", "heads", "seeds", "output_dir"):
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    return cfg


# ── Run generation ────────────────────────────────────────────────────────────

def generate_main_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    for ds in config["datasets"]:
        ds_name = ds["name"] if isinstance(ds, dict) else ds
        for bb in config["backbones"]:
            bb_name = bb["name"] if isinstance(bb, dict) else bb
            for hd in config["heads"]:
                hd_name = hd["name"] if isinstance(hd, dict) else hd
                if hd_name not in ["mlp", "qk_ideal", "qk_noisy"]:
                    continue
                if ds_name == "split_cifar100_20" and hd_name not in ["mlp", "qk_ideal", "qk_noisy"]:
                    continue
                for seed in config["seeds"]:
                    runs.append(RunConfig(
                        run_id=f"{ds_name}_{bb_name}_{hd_name}_{seed}",
                        dataset=ds_name, backbone=bb_name, head=hd_name, seed=seed,
                    ))
    return runs


def generate_ablation_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    ab = config.get("ablation", {})
    if not ab or not ab.get("enabled", True):
        return runs
    default_q, default_d = 4, 2
    for ds in ab.get("datasets", []):
        for bb in ab.get("backbones", []):
            for hd in ab.get("heads", []):
                for q in ab.get("qubits", [default_q]):
                    for d in ab.get("depths", [default_d]):
                        if q == default_q and d == default_d:
                            continue
                        for seed in ab.get("seeds", [42]):
                            runs.append(RunConfig(
                                run_id=f"{ds}_{bb}_{hd}_q{q}_d{d}_{seed}",
                                dataset=ds, backbone=bb, head=hd, seed=seed,
                                study="ablation",
                                overrides={"n_qubits": q, "depth": d},
                            ))
                # Extra qubit counts tested only at depth 1
                for q in ab.get("extra_qubits", []):
                    for seed in ab.get("seeds", [42]):
                        runs.append(RunConfig(
                            run_id=f"{ds}_{bb}_{hd}_q{q}_d1_{seed}",
                            dataset=ds, backbone=bb, head=hd, seed=seed,
                            study="ablation",
                            overrides={"n_qubits": q, "depth": 1},
                        ))
    return runs


def generate_lambda_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    ls = config.get("lambda_sensitivity", {})
    if not ls or not ls.get("enabled", True):
        return runs
    for ds in ls.get("datasets", []):
        for bb in ls.get("backbones", []):
            for hd in ls.get("heads", []):
                for lam in ls.get("lambdas", [5000]):
                    for seed in ls.get("seeds", [42]):
                        runs.append(RunConfig(
                            run_id=f"{ds}_{bb}_{hd}_lam{lam}_{seed}",
                            dataset=ds, backbone=bb, head=hd, seed=seed,
                            study="lambda_sensitivity",
                            overrides={"lambda": lam},
                        ))
    return runs


def generate_derpp_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    ds = config.get("derpp_sensitivity", {})
    if not ds or not ds.get("enabled", True):
        return runs
    for dataset in ds.get("datasets", []):
        for bb in ds.get("backbones", []):
            for hd in ds.get("heads", []):
                for alpha in ds.get("alphas", [0.5]):
                    for beta in ds.get("betas", [0.5]):
                        for seed in ds.get("seeds", [42]):
                            runs.append(RunConfig(
                                run_id=f"{dataset}_{bb}_{hd}_a{alpha}_b{beta}_{seed}",
                                dataset=dataset, backbone=bb, head=hd, seed=seed,
                                study="derpp_sensitivity",
                                overrides={"alpha": alpha, "beta": beta},
                            ))
    return runs


def generate_scalability_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    sc = config.get("scalability", {})
    if not sc or not sc.get("enabled", True):
        return runs
    for ds in sc.get("datasets", []):
        ds_cfg = next((d for d in config["datasets"] if d["name"] == ds), {})
        max_tasks = ds_cfg.get("n_tasks", 5)
        for bb in sc.get("backbones", []):
            for hd in sc.get("heads", []):
                for seed in sc.get("seeds", [42]):
                    # One run per (dataset, backbone, head, seed) — n_tasks from dataset config
                    runs.append(RunConfig(
                        run_id=f"{ds}_{bb}_{hd}_scale_{seed}",
                        dataset=ds, backbone=bb, head=hd, seed=seed,
                        study="scalability",
                    ))
    return runs


def generate_cl_methods_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    cs = config.get("cl_methods_study", {})
    if not cs or not cs.get("enabled", True):
        return runs
    for ds in cs.get("datasets", []):
        for bb in cs.get("backbones", []):
            for hd in cs.get("heads", []):
                for method in cs.get("methods", ["ewc"]):
                    for seed in cs.get("seeds", [42]):
                        runs.append(RunConfig(
                            run_id=f"{ds}_{bb}_{hd}_{method}_{seed}",
                            dataset=ds, backbone=bb, head=hd, seed=seed,
                            study="cl_methods",
                            overrides={"cl_method": method},
                        ))
    return runs


def generate_arch_variants_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    av = config.get("arch_variants", {})
    if not av or not av.get("enabled", True):
        return runs
    for ds in av.get("datasets", []):
        for bb in av.get("backbones", []):
            for hd in av.get("heads", []):
                for seed in av.get("seeds", [42]):
                    runs.append(RunConfig(
                        run_id=f"{ds}_{bb}_{hd}_archvar_{seed}",
                        dataset=ds, backbone=bb, head=hd, seed=seed,
                        study="arch_variants",
                    ))
    return runs


def generate_noise_decomposition_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    nd = config.get("noise_decomposition", {})
    if not nd or not nd.get("enabled", True):
        return runs
    for ds in nd.get("datasets", []):
        for bb in nd.get("backbones", []):
            for ch in nd.get("channels", []):
                ch_name = ch["name"]
                for seed in nd.get("seeds", [42]):
                    runs.append(RunConfig(
                        run_id=f"{ds}_{bb}_{ch_name}_{seed}",
                        dataset=ds, backbone=bb,
                        head=nd.get("base_head", "qk_noisy"),
                        seed=seed,
                        study="noise_decomposition",
                        overrides={
                            "noise_channels": ch.get("noise_channels", []),
                            "n_qubits": nd.get("n_qubits", 4),
                            "depth": nd.get("depth", 2),
                        },
                    ))
    return runs


def generate_noise_mitigation_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    nm = config.get("noise_mitigation_study", {})
    if not nm or not nm.get("enabled", True):
        return runs
    for ds in nm.get("datasets", []):
        for bb in nm.get("backbones", []):
            for hd in nm.get("heads", []):
                for mit in nm.get("mitigations", []):
                    for seed in nm.get("seeds", [42]):
                        runs.append(RunConfig(
                            run_id=f"{ds}_{bb}_{hd}_mit_{mit}_{seed}",
                            dataset=ds, backbone=bb, head=hd, seed=seed,
                            study="noise_mitigation_study",
                            overrides={"noise_mitigation": mit},
                        ))
    return runs


def generate_spsa_variants_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    sv = config.get("spsa_variants_study", {})
    if not sv or not sv.get("enabled", True):
        return runs
    for ds in sv.get("datasets", []):
        for bb in sv.get("backbones", []):
            for hd in sv.get("heads", []):
                for var in sv.get("variants", []):
                    for seed in sv.get("seeds", [42]):
                        runs.append(RunConfig(
                            run_id=f"{ds}_{bb}_{hd}_var_{var}_{seed}",
                            dataset=ds, backbone=bb, head=hd, seed=seed,
                            study="spsa_variants_study",
                            overrides={"spsa_variant": var},
                        ))
    return runs


def generate_realhardware_runs(config: Dict[str, Any]) -> List[RunConfig]:
    runs = []
    rh = config.get("realhardware", {})
    if not rh or not rh.get("enabled", True):
        return runs
    for ds in rh.get("datasets", []):
        for bb in rh.get("backbones", []):
            for hd in rh.get("heads", []):
                for mit in rh.get("mitigations", []):
                    for seed in rh.get("seeds", [42]):
                        runs.append(RunConfig(
                            run_id=f"{ds}_{bb}_{hd}_realhardware_{mit}_{seed}",
                            dataset=ds, backbone=bb, head=hd, seed=seed,
                            study="realhardware",
                            overrides={"noise_mitigation": mit},
                        ))
    return runs


# ── Filtering ─────────────────────────────────────────────────────────────────

def apply_filters(runs: List[RunConfig], args: argparse.Namespace) -> List[RunConfig]:
    if args.dataset:
        allowed = set(args.dataset.split(","))
        runs = [r for r in runs if r.dataset in allowed]
    if args.backbone:
        allowed = set(args.backbone.split(","))
        runs = [r for r in runs if r.backbone in allowed]
    if args.head:
        allowed = set(args.head.split(","))
        runs = [r for r in runs if r.head in allowed]
    if args.seed:
        allowed = set(int(s) for s in args.seed.split(","))
        runs = [r for r in runs if r.seed in allowed]

    # Study-specific parameter filters to prevent duplicate runs
    if args.qubits is not None:
        runs = [r for r in runs if r.overrides and r.overrides.get("n_qubits") == args.qubits]
    if args.depth is not None:
        runs = [r for r in runs if r.overrides and r.overrides.get("depth") == args.depth]
    if args.lambda_val is not None:
        runs = [r for r in runs if r.overrides and r.overrides.get("lambda") == args.lambda_val]
    if args.alpha is not None:
        runs = [r for r in runs if r.overrides and r.overrides.get("alpha") == args.alpha]
    if args.beta is not None:
        runs = [r for r in runs if r.overrides and r.overrides.get("beta") == args.beta]
    if args.cl_method is not None:
        runs = [r for r in runs if r.overrides and r.overrides.get("cl_method") == args.cl_method]
    if args.noise_channels is not None:
        allowed_channels = set(args.noise_channels.split(","))
        runs = [
            r for r in runs
            if r.overrides and "noise_channels" in r.overrides
            and set(r.overrides["noise_channels"]) == allowed_channels
        ]
    if args.noise_mitigation is not None:
        runs = [r for r in runs if r.overrides and r.overrides.get("noise_mitigation") == args.noise_mitigation]
    if args.spsa_variant is not None:
        runs = [r for r in runs if r.overrides and r.overrides.get("spsa_variant") == args.spsa_variant]
    return runs


# ── Output helpers ────────────────────────────────────────────────────────────

def scan_completed(output_dir: str) -> Set[str]:
    completed: Set[str] = set()
    if not os.path.exists(output_dir):
        return completed
    folders = os.listdir(output_dir)
    import concurrent.futures
    
    def process_folder(folder):
        csv_path = os.path.join(output_dir, folder, "runs.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if "run_id" in df.columns:
                    return set(df["run_id"].astype(str))
            except Exception:
                pass
        return set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(process_folder, folders)
        for r in results:
            completed.update(r)
    return completed


def create_run_folder(base: str, machine_id: Optional[str], args: argparse.Namespace, config: Dict[str, Any]) -> str:
    os.makedirs(base, exist_ok=True)
    existing = [int(n.split("_")[0]) for n in os.listdir(base)
                if os.path.isdir(os.path.join(base, n)) and n.split("_")[0].isdigit()]
    next_id = (max(existing) + 1) if existing else 1
    parts = [machine_id] if machine_id else []
    if args.dataset:
        parts.append(args.dataset.replace(",", "-"))
    if args.backbone:
        parts.append(args.backbone.replace(",", "-"))
    if args.head:
        parts.append(args.head.replace(",", "-"))
    if args.study:
        parts.append(args.study)
    if not parts:
        parts.append(config.get("experiment_name", "experiment"))
    folder = os.path.join(base, f"{next_id:03d}_{'_'.join(parts)}")
    os.makedirs(folder, exist_ok=True)
    return folder


def append_csv(path: str, data: Any, is_list: bool = False) -> None:
    with _csv_lock:
        df = pd.DataFrame(data if is_list else [data])
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)


# ── Execution ─────────────────────────────────────────────────────────────────

def execute_run(run: RunConfig, config: Dict[str, Any], args: argparse.Namespace):
    from trainer import train_and_evaluate
    return train_and_evaluate(run, config, run.overrides)


def run_sequential(runs, config, args, existing_ids):
    folder = config["output_dir"]
    paths = {k: os.path.join(folder, f"{k}.csv") for k in ("runs", "predictions", "training_log")}
    errors_log = os.path.join(folder, "errors.log")
    done = skip = errs = 0
    total = len(runs)
    for i, run in enumerate(runs, 1):
        if run.run_id in existing_ids:
            logger.info(f"[SKIP {i}/{total}] {run.run_id}")
            skip += 1
            continue
        logger.info(f"[RUN {i}/{total}] {run.run_id}")
        try:
            result, preds, tlog = execute_run(run, config, args)
            append_csv(paths["runs"], result)
            append_csv(paths["training_log"], tlog, is_list=True)
            logger.info(f"[DONE {i}/{total}] {run.run_id} AA={result.get('AA', '?'):.4f}")
            done += 1
        except Exception as e:
            logger.error(f"[ERROR {i}/{total}] {run.run_id}: {e}")
            with open(errors_log, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [{run.run_id}] {e}\n{traceback.format_exc()}\n{'='*60}\n")
            errs += 1
    return done, skip, errs


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QTCL Experiment Orchestrator")
    p.add_argument("--config",      required=True, help="Path to config.yaml")
    p.add_argument("--machine-id",  default=None)
    p.add_argument("--dataset",     default=None)
    p.add_argument("--backbone",    default=None)
    p.add_argument("--head",        default=None)
    p.add_argument("--seed",        default=None)
    p.add_argument("--study",       default=None,
                   choices=["ablation", "lambda_sensitivity", "derpp_sensitivity",
                            "scalability", "noise_decomposition", "cl_methods", "arch_variants",
                            "noise_mitigation_study", "spsa_variants_study", "realhardware"])
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--count",       action="store_true")
    p.add_argument("--export-commands", action="store_true")
    p.add_argument("--parallel",    type=int, default=1)
    p.add_argument("--overwrite",   action="store_true")

    # Study-specific parameters for fine-grained filtering
    p.add_argument("--qubits",      type=int, default=None, help="Filter by number of qubits for ablation")
    p.add_argument("--depth",       type=int, default=None, help="Filter by circuit depth for ablation")
    p.add_argument("--lambda",      type=float, dest="lambda_val", default=None, help="Filter by lambda coefficient")
    p.add_argument("--alpha",       type=float, default=None, help="Filter by alpha for DER++")
    p.add_argument("--beta",        type=float, default=None, help="Filter by beta for DER++")
    p.add_argument("--cl-method",   default=None, help="Filter by continual learning method name")
    p.add_argument("--noise-channels", default=None, help="Filter by noise channels for noise decomposition")
    p.add_argument("--noise-mitigation", default=None, help="Filter by noise mitigation technique")
    p.add_argument("--spsa-variant",     default=None, help="Filter by SPSA variant")
    p.add_argument("--shots",            type=int, default=None, help="Override number of shots")
    p.add_argument("--tasks",            type=int, default=None, help="Override number of tasks")
    p.add_argument("--epochs",           type=int, default=None, help="Override epochs per task")
    p.add_argument("--backend",          default=None, help="Override backend name")
    return p.parse_args()


def export_commands(runs: List[RunConfig], args: argparse.Namespace) -> None:
    # Use absolute paths so SLURM job arrays work regardless of working directory.
    python_bin  = sys.executable
    runner_path = os.path.abspath(__file__)
    config_path = os.path.abspath(args.config)
    for r in runs:
        cmd = [
            f'"{python_bin}" "{runner_path}" --config "{config_path}"',
            f"--dataset {r.dataset}",
            f"--backbone {r.backbone}",
            f"--head {r.head}",
            f"--seed {r.seed}",
        ]
        if args.study:
            cmd.append(f"--study {args.study}")
            # Add overrides as explicit command-line parameters to prevent duplicate executions
            if r.overrides:
                if "n_qubits" in r.overrides:
                    cmd.append(f"--qubits {r.overrides['n_qubits']}")
                if "depth" in r.overrides:
                    cmd.append(f"--depth {r.overrides['depth']}")
                if "lambda" in r.overrides:
                    cmd.append(f"--lambda {r.overrides['lambda']}")
                if "alpha" in r.overrides:
                    cmd.append(f"--alpha {r.overrides['alpha']}")
                if "beta" in r.overrides:
                    cmd.append(f"--beta {r.overrides['beta']}")
                if "cl_method" in r.overrides:
                    cmd.append(f"--cl-method {r.overrides['cl_method']}")
                if "noise_channels" in r.overrides:
                    ch_str = ",".join(r.overrides["noise_channels"])
                    cmd.append(f"--noise-channels {ch_str}")
                if "noise_mitigation" in r.overrides:
                    cmd.append(f"--noise-mitigation {r.overrides['noise_mitigation']}")
                if "spsa_variant" in r.overrides:
                    cmd.append(f"--spsa-variant {r.overrides['spsa_variant']}")
                if "shots" in r.overrides:
                    cmd.append(f"--shots {r.overrides['shots']}")
                if "n_tasks" in r.overrides:
                    cmd.append(f"--tasks {r.overrides['n_tasks']}")
                if "epochs_per_task" in r.overrides:
                    cmd.append(f"--epochs {r.overrides['epochs_per_task']}")
                if "backend" in r.overrides:
                    cmd.append(f"--backend {r.overrides['backend']}")

        if args.machine_id:
            cmd.append(f"--machine-id {args.machine_id}")
        print(" ".join(cmd))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.study == "ablation":
        runs = generate_ablation_runs(config)
    elif args.study == "lambda_sensitivity":
        runs = generate_lambda_runs(config)
    elif args.study == "derpp_sensitivity":
        runs = generate_derpp_runs(config)
    elif args.study == "scalability":
        runs = generate_scalability_runs(config)
    elif args.study == "noise_decomposition":
        runs = generate_noise_decomposition_runs(config)
    elif args.study == "cl_methods":
        runs = generate_cl_methods_runs(config)
    elif args.study == "arch_variants":
        runs = generate_arch_variants_runs(config)
    elif args.study == "noise_mitigation_study":
        runs = generate_noise_mitigation_runs(config)
    elif args.study == "spsa_variants_study":
        runs = generate_spsa_variants_runs(config)
    elif args.study == "realhardware":
        runs = generate_realhardware_runs(config)
    else:
        runs = generate_main_runs(config)

    runs = apply_filters(runs, args)

    for r in runs:
        if args.shots is not None:
            r.overrides["shots"] = args.shots
        if args.tasks is not None:
            r.overrides["n_tasks"] = args.tasks
        if args.epochs is not None:
            r.overrides["epochs_per_task"] = args.epochs
        if args.backend is not None:
            r.overrides["backend"] = args.backend

    if args.dry_run:
        if args.export_commands:
            export_commands(runs, args)
        elif args.count:
            print(len(runs))
        else:
            print(f"Total runs: {len(runs)}")
            datasets = sorted(set(r.dataset for r in runs))
            backbones = sorted(set(r.backbone for r in runs))
            heads = sorted(set(r.head for r in runs))
            seeds = sorted(set(r.seed for r in runs))
            print(f"Datasets:  {datasets}")
            print(f"Backbones: {backbones}")
            print(f"Heads:     {heads}")
            print(f"Seeds:     {seeds}")
        return 0

    if len(runs) == 0:
        print("No runs match the specified filters.")
        return 0

    existing_ids = scan_completed(config["output_dir"]) & {r.run_id for r in runs}

    if not args.overwrite and len(existing_ids) == len(runs):
        print(f"All {len(runs)} runs already completed. Skipping execution.")
        return 0

    run_folder = create_run_folder(config["output_dir"], args.machine_id, args, config)
    config["output_dir"] = run_folder

    if args.overwrite:
        existing_ids = set()

    done, skip, errs = run_sequential(runs, config, args, existing_ids)

    print(f"\nCompleted: {done}  Skipped: {skip}  Errors: {errs}  Total: {len(runs)}")
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
