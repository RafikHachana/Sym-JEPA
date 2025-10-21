import argparse
import numbers
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pytorch_lightning as pl
import ray
from ray import air, tune
import torch

from src.train import build_train_parser, run_training
from src.fine_tune import build_fine_tune_parser, run_fine_tuning


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def _build_args(parser_builder, overrides: Dict[str, Any]) -> argparse.Namespace:
    parser = parser_builder()
    args = parser.parse_args([])
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(args, key):
            raise KeyError(f"Unknown argument override '{key}' for parser {parser_builder.__name__}")
        setattr(args, key, value)
    return args


def _prefix_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    prefixed = {}
    for key, value in metrics.items():
        if isinstance(value, numbers.Number):
            prefixed[f"{prefix}_{key}"] = float(value)
    return prefixed


def train_and_finetune(config: Dict[str, Any]) -> None:
    trial_dir = Path(tune.get_trial_dir())
    trial_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 0))
    _set_global_seed(seed)

    pretrain_cfg = dict(config.get("pretrain", {}))
    finetune_cfg = dict(config.get("finetune", {}))

    # Prepare directories
    pretrain_dir = trial_dir / "pretrain"
    finetune_root = trial_dir / "finetune"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    finetune_root.mkdir(parents=True, exist_ok=True)

    os.environ["OUTPUT_DIR"] = str(pretrain_dir)

    # Run pretraining
    pretrain_cfg.setdefault("output_dir", str(pretrain_dir))
    pretrain_cfg.setdefault("train_downstream_tasks", False)
    pretrain_args = _build_args(build_train_parser, pretrain_cfg)

    pretrain_result = run_training(pretrain_args)
    best_model_path = pretrain_result.get("best_model_path")

    report_metrics: Dict[str, float] = {}
    report_metrics.update(_prefix_metrics("pretrain", pretrain_result.get("metrics", {})))
    report_metrics["pretrain_skipped_files"] = float(pretrain_result.get("skipped_files", 0))

    # Prepare fine-tuning configuration
    tasks: Iterable[str] = finetune_cfg.get("tasks", [])
    per_task_overrides = finetune_cfg.get("per_task_overrides", {})
    shared_overrides = {
        key: value
        for key, value in finetune_cfg.items()
        if key not in {"tasks", "per_task_overrides"}
    }

    for task in tasks:
        task_dir = finetune_root / task
        task_dir.mkdir(parents=True, exist_ok=True)

        overrides = dict(shared_overrides)
        overrides.update(per_task_overrides.get(task, {}))

        overrides.setdefault("task", task)
        overrides.setdefault("output_dir", str(task_dir))
        overrides.setdefault("model_path", best_model_path)
        overrides.setdefault("run_name", f"{task}-seed-{seed}")

        os.environ["FINE_TUNE_OUTPUT_DIR"] = str(task_dir)
        finetune_args = _build_args(build_fine_tune_parser, overrides)

        finetune_result = run_fine_tuning(finetune_args) or {}
        metrics = finetune_result.get("metrics", {})
        report_metrics.update(_prefix_metrics(f"finetune_{task}", metrics))

    tune.report(**report_metrics)


def main():
    parser = argparse.ArgumentParser(description="Launch Sym-JEPA pretraining + fine-tuning with Ray Tune")
    parser.add_argument("--midi-dir", type=str, default="./dataset/clean_midi/",
                        help="Path to MIDI corpus for pretraining")
    parser.add_argument("--tasks", nargs="+", default=["genre"],
                        help="Downstream tasks to fine-tune after pretraining")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds to evaluate (grid search)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Additional random samples Ray should run per configuration")
    parser.add_argument("--tokenization", type=str, default="remi", choices=["remi", "octuple"])
    parser.add_argument("--max-pretrain-epochs", type=int, default=10)
    parser.add_argument("--max-finetune-epochs", type=int, default=5)
    parser.add_argument("--pretrain-limit", type=int, default=None,
                        help="Optional limit on number of MIDI files for pretraining")
    parser.add_argument("--finetune-limit", type=int, default=None,
                        help="Optional limit on data points during fine-tuning")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Use Lightning fast_dev_run for smoke tests")
    parser.add_argument("--address", type=str, default=None,
                        help="Optional Ray cluster address (defaults to local mode)")
    parser.add_argument("--cpus-per-trial", type=float, default=4.0)
    parser.add_argument("--gpus-per-trial", type=float, default=0.0)

    args = parser.parse_args()

    if args.address:
        ray.init(address=args.address)
    else:
        ray.init()

    search_space = {
        "seed": tune.grid_search(sorted(set(args.seeds))),
        "pretrain": {
            "midi_dir": args.midi_dir,
            "tokenization": args.tokenization,
            "max_epochs": args.max_pretrain_epochs,
            "limit": args.pretrain_limit,
            "fast_dev_run": args.fast_dev_run,
        },
        "finetune": {
            "tasks": args.tasks,
            "tokenization": args.tokenization,
            "max_epochs": args.max_finetune_epochs,
            "limit_data": args.finetune_limit,
            "fast_dev_run": args.fast_dev_run,
        },
    }

    trainable = tune.with_resources(
        train_and_finetune,
        resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
    )

    tuner = tune.Tuner(
        trainable=trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
        ),
        run_config=air.RunConfig(
            name="symjepa-ray-tune",
        ),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
