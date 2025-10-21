import argparse
import copy
import numbers
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pytorch_lightning as pl
import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
try:  # Optional dependency
    from ray.tune.search.optuna import OptunaSearch  # type: ignore
except Exception:  # pragma: no cover - fallback if Optuna is unavailable
    OptunaSearch = None  # type: ignore
import torch
import yaml

from src.train import load_train_config, run_training
from src.fine_tune import load_fine_tune_config, run_fine_tuning

# Enable better Tensor Core utilization on NVIDIA GPUs
torch.set_float32_matmul_precision('high')


TUNE_TAGS = {
    "!choice": lambda seq: tune.choice(seq),
    "!uniform": lambda lo_hi: tune.uniform(*lo_hi),
    "!loguniform": lambda lo_hi: tune.loguniform(*lo_hi),
    "!randint": lambda lo_hi: tune.randint(*lo_hi),
    "!quniform": lambda lo_hi_q: tune.quniform(*lo_hi_q),
}

for tag, fn in TUNE_TAGS.items():
    yaml.SafeLoader.add_constructor(
        tag,
        lambda loader, node, fn=fn: fn(loader.construct_sequence(node)),  # noqa: B023
    )


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def _prefix_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    prefixed = {}
    for key, value in metrics.items():
        if isinstance(value, numbers.Number):
            prefixed[f"{prefix}_{key}"] = float(value)
    return prefixed


def _maybe_get_trial_dir() -> Path:
    try:
        trial_dir = Path(tune.get_trial_dir())
        if trial_dir:
            return trial_dir
    except Exception:
        pass
    return Path("./trial_dir")


def _prepare_pipeline_context(cfg: Dict[str, Any], fast_dev_override: bool = False) -> Dict[str, Any]:
    context: Dict[str, Any] = {}

    pretrain_cfg = cfg.get("pretrain")
    if not pretrain_cfg or "config_path" not in pretrain_cfg:
        raise ValueError("Pipeline config must include 'pretrain.config_path'.")

    pretrain_overrides = copy.deepcopy(pretrain_cfg.get("overrides", {}))
    if fast_dev_override:
        pretrain_overrides.setdefault("training", {})["fast_dev_run"] = True
    context["pretrain"] = load_train_config(pretrain_cfg["config_path"], overrides=pretrain_overrides)

    finetune_cfg = cfg.get("finetune") or {}
    shared_overrides = copy.deepcopy(finetune_cfg.get("shared_overrides", {}))
    if fast_dev_override:
        shared_overrides.setdefault("training", {})["fast_dev_run"] = True

    tasks_cfg = finetune_cfg.get("tasks") or {}
    task_order = list(tasks_cfg.keys())
    finetune_tasks: Dict[str, Dict[str, Any]] = {}
    for task_name, task_info in tasks_cfg.items():
        if not isinstance(task_info, dict):
            raise ValueError(f"Task configuration for '{task_name}' must be a mapping.")
        base_overrides = copy.deepcopy(task_info.get("overrides", {}))
        base_overrides.setdefault("misc", {})
        if not isinstance(base_overrides["misc"], dict):
            raise ValueError(f"Overrides for task '{task_name}' must provide 'misc' as a mapping.")
        base_overrides["misc"].setdefault("task", task_name)
        if fast_dev_override:
            base_overrides.setdefault("training", {})["fast_dev_run"] = True

        task_config_path = task_info.get("config_path")
        if task_config_path:
            task_config = load_fine_tune_config(task_config_path, overrides=base_overrides)
        else:
            task_config = load_fine_tune_config(None, overrides=base_overrides)
        finetune_tasks[task_name] = task_config

    context["finetune"] = {
        "tasks": finetune_tasks,
        "shared_overrides": shared_overrides,
        "task_order": list(task_order),
    }
    return context


def train_and_finetune(config: Dict[str, Any], pipeline_context: Dict[str, Any]) -> None:
    trial_dir = _maybe_get_trial_dir()
    pretrain_dir = trial_dir / "pretrain"
    finetune_root = trial_dir / "finetune"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    finetune_root.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 0))
    _set_global_seed(seed)

    os.environ["OUTPUT_DIR"] = str(pretrain_dir)

    pretrain_config = copy.deepcopy(pipeline_context["pretrain"])
    _deep_update(pretrain_config, copy.deepcopy(config.get("pretrain", {})))
    pretrain_misc = pretrain_config.setdefault("misc", {})
    if not isinstance(pretrain_misc, dict):
        raise ValueError("Pretrain configuration 'misc' section must be a mapping.")
    pretrain_misc["output_dir"] = str(pretrain_dir)
    pretrain_result = run_training(pretrain_config)
    best_model_path = pretrain_result.get("best_model_path")

    report_metrics: Dict[str, float] = {}
    report_metrics.update(_prefix_metrics("pretrain", pretrain_result.get("metrics", {})))
    report_metrics["pretrain_skipped_files"] = float(pretrain_result.get("skipped_files", 0))

    finetune_context = pipeline_context.get("finetune", {})
    available_tasks: Dict[str, Dict[str, Any]] = finetune_context.get("tasks", {})
    if not available_tasks:
        tune.report(**report_metrics)
        return

    finetune_section = copy.deepcopy(config.get("finetune", {}))
    requested_tasks: Optional[Iterable[str]] = finetune_section.pop("tasks", None)
    shared_overrides_dynamic = finetune_section.pop("shared", {})
    per_task_overrides_dynamic = finetune_section.pop("per_task", {})

    tasks_to_run = list(requested_tasks) if requested_tasks else list(finetune_context.get("task_order", available_tasks.keys()))
    if not tasks_to_run:
        tune.report(**report_metrics)
        return

    for task in tasks_to_run:
        if task not in available_tasks:
            raise KeyError(f"Requested task '{task}' is not defined in pipeline configuration.")

        task_dir = finetune_root / task
        task_dir.mkdir(parents=True, exist_ok=True)

        task_config = copy.deepcopy(available_tasks[task])
        if finetune_context.get("shared_overrides"):
            _deep_update(task_config, copy.deepcopy(finetune_context["shared_overrides"]))
        if shared_overrides_dynamic:
            _deep_update(task_config, copy.deepcopy(shared_overrides_dynamic))
        if task in per_task_overrides_dynamic:
            overrides = per_task_overrides_dynamic[task]
            if not isinstance(overrides, dict):
                raise ValueError(f"Overrides for task '{task}' must be a mapping.")
            _deep_update(task_config, copy.deepcopy(overrides))
        if finetune_section:
            _deep_update(task_config, finetune_section)

        task_misc = task_config.setdefault("misc", {})
        if not isinstance(task_misc, dict):
            raise ValueError(f"Fine-tune configuration for task '{task}' must include a mapping for 'misc'.")
        task_misc["output_dir"] = str(task_dir)
        task_misc.setdefault("run_name", f"{task}-seed-{seed}")
        task_misc.setdefault("task", task)
        if best_model_path and not task_misc.get("model_path"):
            task_misc["model_path"] = best_model_path

        os.environ["FINE_TUNE_OUTPUT_DIR"] = str(task_dir)
        finetune_result = run_fine_tuning(task_config) or {}
        metrics = finetune_result.get("metrics", {})
        report_metrics.update(_prefix_metrics(f"finetune_{task}", metrics))

    tune.report(**report_metrics)


def _create_scheduler(cfg: Dict[str, Any]) -> Optional[Any]:
    if not cfg:
        return None
    scheduler_type = str(cfg.get("type", "ASHA")).upper()
    if scheduler_type == "ASHA":
        return ASHAScheduler(
            max_t=int(cfg.get("max_t", 50)),
            grace_period=int(cfg.get("grace_period", 1)),
            reduction_factor=int(cfg.get("reduction_factor", 3)),
        )
    raise ValueError(f"Unsupported scheduler type '{scheduler_type}'.")


def _create_search_alg(cfg: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not cfg:
        return None
    alg_type = str(cfg.get("type", "")).lower()
    if alg_type == "optuna":
        if OptunaSearch is None:
            raise ImportError("OptunaSearch is unavailable. Install ray[tune] with Optuna extras to use this search algorithm.")
        return OptunaSearch(seed=cfg.get("seed"))
    raise ValueError(f"Unsupported search algorithm type '{alg_type}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Sym-JEPA pretraining + fine-tuning with Ray Tune.")
    parser.add_argument("--config", required=True, help="Path to Ray Tune pipeline YAML configuration.")
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Override pipeline config to enable fast_dev_run for pretrain and fine-tune.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    exp_cfg = cfg.get("experiment") or {}
    if not all(k in exp_cfg for k in ("name", "metric", "mode")):
        raise ValueError("Pipeline experiment configuration must include 'name', 'metric', and 'mode'.")

    pipeline_context = _prepare_pipeline_context(cfg, fast_dev_override=args.fast_dev_run)

    resources_cfg = cfg.get("resources") or {}
    trainable = tune.with_parameters(train_and_finetune, pipeline_context=pipeline_context)
    trainable = tune.with_resources(trainable, resources=resources_cfg)

    scheduler = _create_scheduler(cfg.get("scheduler", {}))
    search_alg = _create_search_alg(cfg.get("search_alg"))
    param_space = cfg.get("param_space") or {}

    ray_cfg = cfg.get("ray", {}) or {}
    if ray.is_initialized():
        pass
    else:
        ray.init(address=ray_cfg.get("address"), namespace=ray_cfg.get("namespace"))

    tuner = tune.Tuner(
        trainable=trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric=exp_cfg["metric"],
            mode=str(exp_cfg["mode"]).lower(),
            num_samples=int(exp_cfg.get("num_samples", 1)),
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=air.RunConfig(name=exp_cfg["name"]),
    )

    results = tuner.fit()
    best = results.get_best_result(metric=exp_cfg["metric"], mode=str(exp_cfg["mode"]).lower())
    print("Best configuration:", best.config)
    print(f"Best {exp_cfg['metric']}: {best.metrics.get(exp_cfg['metric'])}")


if __name__ == "__main__":
    main()
