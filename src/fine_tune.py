import argparse
import copy
import numbers
import os
from glob import glob
from types import SimpleNamespace
from typing import Any, Dict, Optional

from tasks.genre_classification import GenreClassificationModel
from tasks.melody_completion import MelodyCompletionModel, train as train_melody_completion
from tasks.performer_composer_classification import train as train_performer_composer_classification
from tasks.generate_midi import MusicDecoder
from tasks.emotion_classification import train as train_emotion
from src.dataset import MidiDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import torch
import yaml
torch.set_float32_matmul_precision('high')

DEFAULT_FINE_TUNE_OUTPUT = os.getenv('FINE_TUNE_OUTPUT_DIR', './output/fine_tune')


def _default_fine_tune_config() -> Dict[str, Any]:
    base_output = os.getenv('FINE_TUNE_OUTPUT_DIR', './output/fine_tune')
    return {
        "dataset": {
            "limit_data": None,
        },
        "model": {
            "tokenization": "remi",
        },
        "training": {
            "max_epochs": 10,
            "fast_dev_run": False,
        },
        "misc": {
            "task": 'genre',
            "output_dir": base_output,
            "run_name": None,
            "model_path": None,
        },
    }


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping.")
    return data


def load_fine_tune_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load fine-tuning configuration from YAML and merge with defaults and overrides."""
    defaults = _default_fine_tune_config()
    config = copy.deepcopy(defaults)
    file_config = _load_yaml_config(path)
    _deep_update(config, file_config)
    if overrides:
        _deep_update(config, overrides)

    for section, section_defaults in defaults.items():
        section_cfg = config.setdefault(section, {})
        if not isinstance(section_cfg, dict):
            raise ValueError(f"Fine-tune config section '{section}' must be a mapping.")
        if isinstance(section_defaults, dict):
            for key, default_value in section_defaults.items():
                if section_cfg.get(key) is None:
                    section_cfg[key] = default_value

    return config


def _extract_scalar_metrics(metric_dict):
    scalars = {}
    for key, value in metric_dict.items():
        if isinstance(value, numbers.Number):
            scalars[key] = float(value)
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                scalars[key] = float(value.detach().cpu().item())
        elif hasattr(value, "item"):
            try:
                scalars[key] = float(value.item())
            except Exception:
                continue
    return scalars


def run_fine_tuning(config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    defaults = _default_fine_tune_config()
    for section, section_defaults in defaults.items():
        section_cfg = config.setdefault(section, {})
        if not isinstance(section_cfg, dict):
            raise ValueError(f"Fine-tune config section '{section}' must be a mapping.")
        if isinstance(section_defaults, dict):
            for key, default_value in section_defaults.items():
                if section_cfg.get(key) is None:
                    section_cfg[key] = default_value

    combined: Dict[str, Any] = {}
    for section in ("dataset", "model", "training", "misc"):
        combined.update(config.get(section, {}))

    args = SimpleNamespace(**combined)

    args.fast_dev_run = bool(getattr(args, "fast_dev_run", False))
    if getattr(args, "limit_data", None) is not None:
        args.limit_data = int(args.limit_data)
    args.max_epochs = int(getattr(args, "max_epochs", 10))
    args.tokenization = str(getattr(args, "tokenization", "remi"))
    args.task = str(getattr(args, "task", "genre"))
    if getattr(args, "output_dir", None) is None:
        args.output_dir = os.getenv('FINE_TUNE_OUTPUT_DIR', DEFAULT_FINE_TUNE_OUTPUT)
    args.output_dir = os.fspath(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task in {'melody_completion', 'accompaniment_suggestion'}:
        result = train_melody_completion(args)
        return result or {"task": args.task}
    
    if args.task in {'performer', 'composer', 'difficulty'}:
        result = train_performer_composer_classification(args)
        return result or {"task": args.task}

    if args.task == 'emotion':
        result = train_emotion(args)
        return result or {"task": args.task}

    midi_files = glob(os.path.join("/root/Sym-JEPA/dataset/clean_midi", "**/*.mid"), recursive=True)
    if args.limit_data:
        midi_files = midi_files[:args.limit_data]

    data_module = MidiDataModule(
        midi_files,
        max_len=2048,
        batch_size=16,
        num_workers=4,
        jepa_context_ratio_start=0.975,
        jepa_context_ratio_end=0.975,
        jepa_context_ratio_steps=100,
        skip_unknown_genres=args.task == 'genre',
        skip_unknown_styles=args.task == 'style',
        tokenization=args.tokenization,
        masking_probabilities={
            "none": 1.0,
        },
        genre_map='/root/Sym-JEPA/metadata/midi_genre_map.json'
    )

    data_module.setup()

    if args.task in {'genre', 'style'}:
        model = GenreClassificationModel(
            num_classes=len(data_module.train_ds.all_genres) if args.task == 'genre' else len(data_module.train_ds.all_styles),
            lr=5e-7,
            d_model=512,
            encoder_layers=8,
            num_attention_heads=8,
            symjepa_config=args.model_config,
            class_weights=torch.tensor([1/(x+1e-7) for x in data_module.train_ds.genre_counts]) if args.task == 'genre' else torch.tensor([1/(x+1e-7) for x in data_module.train_ds.style_counts]),
            task=args.task
        )
    elif args.task == 'decode':
        model = MusicDecoder(
            lr=1e-6
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    if args.model_path:
        model.load_jepa(args.model_path)

    if args.fast_dev_run:
        logger = None
    else:
        experiment_name = f"symjepa-{args.task}-classification" if args.task != "decode" else "symjepa-decode"
        logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri="/root/Sym-JEPA/mlruns",
            run_name=args.run_name
        )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        dirpath=os.path.join(args.output_dir, f"{args.task}_checkpoints")
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step')
        ],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=200,
        log_every_n_steps=max(1, 50 // 50),
        default_root_dir=args.output_dir
    )

    trainer.validate(model, data_module)

    trainer.fit(model, data_module)

    metrics = _extract_scalar_metrics(trainer.callback_metrics)
    best_model_path = None
    if checkpoint_callback is not None:
        best_model_path = getattr(checkpoint_callback, "best_model_path", None) or getattr(checkpoint_callback, "last_model_path", None)

    return {
        "task": args.task,
        "metrics": metrics,
        "best_model_path": best_model_path,
        "train_dataset_size": len(getattr(data_module, "train_ds", [])),
        "val_dataset_size": len(getattr(data_module, "valid_ds", [])),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sym-JEPA using a YAML configuration file.")
    parser.add_argument("--config", type=str, help="Path to YAML config describing the fine-tuning run.")
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Override config to enable Lightning fast_dev_run for smoke testing.",
    )
    cli_args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    if cli_args.fast_dev_run:
        overrides.setdefault("training", {})["fast_dev_run"] = True

    config = load_fine_tune_config(cli_args.config, overrides=overrides)
    run_fine_tuning(config)


if __name__ == "__main__":
    main()
