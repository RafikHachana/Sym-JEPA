import argparse
import copy
import numbers
import os
from glob import glob
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import yaml
from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

# Enable better Tensor Core utilization on NVIDIA GPUs
torch.set_float32_matmul_precision('high')

from src.model import SymJEPA
from src.dataset import MidiDataModule


def _default_train_config() -> Dict[str, Any]:
    """Return the default grouped configuration for pretraining."""
    return {
        "dataset": {
            "midi_dir": os.getenv("MIDI_DIR", "./dataset/clean_midi/"),
            "limit": None,
            "max_len": 2048,
        },
        "model": {
            "tokenization": "remi",
            "use_mask_padding": False,
            "masking_mode": "contiguous",
            "masking_probability": 0.25,
            "segment_size_ratio": 0.1,
            "num_segments": 3,
            "use_vicreg": False,
            "vicreg_sim_weight": 1.0,
            "vicreg_var_weight": 40.0,
            "vicreg_cov_weight": 10.0,
            "vicreg_loss_ratio": 0.3,
            "pass_target_mask_to_predictor": False,
        },
        "training": {
            "max_epochs": int(os.getenv("MAX_EPOCHS", 100)),
            "lr": 5e-2,
            "limit_batches": None,
            "fast_dev_run": False,
            "jepa_context_ratio_start": float(os.getenv("JEPA_CONTEXT_RATIO_START", 0.95)),
            "jepa_context_ratio_end": float(os.getenv("JEPA_CONTEXT_RATIO_END", 0.6)),
        },
        "fine_tuning": {},
        "misc": {
            "output_dir": os.getenv("OUTPUT_DIR", "./output"),
            "run_name": None,
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


def load_train_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load training configuration from YAML and merge with defaults and overrides."""
    defaults = _default_train_config()
    config = copy.deepcopy(defaults)
    file_config = _load_yaml_config(path)
    _deep_update(config, file_config)
    if overrides:
        _deep_update(config, overrides)

    for section, section_defaults in defaults.items():
        section_cfg = config.setdefault(section, {})
        if not isinstance(section_cfg, dict):
            raise ValueError(f"Training config section '{section}' must be a mapping.")
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

def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    # Load environment variables from .env file
    load_dotenv()
    config = copy.deepcopy(config)
    defaults = _default_train_config()
    for section, section_defaults in defaults.items():
        section_cfg = config.setdefault(section, {})
        if not isinstance(section_cfg, dict):
            raise ValueError(f"Training config section '{section}' must be a mapping.")
        if isinstance(section_defaults, dict):
            for key, default_value in section_defaults.items():
                if section_cfg.get(key) is None:
                    section_cfg[key] = default_value

    combined: Dict[str, Any] = {}
    for section in ("dataset", "model", "training", "misc"):
        combined.update(config.get(section, {}))

    args = SimpleNamespace(**combined)
    args.fast_dev_run = bool(getattr(args, "fast_dev_run", False))
    args.use_mask_padding = bool(getattr(args, "use_mask_padding", False))
    args.use_vicreg = bool(getattr(args, "use_vicreg", False))
    args.pass_target_mask_to_predictor = bool(getattr(args, "pass_target_mask_to_predictor", False))

    if getattr(args, "limit", None) is not None:
        args.limit = int(args.limit)
    if getattr(args, "limit_batches", None) is not None:
        args.limit_batches = int(args.limit_batches)
    args.max_epochs = int(args.max_epochs)
    args.num_segments = int(args.num_segments)
    args.max_len = int(args.max_len)

    args.lr = float(args.lr)
    args.vicreg_sim_weight = float(args.vicreg_sim_weight)
    args.vicreg_var_weight = float(args.vicreg_var_weight)
    args.vicreg_cov_weight = float(args.vicreg_cov_weight)
    args.vicreg_loss_ratio = float(args.vicreg_loss_ratio)
    args.masking_probability = float(args.masking_probability)
    args.segment_size_ratio = float(args.segment_size_ratio)
    args.jepa_context_ratio_start = float(args.jepa_context_ratio_start)
    args.jepa_context_ratio_end = float(args.jepa_context_ratio_end)
    args.tokenization = str(getattr(args, "tokenization", "remi"))
    if getattr(args, "output_dir", None) is None:
        args.output_dir = os.getenv("OUTPUT_DIR", "./output")
    args.output_dir = os.fspath(args.output_dir)
    if getattr(args, "midi_dir", None) is None:
        args.midi_dir = os.getenv("MIDI_DIR", "./dataset/clean_midi/")
    args.midi_dir = os.fspath(args.midi_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    # Initialize logger based on fast_dev_run
    if args.fast_dev_run:
        logger = None
        print("Fast dev run enabled - MLflow logging disabled")
    else:
        logger = MLFlowLogger(
            experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME', 'symjepa'),
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI', '/root/Sym-JEPA/mlruns'),
            run_name=args.run_name
        )

   

    # Prepare the data module
    midi_files = glob(os.path.join(args.midi_dir, "**/*.mid"), recursive=True)
    print(f"MIDI glob: {os.path.join(args.midi_dir, '**/*.mid')}")
    print(f"Found {len(midi_files)} MIDI files")
    if args.limit:
        midi_files = midi_files[:args.limit]
        print(f"Limiting to {args.limit} files")

    
    data_module = MidiDataModule(
        midi_files, 
        max_len=args.max_len,
        jepa_context_ratio_start=args.jepa_context_ratio_start,
        jepa_context_ratio_end=args.jepa_context_ratio_end,
        num_epochs=args.max_epochs,
        use_mask_padding=args.use_mask_padding,
        masking_mode=args.masking_mode,
        masking_probability=args.masking_probability,
        segment_size_ratio=args.segment_size_ratio,
        num_segments=args.num_segments,
        tokenization=args.tokenization,
        genre_map='/root/Sym-JEPA/metadata/midi_genre_map.json'
    )

    data_module.setup()

    GRADIENT_ACCUMULATION_N_BATCHES= 25


    # Create the model instance
    model = SymJEPA(
        num_epochs=args.max_epochs,
        use_vicreg=args.use_vicreg,
        vicreg_sim_weight=args.vicreg_sim_weight,
        vicreg_var_weight=args.vicreg_var_weight,
        vicreg_cov_weight=args.vicreg_cov_weight,
        vicreg_loss_ratio=args.vicreg_loss_ratio,
        lr_schedule='cosine',
        max_steps=len(data_module.train_dataloader()) * args.max_epochs // GRADIENT_ACCUMULATION_N_BATCHES,
        tokenization=args.tokenization,
        pass_target_mask_to_predictor=args.pass_target_mask_to_predictor,
        lr=args.lr
    )

    

    # Configure the PyTorch Lightning Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        dirpath=os.path.join(args.output_dir, "checkpoints")
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_batches if args.limit_batches else 1.0,
        limit_val_batches=args.limit_batches if args.limit_batches else 1.0,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=GRADIENT_ACCUMULATION_N_BATCHES,
        gradient_clip_val=1.0,
        log_every_n_steps=max(1, 50 // GRADIENT_ACCUMULATION_N_BATCHES),
        val_check_interval=0.25,  # Run validation 10 times per epoch
        default_root_dir=args.output_dir
    )

    # Log hyperparameters only if not in fast_dev_run mode
    if not args.fast_dev_run and logger is not None:
        logger.log_hyperparams({
            **model.hparams,
            'jepa_context_ratio_start': args.jepa_context_ratio_start,
            'jepa_context_ratio_end': args.jepa_context_ratio_end,
            'train_dataset_size': len(data_module.train_dataloader().dataset),
            'val_dataset_size': len(data_module.val_dataloader().dataset),
            'test_dataset_size': len(data_module.test_dataloader().dataset),
            'skipped_files': len(data_module.skipped_files),
            'total_files': len(data_module.files)
        })

    # Begin training
    trainer.fit(model, data_module)

    # Collect metrics before optional downstream fine-tuning
    metrics = _extract_scalar_metrics(trainer.callback_metrics)

    best_model_path = None
    if checkpoint_callback is not None:
        best_model_path = getattr(checkpoint_callback, "best_model_path", None) or getattr(checkpoint_callback, "last_model_path", None)
    skipped_files = len(getattr(data_module, "skipped_files", []))

    return {
        "best_model_path": best_model_path,
        "metrics": metrics,
        "skipped_files": skipped_files,
        "train_dataset_size": len(getattr(data_module, "train_ds", [])),
        "val_dataset_size": len(getattr(data_module, "valid_ds", [])),
        "test_dataset_size": len(getattr(data_module, "test_ds", [])),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Sym-JEPA using a YAML configuration file.")
    parser.add_argument("--config", type=str, help="Path to YAML config describing the training run.")
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Override config to enable Lightning fast_dev_run for smoke testing.",
    )
    cli_args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    if cli_args.fast_dev_run:
        overrides.setdefault("training", {})["fast_dev_run"] = True

    config = load_train_config(cli_args.config, overrides=overrides)
    run_training(config)


if __name__ == "__main__":
    main() 
