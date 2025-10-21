import numbers
import os
from glob import glob

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from .dataset import EMOPIADataModule
from .model import EmotionClassifier
torch.set_float32_matmul_precision('high')


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


def train(args):
    root_path = os.curdir
    midi_files = glob(os.path.join("dataset/EMOPIA_1.0/midis", "**/*.mid"), recursive=True)[:20000]
    os.makedirs(args.output_dir, exist_ok=True)
    data_module = EMOPIADataModule(
        midi_files,
        batch_size=64,
        num_workers=4,
    )

    data_module.setup()
    
    model = EmotionClassifier(
        lr=1e-7,
        d_model=512,
        encoder_layers=8,
        tokenization=args.tokenization,
        class_key='emotional_quadrant',
        num_classes=4,  # Assuming 9 difficulties if DIFFICULTY_IN_META is True
    )

    if args.model_path:
        model.load_jepa(args.model_path)

    if args.fast_dev_run:
        logger = None
    else:
        logger = MLFlowLogger(
            experiment_name=f"symjepa-{args.task}-cls",
            tracking_uri="./mlruns",
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
        max_epochs=400,
        callbacks=[checkpoint_callback],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        num_sanity_val_steps=10,
        accumulate_grad_batches=200,
        log_every_n_steps=max(1, 50 // 50),
        default_root_dir=args.output_dir
    )

    # trainer.validate(model, data_module)

    trainer.fit(model, data_module)

    metrics = _extract_scalar_metrics(trainer.callback_metrics)
    best_model_path = None
    if checkpoint_callback is not None:
        best_model_path = getattr(checkpoint_callback, "best_model_path", None) or getattr(checkpoint_callback, "last_model_path", None)

    return {
        "task": args.task,
        "metrics": metrics,
        "best_model_path": best_model_path,
    }


if __name__ == "__main__":
    train()
