import numbers
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from .atepp_dataset import PerformerClassificationDataModule
from .model import PerformerClassifier
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
    midi_base_path=os.path.join(root_path, 'dataset/ATEPP-1.2')
    metadata_path=os.path.join(root_path, 'dataset/ATEPP-metadata-1.2.csv')
    os.makedirs(args.output_dir, exist_ok=True)
    data_module = PerformerClassificationDataModule(
        midi_base_path=midi_base_path,
        metadata_path=metadata_path,
        batch_size=64,
        num_workers=4,
        top_k_composers=9 if args.task == 'composer' else None,
        top_k_performers=20 if args.task == 'performer' else None
    )

    data_module.setup()
    
    model = PerformerClassifier(
        lr=1e-4,
        d_model=512,
        encoder_layers=8,
        tokenization=args.tokenization,
        class_key=args.task,
        num_classes=len(data_module.train_dataset.unique_performers) if args.task == "performer"
        else len(data_module.train_dataset.unique_composers) if args.task == "composer" else 9,  # Assuming 9 difficulties if DIFFICULTY_IN_META is True
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
        max_epochs=40,
        callbacks=[checkpoint_callback],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        num_sanity_val_steps=10,
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
