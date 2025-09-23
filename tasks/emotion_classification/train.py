from .model import EmotionClassifier
from .dataset import EMOPIADataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import os
import torch
from glob import glob
torch.set_float32_matmul_precision('high')

def train(args):
    root_path = os.curdir
    midi_files = glob(os.path.join("dataset/EMOPIA_1.0/midis", "**/*.mid"), recursive=True)[:20000]
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

    trainer = pl.Trainer(
        max_epochs=400,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True)
        ],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        num_sanity_val_steps=10,
        accumulate_grad_batches=200,
        log_every_n_steps=max(1, 50 // 50)
    )

    # trainer.validate(model, data_module)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
