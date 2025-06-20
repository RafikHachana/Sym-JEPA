from .model import MelodyCompletionModel
from .dataset import MelodyPredictionDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import argparse
from glob import glob
import os
import torch
torch.set_float32_matmul_precision('high')

def train(args):
    midi_files = glob(os.path.join("dataset/clean_midi", "**/*.mid"), recursive=True)[:20000]
    print("Len of midi files: ", len(midi_files))
    data_module = MelodyPredictionDataModule(
        midi_files,
        batch_size=64,
        num_workers=4,
    )
    
    model = MelodyCompletionModel(
        lr=1e-5,
        d_model=512,
        encoder_layers=8,
        tokenization=args.tokenization
    )

    if args.model_path:
        model.load_jepa(args.model_path)

    if args.fast_dev_run:
        logger = None
    else:
        logger = WandbLogger(
            project=f"symjepa-{args.task}-classification",
            entity="rh-iu",
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True)
        ],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        num_sanity_val_steps=10
    )

    # trainer.validate(model, data_module)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
