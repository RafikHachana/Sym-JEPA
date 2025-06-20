from .model import PerformerClassifier
from .atepp_dataset import PerformerClassificationDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
import torch
torch.set_float32_matmul_precision('high')

def train(args):
    root_path = os.curdir
    midi_base_path=os.path.join(root_path, 'dataset/ATEPP-1.2')
    metadata_path=os.path.join(root_path, 'dataset/ATEPP-metadata-1.2.csv')
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
        num_classes=len(data_module.train_dataset.unique_performers) if args.task == "performer" else len(data_module.train_dataset.unique_composers),
    )

    if args.model_path:
        model.load_jepa(args.model_path)

    if args.fast_dev_run:
        logger = None
    else:
        logger = WandbLogger(
            project=f"symjepa-{args.task}-cls",
            entity="rh-iu",
        )

    trainer = pl.Trainer(
        max_epochs=40,
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
