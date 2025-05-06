from tasks.genre_classification import GenreClassificationModel
from dataset import MidiDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import argparse
from glob import glob
import os
import torch
torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()


    model = GenreClassificationModel(
        num_genres=13,
        lr=1e-4,
        d_model=512,
        encoder_layers=8,
        num_attention_heads=8,
    )

    model.load_encoder(args.model_path, "remi_in.ckpt")

    logger = WandbLogger(
        project="symjepa-genre-classification",
        entity="rh-iu",
    )
    midi_files = glob(os.path.join("dataset/clean_midi", "**/*.mid"), recursive=True)

    data_module = MidiDataModule(
        midi_files,
        max_len=512,
        batch_size=32,
        num_workers=4,
        skip_unknown_genres=True,
    )


    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True)
        ],
        logger=logger
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
