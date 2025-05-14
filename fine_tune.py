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
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument('--tokenization', type=str, default='remi',
                      choices=['remi', 'octuple'],
                      help='Tokenization method to use (remi or octuple)')
                      
    args = parser.parse_args()

    model = GenreClassificationModel(
        num_genres=13,
        lr=1e-4,
        d_model=512,
        encoder_layers=8,
        num_attention_heads=8,
        tokenization=args.tokenization
    )

    if args.model_path:
        model.load_jepa(args.model_path)

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
        tokenization=args.tokenization
    )


    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True)
        ],
        logger=logger
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
