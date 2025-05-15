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

    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--task', type=str, default='genre',
    choices=['genre', 'style'], help='Task to fine-tune on')
                      
    args = parser.parse_args()

    midi_files = glob(os.path.join("dataset/clean_midi", "**/*.mid"), recursive=True)
    data_module = MidiDataModule(
        midi_files,
        max_len=512,
        batch_size=32,
        num_workers=4,
        skip_unknown_genres=args.task == 'genre',
        skip_unknown_styles=args.task == 'style',
        tokenization=args.tokenization
    )

    data_module.setup()

    model = GenreClassificationModel(
        num_classes=len(data_module.train_ds.all_genres) if args.task == 'genre' else len(data_module.train_ds.all_styles),
        lr=1e-2,
        d_model=512,
        encoder_layers=8,
        num_attention_heads=8,
        tokenization=args.tokenization,
        class_weights=torch.tensor([1/(x+1e-7) for x in data_module.train_ds.genre_counts]),
        task=args.task
    )

    if args.model_path:
        model.load_jepa(args.model_path)

    logger = WandbLogger(
        project=f"symjepa-{args.task}-classification",
        entity="rh-iu",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True)
        ],
        logger=logger
    )

    trainer.validate(model, data_module)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
