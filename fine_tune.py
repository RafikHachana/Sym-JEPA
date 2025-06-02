from tasks.genre_classification import GenreClassificationModel
from tasks.melody_completion import MelodyCompletionModel
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
    choices=['genre', 'style', 'melody_completion'], help='Task to fine-tune on')

    parser.add_argument("--fast_dev_run", action="store_true",
                       help="Do a test run with 1 batch for training and validation")
                      
    args = parser.parse_args()

    midi_files = glob(os.path.join("dataset/lmd_full", "**/*.mid"), recursive=True)[:50000]
    data_module = MidiDataModule(
        midi_files,
        max_len=2048,
        batch_size=32,
        num_workers=4,
        jepa_context_ratio_start=0.975,
        jepa_context_ratio_end=0.975,
        jepa_context_ratio_steps=100,
        skip_unknown_genres=args.task == 'genre',
        skip_unknown_styles=args.task == 'style',
        tokenization=args.tokenization,
        generate_melody_completion_pairs=args.task == 'melody_completion'
    )

    data_module.setup()

    if args.task == 'genre' or args.task == 'style':
        model = GenreClassificationModel(
            num_classes=len(data_module.train_ds.all_genres) if args.task == 'genre' else len(data_module.train_ds.all_styles),
            lr=1e-3,
            d_model=512,
            encoder_layers=8,
            num_attention_heads=8,
        tokenization=args.tokenization,
        class_weights=torch.tensor([1/(x+1e-7) for x in data_module.train_ds.genre_counts]) if args.task == 'genre' else torch.tensor([1/(x+1e-7) for x in data_module.train_ds.style_counts]),
        task=args.task
        )
    elif args.task == 'melody_completion':
        model = MelodyCompletionModel(
            lr=1e-2,
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
        fast_dev_run=args.fast_dev_run
    )

    trainer.validate(model, data_module)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
