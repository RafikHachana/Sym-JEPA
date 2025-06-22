from tasks.genre_classification import GenreClassificationModel
from tasks.melody_completion import MelodyCompletionModel, train as train_melody_completion
from tasks.performer_composer_classification import train as train_performer_composer_classification
from tasks.generate_midi import MusicDecoder
from tasks.emotion_classification import train as train_emotion
from dataset import MidiDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
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

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--task', type=str, default='genre',
    choices=['genre', 'style', 'melody_completion',
             'performer', 'composer', 'decode',
             'accompaniment_suggestion', 'difficulty', 'emotion'], help='Task to fine-tune on')

    parser.add_argument("--fast_dev_run", action="store_true",
                       help="Do a test run with 1 batch for training and validation")

    parser.add_argument("--limit_data", type=int, default=None,
                       help="Limit the number of data points to use")
                      
    args = parser.parse_args()

    if args.task == 'melody_completion' or args.task == 'accompaniment_suggestion':
        return train_melody_completion(args)
    
    if args.task == 'performer' or args.task == 'composer':
        return train_performer_composer_classification(args)

    if args.task == 'emotion':
        return train_emotion(args)

    midi_files = glob(os.path.join("dataset/lmd_full", "**/*.mid"), recursive=True)
    if args.limit_data:
        midi_files = midi_files[:args.limit_data]

    data_module = MidiDataModule(
        midi_files,
        max_len=2048,
        batch_size=16,
        num_workers=4,
        jepa_context_ratio_start=0.975,
        jepa_context_ratio_end=0.975,
        jepa_context_ratio_steps=100,
        skip_unknown_genres=args.task == 'genre',
        skip_unknown_styles=args.task == 'style',
        tokenization=args.tokenization,
        masking_probabilities={
            "none": 1.0,
        }
    )

    data_module.setup()

    if args.task == 'genre' or args.task == 'style':
        model = GenreClassificationModel(
            num_classes=len(data_module.train_ds.all_genres) if args.task == 'genre' else len(data_module.train_ds.all_styles),
            lr=5e-7,
            d_model=512,
            encoder_layers=8,
            num_attention_heads=8,
        tokenization=args.tokenization,
        class_weights=torch.tensor([1/(x+1e-7) for x in data_module.train_ds.genre_counts]) if args.task == 'genre' else torch.tensor([1/(x+1e-7) for x in data_module.train_ds.style_counts]),
        task=args.task
        )
    elif args.task == 'decode':
        model = MusicDecoder(
            lr=1e-6
        )

    if args.model_path:
        model.load_jepa(args.model_path)

    if args.fast_dev_run:
        logger = None
    else:
        project_name = f"symjepa-{args.task}-classification" if args.task != "decode" else "symjepa-decode"
        logger = WandbLogger(
            project=project_name,
            entity="rh-iu",
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True),
            LearningRateMonitor(logging_interval='step')
        ],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=200,
        log_every_n_steps=max(1, 50 // 50)
    )

    trainer.validate(model, data_module)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
