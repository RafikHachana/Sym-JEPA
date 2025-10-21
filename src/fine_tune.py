import argparse
import numbers
import os
from glob import glob

from tasks.genre_classification import GenreClassificationModel
from tasks.melody_completion import MelodyCompletionModel, train as train_melody_completion
from tasks.performer_composer_classification import train as train_performer_composer_classification
from tasks.generate_midi import MusicDecoder
from tasks.emotion_classification import train as train_emotion
from src.dataset import MidiDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import torch
torch.set_float32_matmul_precision('high')

DEFAULT_FINE_TUNE_OUTPUT = os.getenv('FINE_TUNE_OUTPUT_DIR', './output/fine_tune')


def build_fine_tune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_FINE_TUNE_OUTPUT)
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

    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for logging")
    return parser


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


def run_fine_tuning(args: argparse.Namespace) -> dict:
    os.makedirs(args.output_dir, exist_ok=True)

    if args.task in {'melody_completion', 'accompaniment_suggestion'}:
        result = train_melody_completion(args)
        return result or {"task": args.task}
    
    if args.task in {'performer', 'composer', 'difficulty'}:
        result = train_performer_composer_classification(args)
        return result or {"task": args.task}

    if args.task == 'emotion':
        result = train_emotion(args)
        return result or {"task": args.task}

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

    if args.task in {'genre', 'style'}:
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
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    if args.model_path:
        model.load_jepa(args.model_path)

    if args.fast_dev_run:
        logger = None
    else:
        experiment_name = f"symjepa-{args.task}-classification" if args.task != "decode" else "symjepa-decode"
        logger = MLFlowLogger(
            experiment_name=experiment_name,
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
        max_epochs=args.max_epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step')
        ],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=200,
        log_every_n_steps=max(1, 50 // 50),
        default_root_dir=args.output_dir
    )

    trainer.validate(model, data_module)

    trainer.fit(model, data_module)

    metrics = _extract_scalar_metrics(trainer.callback_metrics)
    best_model_path = None
    if checkpoint_callback is not None:
        best_model_path = getattr(checkpoint_callback, "best_model_path", None) or getattr(checkpoint_callback, "last_model_path", None)

    return {
        "task": args.task,
        "metrics": metrics,
        "best_model_path": best_model_path,
        "train_dataset_size": len(getattr(data_module, "train_ds", [])),
        "val_dataset_size": len(getattr(data_module, "valid_ds", [])),
    }


def main():
    parser = build_fine_tune_parser()
    args = parser.parse_args()
    run_fine_tuning(args)


if __name__ == "__main__":
    main()
