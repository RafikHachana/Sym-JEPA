import argparse
import os
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv
import torch

# Enable better Tensor Core utilization on NVIDIA GPUs
torch.set_float32_matmul_precision('high')

from model import SymJEPA
from dataset import MidiDataModule

def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train Sym-JEPA with Weights & Biases logging")
    parser.add_argument("--midi_dir", type=str, 
                       default=os.getenv('MIDI_DIR', './dataset'),
                       help="Directory containing MIDI files")
    parser.add_argument("--max_epochs", type=int,
                       default=int(os.getenv('MAX_EPOCHS', 100)),
                       help="Number of training epochs")
    parser.add_argument("--jepa_context_ratio", type=float,
                       default=float(os.getenv('JEPA_CONTEXT_RATIO', 0.75)),
                       help="Ratio of sequence length to use as context (default: 0.75)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of files to load")
    args = parser.parse_args()

    # Initialize the Wandb logger with values from .env
    wandb_logger = WandbLogger(
        project=os.getenv('WANDB_PROJECT', 'symjepa'),
        entity=os.getenv('WANDB_ENTITY'),
        log_model=True
    )

    # Create the model instance
    model = SymJEPA(num_epochs=args.max_epochs)

    # Prepare the data module with JEPA context ratio
    midi_files = glob(os.path.join(args.midi_dir, "**/*.mid"), recursive=True)
    if args.limit:
        midi_files = midi_files[:args.limit]
        print(f"Limiting to {args.limit} files")
    data_module = MidiDataModule(
        midi_files, 
        max_len=512,
        jepa_context_ratio=args.jepa_context_ratio
    )

    # Configure the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
    )

    # Log hyperparameters including JEPA context ratio
    wandb_logger.log_hyperparams({
        **model.hparams,
        'jepa_context_ratio': args.jepa_context_ratio
    })

    # Begin training
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main() 