import argparse
import os
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import SymJEPA
from dataset import MidiDataModule

def main():
    parser = argparse.ArgumentParser(description="Train Sym-JEPA with Weights & Biases logging")
    parser.add_argument("--project", type=str, default="symjepa", help="Weights & Biases project name")
    parser.add_argument("--entity", type=str, default=None, help="Weights & Biases entity/team name")
    parser.add_argument("--midi_dir", type=str, default="./data", help="Directory containing MIDI files")
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    # Initialize the Wandb logger
    wandb_logger = WandbLogger(project=args.project, entity=args.entity, log_model=True)

    # Create the model instance
    model = SymJEPA(num_epochs=args.max_epochs)

    # Prepare the data module by collecting all MIDI files from the specified directory
    midi_files = glob(os.path.join(args.midi_dir, "**/*.mid"), recursive=True)
    data_module = MidiDataModule(midi_files, max_len=512)

    # Configure the PyTorch Lightning Trainer to use the Wandb logger
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        # You can add additional Trainer parameters here as needed (e.g., accelerator, devices, etc.)
    )

    # Optionally log hyperparameters
    wandb_logger.log_hyperparams(model.hparams)

    # Begin training
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main() 