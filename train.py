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
    parser.add_argument("--fast_dev_run", action="store_true",
                       help="Do a test run with 1 batch for training and validation")
    parser.add_argument("--limit_batches", type=int, default=None,
                       help="Limit number of batches per epoch (for testing)")
    parser.add_argument("--use_mask_padding", action="store_true",
                       help="Use MASK tokens instead of PAD tokens for JEPA context/target")
    parser.add_argument("--masking_mode", type=str, 
                       choices=['contiguous', 'random', 'segments'],
                       default='contiguous',
                       help="Masking mode when use_mask_padding is True")
    parser.add_argument("--masking_probability", type=float,
                       default=0.25,
                       help="Probability of masking tokens in random mode")
    parser.add_argument("--segment_size_ratio", type=float,
                       default=0.1,
                       help="Size of segments to mask as fraction of sequence length")
    parser.add_argument("--num_segments", type=int,
                       default=3,
                       help="Number of segments to mask in segments mode")
    parser.add_argument("--use_vicreg", action="store_true",
                       help="Use VicReg loss in addition to JEPA loss")
    parser.add_argument("--vicreg_sim_weight", type=float, default=25.0,
                       help="Weight for VicReg similarity loss")
    parser.add_argument("--vicreg_var_weight", type=float, default=25.0,
                       help="Weight for VicReg variance loss")
    parser.add_argument("--vicreg_cov_weight", type=float, default=1.0,
                       help="Weight for VicReg covariance loss")

    args = parser.parse_args()

    # Initialize logger based on fast_dev_run
    if args.fast_dev_run:
        logger = None
        print("Fast dev run enabled - wandb logging disabled")
    else:
        logger = WandbLogger(
            project=os.getenv('WANDB_PROJECT', 'symjepa'),
            entity=os.getenv('WANDB_ENTITY'),
            log_model=True
        )

    # Create the model instance
    model = SymJEPA(
        num_epochs=args.max_epochs,
        use_vicreg=args.use_vicreg,
        vicreg_sim_weight=args.vicreg_sim_weight,
        vicreg_var_weight=args.vicreg_var_weight,
        vicreg_cov_weight=args.vicreg_cov_weight
    )

    # Prepare the data module
    midi_files = glob(os.path.join(args.midi_dir, "**/*.mid"), recursive=True)
    if args.limit:
        midi_files = midi_files[:args.limit]
        print(f"Limiting to {args.limit} files")
    data_module = MidiDataModule(
        midi_files, 
        max_len=512,
        jepa_context_ratio=args.jepa_context_ratio,
        use_mask_padding=args.use_mask_padding,
        masking_mode=args.masking_mode,
        masking_probability=args.masking_probability,
        segment_size_ratio=args.segment_size_ratio,
        num_segments=args.num_segments
    )

    # Configure the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_batches if args.limit_batches else 1.0,
        limit_val_batches=args.limit_batches if args.limit_batches else 1.0,
    )

    # Log hyperparameters only if not in fast_dev_run mode
    if not args.fast_dev_run and logger is not None:
        logger.log_hyperparams({
            **model.hparams,
            'jepa_context_ratio': args.jepa_context_ratio
        })

    # Begin training
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main() 