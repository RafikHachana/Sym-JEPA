import argparse
import numbers
import os
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from dotenv import load_dotenv
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
# Enable better Tensor Core utilization on NVIDIA GPUs
torch.set_float32_matmul_precision('high')

DEFAULT_OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')

from src.model import SymJEPA
from src.dataset import MidiDataModule

def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("model")
    parser.add_argument('--tokenization', type=str, default='remi',
                      choices=['remi', 'octuple'],
                      help='Tokenization method to use (remi or octuple)')

    parser.add_argument("--jepa_context_ratio_start", type=float,
                       default=float(os.getenv('JEPA_CONTEXT_RATIO_START', 0.95)),
                       help="Ratio of sequence length to use as context at the start of training (default: 0.98)")
    parser.add_argument("--jepa_context_ratio_end", type=float,
                       default=float(os.getenv('JEPA_CONTEXT_RATIO_END', 0.6)),
                       help="Ratio of sequence length to use as context at the end of training (default: 0.9)")

    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of files to load")
    parser.add_argument("--fast_dev_run", action="store_true",
                       help="Do a test run with 1 batch for training and validation")
    parser.add_argument("--limit_batches", type=int, default=None,
                       help="Limit number of batches per epoch (for testing)")
    parser.add_argument("--use_mask_padding", action="store_true",
                       help="Use MASK tokens instead of PAD tokens for JEPA context/target")
    parser.add_argument("--masking_mode", type=str, 
                       choices=['contiguous', 'random', 'segments', 'random_contiguous'],
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
    parser.add_argument("--vicreg_sim_weight", type=float, default=1.0,
                       help="Weight for VicReg similarity loss")
    parser.add_argument("--vicreg_var_weight", type=float, default=40.0,
                       help="Weight for VicReg variance loss")
    parser.add_argument("--vicreg_cov_weight", type=float, default=10.0,
                       help="Weight for VicReg covariance loss")
    parser.add_argument("--vicreg_loss_ratio", type=float, default=0.3,
                       help="Target ratio of VICReg loss to JEPA loss (default: 0.3)")
    parser.add_argument("--pass_target_mask_to_predictor", action="store_true",
                       help="Pass target mask to predictor")
    parser.add_argument("--max_len", type=int, default=2048,
                       help="Maximum length of the input sequence")
    parser.add_argument("--lr", type=float, default=5e-2,
                       help="Learning rate")
    return parent_parser


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Sym-JEPA with MLflow logging")
    parser = add_model_specific_args(parser)
    parser.add_argument("--midi_dir", type=str,
                       default=os.getenv('MIDI_DIR', './dataset/clean_midi/'),
                       help="Directory containing MIDI files")
    parser.add_argument("--max_epochs", type=int,
                       default=int(os.getenv('MAX_EPOCHS', 100)),
                       help="Number of training epochs")
    parser.add_argument("--train_downstream_tasks", action="store_true",
                       help="Automatically train and log downstream task results after pretraining.")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for logging")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Root directory for saving checkpoints and downstream outputs")
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

def run_downstream_task(task, model_path, output_dir, tokenization, max_epochs=10, run_name=None):
    """Run a single downstream task using subprocess."""
    print(f"Starting downstream task: {task}")
    
    cmd = [
        sys.executable, "-m", "src.fine_tune",
        "--task", task,
        "--output_dir", output_dir,
        "--tokenization", tokenization,
        "--max_epochs", "3",
        "--run_name", run_name
    ]
    
    if model_path:
        cmd.extend(["--model_path", model_path])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Completed downstream task: {task}")
        return {"task": task, "success": True, "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed downstream task: {task}")
        print(f"Error: {e.stderr}")
        return {"task": task, "success": False, "error": e.stderr}

def run_all_downstream_tasks(model_path, output_dir, tokenization, max_epochs=10, max_workers=3, run_name=None):
    """Run all downstream tasks with limited concurrency."""
    downstream_tasks = [
        'genre', 'style', 'melody_completion',
        'performer',
        'composer',
        # 'decode',
        'accompaniment_suggestion',
        # 'difficulty', 'emotion'
    ]
    
    print(f"Starting {len(downstream_tasks)} downstream tasks with max {max_workers} concurrent processes...")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_downstream_task, task, model_path, output_dir, tokenization, max_epochs, run_name): task
            for task in downstream_tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            print(f"Progress: {len(results)}/{len(downstream_tasks)} tasks completed")
    
    # Summary
    successful_tasks = [r for r in results if r["success"]]
    failed_tasks = [r for r in results if not r["success"]]
    
    print(f"\n=== Downstream Tasks Summary ===")
    print(f"✓ Successful: {len(successful_tasks)}/{len(downstream_tasks)}")
    if failed_tasks:
        print(f"✗ Failed: {len(failed_tasks)}")
        for task in failed_tasks:
            print(f"  - {task['task']}")
    
    return results

def run_training(args: argparse.Namespace) -> dict:
    # Load environment variables from .env file
    load_dotenv()

    os.makedirs(args.output_dir, exist_ok=True)
    # Initialize logger based on fast_dev_run
    if args.fast_dev_run:
        logger = None
        print("Fast dev run enabled - MLflow logging disabled")
    else:
        logger = MLFlowLogger(
            experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME', 'symjepa'),
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI', './mlruns'),
            run_name=args.run_name
        )

   

    # Prepare the data module
    midi_files = glob(os.path.join(args.midi_dir, "**/*.mid"), recursive=True)
    print(f"MIDI glob: {os.path.join(args.midi_dir, '**/*.mid')}")
    print(f"Found {len(midi_files)} MIDI files")
    if args.limit:
        midi_files = midi_files[:args.limit]
        print(f"Limiting to {args.limit} files")

    
    data_module = MidiDataModule(
        midi_files, 
        max_len=args.max_len,
        jepa_context_ratio_start=args.jepa_context_ratio_start,
        jepa_context_ratio_end=args.jepa_context_ratio_end,
        num_epochs=args.max_epochs,
        use_mask_padding=args.use_mask_padding,
        masking_mode=args.masking_mode,
        masking_probability=args.masking_probability,
        segment_size_ratio=args.segment_size_ratio,
        num_segments=args.num_segments,
        tokenization=args.tokenization
    )

    data_module.setup()

    GRADIENT_ACCUMULATION_N_BATCHES= 25


    # Create the model instance
    model = SymJEPA(
        num_epochs=args.max_epochs,
        use_vicreg=args.use_vicreg,
        vicreg_sim_weight=args.vicreg_sim_weight,
        vicreg_var_weight=args.vicreg_var_weight,
        vicreg_cov_weight=args.vicreg_cov_weight,
        vicreg_loss_ratio=args.vicreg_loss_ratio,
        lr_schedule='cosine',
        max_steps=len(data_module.train_dataloader()) * args.max_epochs // GRADIENT_ACCUMULATION_N_BATCHES,
        tokenization=args.tokenization,
        pass_target_mask_to_predictor=args.pass_target_mask_to_predictor,
        lr=args.lr
    )

    

    # Configure the PyTorch Lightning Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        dirpath=os.path.join(args.output_dir, "checkpoints")
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_batches if args.limit_batches else 1.0,
        limit_val_batches=args.limit_batches if args.limit_batches else 1.0,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=GRADIENT_ACCUMULATION_N_BATCHES,
        gradient_clip_val=1.0,
        log_every_n_steps=max(1, 50 // GRADIENT_ACCUMULATION_N_BATCHES),
        val_check_interval=0.25,  # Run validation 10 times per epoch
        default_root_dir=args.output_dir
    )

    # Log hyperparameters only if not in fast_dev_run mode
    if not args.fast_dev_run and logger is not None:
        logger.log_hyperparams({
            **model.hparams,
            'jepa_context_ratio_start': args.jepa_context_ratio_start,
            'jepa_context_ratio_end': args.jepa_context_ratio_end,
            'train_dataset_size': len(data_module.train_dataloader().dataset),
            'val_dataset_size': len(data_module.val_dataloader().dataset),
            'test_dataset_size': len(data_module.test_dataloader().dataset),
            'skipped_files': len(data_module.skipped_files),
            'total_files': len(data_module.files)
        })

    # Begin training
    trainer.fit(model, data_module)

    # Collect metrics before optional downstream fine-tuning
    metrics = _extract_scalar_metrics(trainer.callback_metrics)

    best_model_path = None
    if checkpoint_callback is not None:
        best_model_path = getattr(checkpoint_callback, "best_model_path", None) or getattr(checkpoint_callback, "last_model_path", None)
    downstream_results = None

    # Run downstream tasks if requested
    if args.train_downstream_tasks:
        print("\n=== Starting Downstream Task Training ===")
        
        # Get the best checkpoint path
        if best_model_path:
            print(f"Using best checkpoint: {best_model_path}")
        else:
            print("No best checkpoint found, running downstream tasks without pretrained weights")
        
        # Create output directory for downstream tasks
        downstream_output_dir = os.path.join(args.output_dir, 'downstream_tasks')
        os.makedirs(downstream_output_dir, exist_ok=True)
        
        # Run all downstream tasks
        downstream_results = run_all_downstream_tasks(
            model_path=best_model_path,
            output_dir=downstream_output_dir,
            tokenization=args.tokenization,
            max_epochs=10,  # Default 10 epochs for downstream tasks
            max_workers=3,
            run_name=args.run_name
        )
        
        # Log downstream results to MLflow if logger is available
        if logger is not None:
            successful_count = sum(1 for r in downstream_results if r["success"])
            logger.experiment.log_metric(
                run_id=logger.run_id,
                key="downstream_tasks_successful",
                value=successful_count
            )
            logger.experiment.log_metric(
                run_id=logger.run_id,
                key="downstream_tasks_total",
                value=len(downstream_results)
            )

    skipped_files = len(getattr(data_module, "skipped_files", []))

    return {
        "best_model_path": best_model_path,
        "metrics": metrics,
        "downstream_results": downstream_results,
        "skipped_files": skipped_files,
        "train_dataset_size": len(getattr(data_module, "train_ds", [])),
        "val_dataset_size": len(getattr(data_module, "valid_ds", [])),
        "test_dataset_size": len(getattr(data_module, "test_ds", [])),
    }


def main():
    parser = build_train_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main() 
