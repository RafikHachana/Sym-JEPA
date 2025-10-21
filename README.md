# Sym-JEPA
JEPA experiments for symbolic music representation learning


## Environment Setup

```bash
conda env create -f environment.yml
conda activate symjepa
```

## Ray Tune Pipeline

You can launch end-to-end pretraining followed by downstream fine-tuning sweeps with Ray Tune:

```bash
python -m src.ray_pipeline \
  --midi-dir dataset/clean_midi \
  --tasks genre performer \
  --seeds 42 1337 \
  --max-pretrain-epochs 5 \
  --max-finetune-epochs 3 \
  --cpus-per-trial 8 \
  --gpus-per-trial 1
```

Each Ray trial trains Sym-JEPA, saves checkpoints under the trial directory, and fine-tunes the requested tasks using the resulting weights. Metrics are reported back to Ray for comparison across seeds or additional hyper-parameter samples.
