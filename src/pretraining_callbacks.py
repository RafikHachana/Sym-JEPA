"""Lightweight PyTorch Lightning callback for Sym-JEPA validation projections."""

from __future__ import annotations

import os
import tempfile
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch

from src.viz import visualize_tsne_clusters, visualize_umap_clusters


class EmbeddingProjectionCallback(pl.Callback):
    """Collect a small batch of embeddings and log UMAP/t-SNE projections."""

    def __init__(
        self,
        *,
        stage: str = "validate",
        max_sequences: int = 10,
        artifact_path: str = "visualizations",
        dataloader_idx: int = 0,
        figure_size: Sequence[float] = (12, 8),
        dpi: int = 150,
    ) -> None:
        super().__init__()
        valid_stages = {"train", "validate", "test"}
        if stage not in valid_stages:
            raise ValueError(f"stage must be one of {valid_stages}, got '{stage}'")

        self.stage = stage
        self.max_sequences = max_sequences
        self.artifact_path = artifact_path
        self.dataloader_idx = dataloader_idx
        self.figure_size = tuple(figure_size)
        self.dpi = dpi

        self._reset()

    # Lightning hook dispatch -------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.stage == "train":
            self._collect(trainer, pl_module, batch, batch_idx)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        if self.stage == "validate" and dataloader_idx == self.dataloader_idx:
            self._collect(trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        if self.stage == "test" and dataloader_idx == self.dataloader_idx:
            self._collect(trainer, pl_module, batch, batch_idx)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.stage == "train":
            self._log(trainer, stage="train")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.stage == "validate":
            self._log(trainer, stage="validate")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if self.stage == "test":
            self._log(trainer, stage="test")

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self._reset()

    # Internal helpers --------------------------------------------------------
    def _collect(self, trainer, pl_module, batch: Any, batch_idx: int) -> None:
        if trainer.sanity_checking:
            return
        if len(self._sequences) >= self.max_sequences:
            return
        if not isinstance(batch, dict):
            return

        context_ids = batch.get("context_ids")
        if context_ids is None:
            return

        context_mask = batch.get("context_mask")

        if not hasattr(pl_module, "encode_context"):
            raise AttributeError(
                "EmbeddingProjectionCallback expects the LightningModule to "
                "implement 'encode_context'."
            )

        with torch.no_grad():
            embeddings = pl_module.encode_context(context_ids, context_mask)

        embeddings_np = embeddings.detach().cpu().numpy()

        # Treat tensors of shape (batch, seq, dim). Degenerate shapes are promoted.
        if embeddings_np.ndim == 2:
            embeddings_np = embeddings_np[None, ...]
        elif embeddings_np.ndim != 3:
            raise ValueError(
                f"encode_context returned tensor with unexpected shape {embeddings_np.shape}"
            )

        for seq_array in embeddings_np:
            if len(self._sequences) >= self.max_sequences:
                break

            self._sequences.append(np.asarray(seq_array))
            seq_id = len(self._sequences) - 1
            labels = np.full(seq_array.shape[0], seq_id, dtype=np.int64)
            self._labels.append(labels)

    def _log(self, trainer: pl.Trainer, stage: str) -> None:
        if trainer.sanity_checking:
            return
        if not self._sequences:
            return

        all_embeddings = np.vstack(self._sequences)
        all_labels = np.concatenate(self._labels)

        epoch = trainer.current_epoch

        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, display, func in (
                ("umap", "UMAP", visualize_umap_clusters),
                ("tsne", "t-SNE", visualize_tsne_clusters),
            ):
                try:
                    self._render_and_log(
                        tmp_dir=tmp_dir,
                        name=name,
                        title=f"{display} Projection ({stage}) - Epoch {epoch}",
                        func=func,
                        embeddings=all_embeddings,
                        labels=all_labels,
                        trainer=trainer,
                        epoch=epoch,
                        stage=stage,
                    )
                except Exception as exc:
                    print(
                        f"[EmbeddingProjectionCallback] Failed to create {name.upper()} "
                        f"projection: {exc}"
                    )

        self._reset()

    def _render_and_log(
        self,
        *,
        tmp_dir: str,
        name: str,
        title: str,
        func,
        embeddings: np.ndarray,
        labels: np.ndarray,
        trainer: pl.Trainer,
        epoch: int,
        stage: str,
    ) -> None:
        plt.figure(figsize=self.figure_size)
        func(embeddings, labels, mode="2d")
        plt.suptitle(title, y=0.98)

        output_path = os.path.join(tmp_dir, f"{name}_{stage}_epoch_{epoch}.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        self._log_artifact(trainer, output_path)

    def _log_artifact(self, trainer: pl.Trainer, local_path: str) -> None:
        if trainer is None:
            self._log_with_mlflow(local_path)
            return

        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        logger = trainer.logger
        if logger is not None:
            experiment = getattr(logger, "experiment", None)
            run_id = getattr(logger, "run_id", None)
            log_artifact = getattr(experiment, "log_artifact", None)
            if callable(log_artifact) and run_id is not None:
                try:
                    log_artifact(run_id, local_path, artifact_path=self.artifact_path)
                    return
                except Exception:
                    pass

        self._log_with_mlflow(local_path)

    def _log_with_mlflow(self, local_path: str) -> None:
        if mlflow.active_run() is not None:
            try:
                mlflow.log_artifact(local_path, artifact_path=self.artifact_path)
            except Exception:
                pass

    def _reset(self) -> None:
        self._sequences: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
