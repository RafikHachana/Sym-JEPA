"""Lightweight PyTorch Lightning callback for Sym-JEPA validation projections."""

from __future__ import annotations

import os
import tempfile
from typing import Any, List, Optional, Sequence

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


class PositionalEncodingProbeCallback(pl.Callback):
    """Fit a ridge regression probe from encoder embeddings to index positional encodings."""

    def __init__(
        self,
        *,
        stage: str = "validate",
        max_tokens: int = 8192,
        ridge_lambda: float = 1e-3,
        metric_name: str = "positional_encoding_probe_mse",
        dataloader_idx: int = 0,
        min_tokens: int = 256,
    ) -> None:
        super().__init__()
        valid_stages = {"train", "validate", "test"}
        if stage not in valid_stages:
            raise ValueError(f"stage must be one of {valid_stages}, got '{stage}'")

        self.stage = stage
        self.max_tokens = max_tokens
        self.ridge_lambda = ridge_lambda
        self.metric_name = metric_name
        self.dataloader_idx = dataloader_idx
        self.min_tokens = min_tokens

        self._reset()

    # Lightning hook dispatch -------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.stage == "train":
            self._collect(trainer, pl_module, batch)

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
            self._collect(trainer, pl_module, batch)

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
            self._collect(trainer, pl_module, batch)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.stage == "train":
            self._fit_probe(pl_module, trainer, stage="train")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.stage == "validate":
            self._fit_probe(pl_module, trainer, stage="validate")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if self.stage == "test":
            self._fit_probe(pl_module, trainer, stage="test")

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self._reset()

    # Internal helpers --------------------------------------------------------
    def _collect(self, trainer, pl_module, batch: Any) -> None:
        if trainer.sanity_checking:
            return
        if self._num_tokens >= self.max_tokens:
            return
        if not isinstance(batch, dict):
            return

        context_ids = batch.get("context_ids")
        if context_ids is None:
            return

        context_mask = batch.get("context_mask")

        if not hasattr(pl_module, "encode_context"):
            raise AttributeError(
                "PositionalEncodingProbeCallback expects the LightningModule to "
                "implement 'encode_context'."
            )

        with torch.no_grad():
            context_hidden = pl_module.encode_context(context_ids, context_mask)
            context_hidden, positional = self._pair_features_with_targets(pl_module, context_hidden)

        features_np = context_hidden.detach().cpu().numpy()
        targets_np = positional.detach().cpu().numpy()

        for seq_feat, seq_target in zip(features_np, targets_np):
            if self._num_tokens >= self.max_tokens:
                break
            tokens_to_take = min(seq_feat.shape[0], self.max_tokens - self._num_tokens)

            if tokens_to_take <= 0:
                break

            self._features.append(seq_feat[:tokens_to_take])
            self._targets.append(seq_target[:tokens_to_take])
            self._num_tokens += tokens_to_take

    def _pair_features_with_targets(
        self,
        pl_module: pl.LightningModule,
        context_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = context_hidden.shape
        pe_buffer = pl_module.positional_encoding
        max_available = pe_buffer.size(1)
        if seq_len > max_available:
            seq_len = max_available
            context_hidden = context_hidden[:, :seq_len, :]

        positional = pe_buffer[:, :seq_len, :].to(context_hidden.device).repeat(batch_size, 1, 1)
        return context_hidden, positional

    def _fit_probe(self, pl_module: pl.LightningModule, trainer: pl.Trainer, stage: str) -> None:
        if self._num_tokens < self.min_tokens or not self._features:
            self._reset()
            return

        features = np.concatenate(self._features, axis=0)
        targets = np.concatenate(self._targets, axis=0)

        X = torch.from_numpy(features).double()
        Y = torch.from_numpy(targets).double()

        XtX = X.T @ X
        reg_eye = self.ridge_lambda * torch.eye(XtX.size(0), dtype=XtX.dtype)
        XtX += reg_eye
        XtY = X.T @ Y

        try:
            weights = torch.linalg.solve(XtX, XtY)
        except RuntimeError:
            weights = torch.linalg.lstsq(XtX, XtY).solution

        predictions = X @ weights
        mse = torch.mean((predictions - Y) ** 2).item()

        metric_name = f"{self.metric_name}_{stage}"
        pl_module.log(
            metric_name,
            float(mse),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        pl_module.log(
            f"{self.metric_name}_n_tokens_{stage}",
            float(self._num_tokens),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        self.coefficients_: Optional[np.ndarray] = weights.detach().cpu().numpy()
        self._reset()

    def _reset(self) -> None:
        self._features: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self._num_tokens: int = 0
        self.coefficients_: Optional[np.ndarray] = None


__all__ = [
    "EmbeddingProjectionCallback",
    "PositionalEncodingProbeCallback",
]
