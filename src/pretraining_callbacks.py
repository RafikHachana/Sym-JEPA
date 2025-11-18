"""Lightweight PyTorch Lightning callback for Sym-JEPA validation projections."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch

from src.model import Utils
from src.octuple_tokenizer import max_inst, max_pitch, pos_resolution, ts_list
from src.key_detection import build_pitch_class_histogram, estimate_key_from_pitch_classes
from src.chord_detection import assign_chords_to_onsets, NUM_CHORD_CLASSES
# from src.viz import visualize_tsne_clusters, visualize_umap_clusters


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
    "TokenAttributeProbeCallback",
]

class TokenAttributeProbeCallback(pl.Callback):
    """Fit linear probes that recover token-level attributes from encoder states."""

    _SEQUENCE_TARGET_FNS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "key": lambda decoded: TokenAttributeProbeCallback._estimate_key_labels(decoded),
        "chord": lambda decoded: TokenAttributeProbeCallback._estimate_chord_labels(decoded),
    }

    _ATTRIBUTE_CONFIG: Dict[str, Dict[str, Any]] = {
        "pitch": {
            "getter": Utils.get_pitch_sequence,
            "type": "regression",
            "metric": "pitch_probe_mse",
            "valid_fn": lambda arr: arr >= 0,
            "scope": "token",
        },
        "duration": {
            "getter": Utils.get_duration_sequence,
            "type": "regression",
            "metric": "duration_probe_mse",
            "valid_fn": None,
            "scope": "token",
        },
        "instrument": {
            "getter": Utils.get_instrument_sequence,
            "type": "classification",
            "metric": "instrument_probe_acc",
            "num_classes": max_inst + 1,
            "valid_fn": lambda arr: arr >= 0,
            "scope": "token",
        },
        "pitch_class": {
            "getter": Utils.get_pitch_class_sequence,
            "type": "classification",
            "metric": "pitch_class_probe_acc",
            "num_classes": 12,
            "valid_fn": lambda arr: arr >= 0,
            "scope": "token",
        },
        "beat_strength": {
            "getter": Utils.get_strong_beat_sequence,
            "type": "classification",
            "metric": "beat_strength_probe_acc",
            "num_classes": 2,
            "valid_fn": None,
            "scope": "token",
        },
        "key": {
            "type": "classification",
            "metric": "key_probe_acc",
            "num_classes": 24,
            "valid_fn": lambda arr: arr >= 0,
            "scope": "sequence",
            "sequence_target_fn": "key",
        },
        "chord": {
            "type": "classification",
            "metric": "chord_probe_acc",
            "num_classes": NUM_CHORD_CLASSES,
            "valid_fn": lambda arr: arr >= 0,
            "scope": "sequence",
            "sequence_target_fn": "chord",
        },
    }

    def __init__(
        self,
        *,
        attribute: str,
        stage: str = "validate",
        max_tokens: int = 8192,
        ridge_lambda: float = 1e-3,
        metric_prefix: Optional[str] = None,
        dataloader_idx: int = 0,
        min_tokens: int = 256,
        max_debug_examples: int = 5,
    ) -> None:
        if attribute not in self._ATTRIBUTE_CONFIG:
            raise ValueError(
                f"Unsupported attribute '{attribute}'. Expected one of {list(self._ATTRIBUTE_CONFIG)}."
            )
        super().__init__()

        valid_stages = {"train", "validate", "test"}
        if stage not in valid_stages:
            raise ValueError(f"stage must be one of {valid_stages}, got '{stage}'")

        self.attribute = attribute
        self.stage = stage
        self.max_tokens = max_tokens
        self.ridge_lambda = ridge_lambda
        attr_cfg = self._ATTRIBUTE_CONFIG[attribute]
        default_metric = attr_cfg["metric"]
        self.metric_prefix = metric_prefix or default_metric
        self.dataloader_idx = dataloader_idx
        self.min_tokens = min_tokens
        self.max_debug_examples = max_debug_examples

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
            self._fit_probe(pl_module, stage="train")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.stage == "validate":
            self._fit_probe(pl_module, stage="validate")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if self.stage == "test":
            self._fit_probe(pl_module, stage="test")

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

        attr_cfg = self._ATTRIBUTE_CONFIG[self.attribute]
        context_ids = batch.get("context_ids")
        if context_ids is None:
            return

        context_mask = batch.get("context_mask")

        if not hasattr(pl_module, "encode_context"):
            raise AttributeError(
                "TokenAttributeProbeCallback expects the LightningModule to implement 'encode_context'."
            )

        scope = attr_cfg.get("scope", "token")

        with torch.no_grad():
            context_hidden = pl_module.encode_context(context_ids, context_mask)
            decoded_tokens = Utils.decode_tokens(context_ids, pl_module.vocab).to(context_hidden.device)
            if scope == "token":
                attribute_tensor = attr_cfg["getter"](decoded_tokens).float()
                targets_np = attribute_tensor.detach().cpu().numpy()
            else:
                sequence_labels = self._compute_sequence_attribute_labels(decoded_tokens, attr_cfg)
                targets_np = sequence_labels.detach().cpu().numpy()

        features_np = context_hidden.detach().cpu().numpy()
        valid_fn = attr_cfg.get("valid_fn")

        for seq_idx, seq_feat in enumerate(features_np):
            if self._num_tokens >= self.max_tokens:
                break

            if scope == "token":
                seq_target = targets_np[seq_idx]
                seq_target_flat = seq_target.reshape(-1)
                mask = np.ones_like(seq_target_flat, dtype=bool)
                if valid_fn is not None:
                    mask &= valid_fn(seq_target_flat)

                if not mask.any():
                    continue
                seq_feat_filtered = seq_feat[mask]
                seq_target_filtered = seq_target_flat[mask]
            else:
                seq_label = targets_np[seq_idx]
                if valid_fn is not None:
                    valid_mask = valid_fn(np.asarray([seq_label]))
                    if not bool(np.all(valid_mask)):
                        continue
                seq_feat_filtered = seq_feat
                seq_target_filtered = np.full(seq_feat.shape[0], seq_label, dtype=np.int64)

            tokens_to_take = min(seq_feat_filtered.shape[0], self.max_tokens - self._num_tokens)
            if tokens_to_take <= 0:
                break

        slice_feats = seq_feat_filtered[:tokens_to_take]
        self._features.append(slice_feats)
        if attr_cfg["type"] == "classification":
            target_slice = seq_target_filtered[:tokens_to_take].astype(np.int64).reshape(-1, 1)
        else:
            target_slice = seq_target_filtered[:tokens_to_take].astype(np.float32).reshape(-1, 1)
        self._targets.append(target_slice)
        self._num_tokens += tokens_to_take
        self._store_debug_example(decoded_tokens[seq_idx], targets_np[seq_idx], slice_feats, target_slice, scope)

    def _compute_sequence_attribute_labels(self, decoded_tokens: torch.Tensor, attr_cfg: Dict[str, Any]) -> torch.Tensor:
        fn_name = attr_cfg.get("sequence_target_fn")
        if fn_name is None:
            raise ValueError(f"No sequence_target_fn specified for attribute '{self.attribute}'.")
        fn = self._SEQUENCE_TARGET_FNS.get(fn_name)
        if fn is None:
            raise ValueError(f"Unknown sequence_target_fn '{fn_name}' for attribute '{self.attribute}'.")
        return fn(decoded_tokens)

    def _fit_probe(self, pl_module: pl.LightningModule, stage: str) -> None:
        if self._num_tokens < self.min_tokens or not self._features:
            self._reset()
            return

        attr_cfg = self._ATTRIBUTE_CONFIG[self.attribute]
        features = np.concatenate(self._features, axis=0)
        targets = np.concatenate(self._targets, axis=0)

        X = torch.from_numpy(features).double()
        if attr_cfg["type"] == "classification":
            class_indices = targets.astype(np.int64).reshape(-1)
            num_classes = attr_cfg["num_classes"]
            Y = torch.zeros(class_indices.shape[0], num_classes, dtype=torch.double)
            class_index_tensor = torch.from_numpy(class_indices).long().unsqueeze(1)
            Y.scatter_(1, class_index_tensor, 1.0)
        else:
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
        if attr_cfg["type"] == "classification":
            predicted_classes = torch.argmax(predictions, dim=1)
            accuracy = (predicted_classes == class_index_tensor.squeeze(1)).float().mean().item()
            metric_value = accuracy
        else:
            mse = torch.mean((predictions - Y) ** 2).item()
            metric_value = mse

        pl_module.log(
            f"{self.metric_prefix}_{stage}",
            float(metric_value),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        # pl_module.log(
        #     f"{self.metric_prefix}_n_tokens_{stage}",
        #     float(self._num_tokens),
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        #     sync_dist=True,
        # )

        self.coefficients_: Optional[np.ndarray] = weights.detach().cpu().numpy()
        self._reset()

    def _reset(self) -> None:
        self._features: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self._num_tokens: int = 0
        self.coefficients_: Optional[np.ndarray] = None

    @staticmethod
    def _estimate_key_labels(decoded_tokens: torch.Tensor) -> torch.Tensor:
        pitch_classes = Utils.get_pitch_class_sequence(decoded_tokens).detach().cpu().numpy()
        labels = []
        for seq_pitch_classes in pitch_classes:
            histogram = build_pitch_class_histogram(seq_pitch_classes)
            key_info = estimate_key_from_pitch_classes(histogram)
            labels.append(key_info["index"])

        return torch.tensor(labels, device=decoded_tokens.device, dtype=torch.long)

    @staticmethod
    def _estimate_chord_labels(decoded_tokens: torch.Tensor) -> torch.Tensor:
        bars = decoded_tokens[:, :, 0].detach().cpu().numpy()
        local_onsets = decoded_tokens[:, :, 1].detach().cpu().numpy()
        pitch_classes = Utils.get_pitch_class_sequence(decoded_tokens).detach().cpu().numpy()

        labels = []
        for seq_idx in range(decoded_tokens.size(0)):
            seq_labels = assign_chords_to_onsets(
                bars[seq_idx],
                local_onsets[seq_idx],
                pitch_classes[seq_idx],
            )
            labels.append(seq_labels)

        return torch.tensor(labels, device=decoded_tokens.device, dtype=torch.long)

        self.debug_examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def _store_debug_example(
        self,
        decoded_sequence: torch.Tensor,
        full_targets: np.ndarray,
        features: np.ndarray,
        targets_slice: np.ndarray,
        scope: str,
    ) -> None:
        if len(self.debug_examples) >= self.max_debug_examples:
            return
        decoded_np = decoded_sequence.detach().cpu().numpy()
        if scope == "token":
            example = (
                decoded_np[: features.shape[0]],
                features.copy(),
                targets_slice.squeeze(-1).copy(),
            )
        else:
            example = (
                decoded_np,
                features.copy(),
                targets_slice.squeeze(-1).copy(),
            )
        self.debug_examples.append(example)
