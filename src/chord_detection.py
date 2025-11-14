"""Chord estimation utilities for linear probes."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.key_detection import PITCH_CLASS_NAMES

# Simple chord templates (rooted triads).
CHORD_QUALITIES: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
}
QUALITY_LIST: List[str] = list(CHORD_QUALITIES.keys())
NUM_CHORD_CLASSES = len(QUALITY_LIST) * 12


def _build_template(root: int, quality: str) -> np.ndarray:
    template = np.zeros(12, dtype=np.float64)
    for interval in CHORD_QUALITIES[quality]:
        template[(root + interval) % 12] = 1.0
    return template / np.linalg.norm(template)


def estimate_chord_from_histogram(histogram: np.ndarray) -> Dict[str, object]:
    """Estimate the best-fitting chord for a pitch-class histogram."""
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.shape != (12,):
        raise ValueError(f"Expected histogram shape (12,), got {hist.shape}")
    if np.sum(hist) <= 0:
        return {
            "index": -1,
            "root": -1,
            "root_name": "N",
            "quality": "unknown",
            "score": float("nan"),
        }
    hist = hist / np.sum(hist)
    norm = np.linalg.norm(hist)
    if norm > 1e-8:
        hist = hist / norm

    best_score = -np.inf
    best_root = -1
    best_quality = "unknown"

    for q_idx, quality in enumerate(QUALITY_LIST):
        for root in range(12):
            template = _build_template(root, quality)
            score = float(np.dot(hist, template))
            if score > best_score:
                best_score = score
                best_root = root
                best_quality = quality

    class_index = -1
    if best_root >= 0:
        quality_idx = QUALITY_LIST.index(best_quality)
        class_index = quality_idx * 12 + best_root

    return {
        "index": class_index,
        "root": best_root,
        "root_name": PITCH_CLASS_NAMES[best_root] if best_root >= 0 else "N",
        "quality": best_quality,
        "score": best_score,
    }


def assign_chords_to_onsets(
    bars: np.ndarray,
    local_onsets: np.ndarray,
    pitch_classes: np.ndarray,
) -> np.ndarray:
    """Compute chord labels per token based on simultaneous onsets."""
    seq_len = pitch_classes.shape[0]
    labels = np.full(seq_len, -1, dtype=np.int64)

    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx in range(seq_len):
        bar = int(bars[idx])
        onset = int(local_onsets[idx])
        key = (bar, onset)
        groups.setdefault(key, []).append(idx)

    for indices in groups.values():
        histogram = np.zeros(12, dtype=np.float64)
        for idx in indices:
            pc = int(pitch_classes[idx])
            if 0 <= pc < 12:
                histogram[pc] += 1
        chord_info = estimate_chord_from_histogram(histogram)
        label = chord_info["index"]
        for idx in indices:
            labels[idx] = label

    return labels


__all__ = [
    "estimate_chord_from_histogram",
    "assign_chords_to_onsets",
    "NUM_CHORD_CLASSES",
    "QUALITY_LIST",
]
