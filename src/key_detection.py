"""Utilities for estimating global musical key from pitch-class statistics.

The implementation follows the Krumhansl-Schmuckler key-finding algorithm by
correlating observed pitch-class distributions with empirically derived major
and minor key profiles. The best-scoring profile is returned together with the
corresponding tonic/mode identifiers.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

# Krumhansl & Kessler (1982) key profiles for major and minor modes.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#",
                     "G", "G#", "A", "A#", "B")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector
    return vector / norm


def estimate_key_from_pitch_classes(
    pitch_histogram: np.ndarray,
) -> Dict[str, object]:
    """Estimate the most likely key for a pitch-class histogram.

    Parameters
    ----------
    pitch_histogram:
        Length-12 array with counts/weights for each pitch class (C..B). The
        histogram is typically built by tallying the pitch classes present in
        the sequence (drums filtered out).

    Returns
    -------
    dict
        Dictionary containing:
        - ``index``: integer key id in [0, 23], where ``key % 12`` is the tonic
          and ``key // 12`` is 0 for major, 1 for minor.
        - ``tonic``: pitch-class index.
        - ``tonic_name``: human-readable tonic string.
        - ``mode``: ``"major"`` or ``"minor"``.
        - ``score``: cosine similarity score of the best-matching profile.
    """
    histogram = np.asarray(pitch_histogram, dtype=np.float64)
    if histogram.shape != (12,):
        raise ValueError(f"Expected pitch_histogram with shape (12,), got {histogram.shape}")
    if np.sum(histogram) <= 0:
        return {
            "index": -1,
            "tonic": -1,
            "tonic_name": "N",
            "mode": "unknown",
            "score": float("nan"),
        }

    histogram = histogram.copy()
    histogram = histogram / np.sum(histogram)
    histogram = _normalize(histogram)

    best_score = -math.inf
    best_tonic = -1
    best_mode = "unknown"

    for tonic in range(12):
        for mode_name, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            rotated_profile = np.roll(profile, -tonic)
            rotated_profile = _normalize(rotated_profile)
            score = float(np.dot(histogram, rotated_profile))

            if score > best_score:
                best_score = score
                best_tonic = tonic
                best_mode = mode_name

    mode_index = 0 if best_mode == "major" else 1
    key_index = best_tonic if best_tonic >= 0 else -1
    if key_index >= 0:
        key_index += mode_index * 12

    return {
        "index": key_index,
        "tonic": best_tonic,
        "tonic_name": PITCH_CLASS_NAMES[best_tonic] if best_tonic >= 0 else "N",
        "mode": best_mode,
        "score": best_score,
    }


def build_pitch_class_histogram(pitch_classes: np.ndarray) -> np.ndarray:
    """Aggregate a histogram over valid pitch classes.

    Parameters
    ----------
    pitch_classes:
        Array of integer pitch-class labels in ``[0, 11]``. Invalid entries
        (negative values) are ignored.

    Returns
    -------
    np.ndarray
        Length-12 histogram of pitch-class counts.
    """
    flat = np.asarray(pitch_classes, dtype=np.int64).ravel()
    valid = flat[(flat >= 0) & (flat < 12)]
    hist = np.zeros(12, dtype=np.float64)
    if valid.size > 0:
        counts = np.bincount(valid, minlength=12)
        hist[: len(counts)] = counts
    return hist


__all__ = [
    "estimate_key_from_pitch_classes",
    "build_pitch_class_histogram",
    "PITCH_CLASS_NAMES",
]
