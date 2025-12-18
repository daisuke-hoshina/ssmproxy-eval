"""Novelty curve computation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy.signal import find_peaks


@dataclass
class NoveltyResult:
    """Container for novelty computation outputs."""

    novelty: np.ndarray
    peaks: np.ndarray
    stats: Dict[str, float]


def _build_checkerboard_kernel(L: int) -> np.ndarray:
    """Create a Foote-style checkerboard kernel.

    The kernel emphasizes contrast across the diagonal by assigning positive
    weights to matching quadrants and negative weights to opposing quadrants.

    Args:
        L: Half-size of the kernel. The resulting kernel has shape
            ``(2 * L + 1, 2 * L + 1)``.
    """

    if L <= 0:
        raise ValueError("Kernel half-size L must be positive")

    size = 2 * L + 1
    kernel = np.ones((size, size), dtype=float)

    kernel[:L, L:] = -1.0
    kernel[L + 1 :, :L] = -1.0

    kernel[L, :] = 0.0
    kernel[:, L] = 0.0

    return kernel


def compute_novelty(ssm: Sequence[Sequence[float]], L: int) -> NoveltyResult:
    """Compute a novelty curve and summary statistics from a self-similarity matrix.

    The novelty is computed by sliding a Foote checkerboard kernel of half-size
    ``L`` along the diagonal of the self-similarity matrix (SSM). The resulting
    curve is min-max normalized to ``[0, 1]``. Peaks are detected with
    ``scipy.signal.find_peaks`` using a prominence of ``0.10`` and minimum peak
    distance ``L``.

    Args:
        ssm: Square self-similarity matrix.
        L: Half-size of the Foote kernel.

    Returns:
        A ``NoveltyResult`` holding the normalized novelty curve, detected peak
        indices, and summary statistics.
    """

    matrix = np.asarray(ssm, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("SSM must be a square 2D matrix")

    B = matrix.shape[0]
    kernel = _build_checkerboard_kernel(L)

    padded = np.pad(matrix, pad_width=L, mode="constant")
    novelty = np.zeros(B, dtype=float)

    for i in range(B):
        window = padded[i : i + 2 * L + 1, i : i + 2 * L + 1]
        novelty[i] = float(np.sum(window * kernel))

    if novelty.size == 0:
        normalized_novelty = novelty
    else:
        min_val = float(novelty.min())
        max_val = float(novelty.max())
        if max_val > min_val:
            normalized_novelty = (novelty - min_val) / (max_val - min_val)
        else:
            normalized_novelty = np.zeros_like(novelty)

    peaks, properties = find_peaks(normalized_novelty, prominence=0.10, distance=L)
    prominences = properties.get("prominences", np.array([]))
    intervals = np.diff(peaks)

    prom_mean = float(np.mean(prominences)) if prominences.size else 0.0
    prom_median = float(np.median(prominences)) if prominences.size else 0.0
    interval_mean = float(np.mean(intervals)) if intervals.size else 0.0
    interval_cv = (
        float(np.std(intervals) / interval_mean) if intervals.size and interval_mean else 0.0
    )

    stats = {
        "peak_rate": float(len(peaks) / B) if B else 0.0,
        "prom_mean": prom_mean,
        "prom_median": prom_median,
        "interval_mean": interval_mean,
        "interval_cv": interval_cv,
    }

    return NoveltyResult(novelty=normalized_novelty, peaks=peaks, stats=stats)
