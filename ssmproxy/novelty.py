"""Novelty curve computation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class NoveltyResult:
    """Container for novelty computation outputs."""

    novelty: list[float]
    peaks: list[int]
    stats: Dict[str, float]


def _build_checkerboard_kernel(L: int) -> list[list[float]]:
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
    kernel = [[1.0 for _ in range(size)] for _ in range(size)]

    for r in range(L):
        for c in range(L + 1, size):
            kernel[r][c] = -1.0
    for r in range(L + 1, size):
        for c in range(L):
            kernel[r][c] = -1.0

    for i in range(size):
        kernel[L][i] = 0.0
        kernel[i][L] = 0.0

    return kernel


def compute_novelty(
    ssm: Sequence[Sequence[float]], L: int, *, prominence: float = 0.10, min_distance: int | None = None
) -> NoveltyResult:
    """Compute a novelty curve and summary statistics from a self-similarity matrix.

    The novelty is computed by sliding a Foote checkerboard kernel of half-size
    ``L`` along the diagonal of the self-similarity matrix (SSM). The resulting
    curve is min-max normalized to ``[0, 1]``. Peaks are detected with
    ``scipy.signal.find_peaks`` using a prominence of ``0.10`` and minimum peak
    distance ``L``.

    Args:
        ssm: Square self-similarity matrix.
        ssm: Square self-similarity matrix.
        L: Half-size of the Foote kernel.
        prominence: Peak detection prominence threshold.
        min_distance: Minimum distance between peaks. If None, defaults to L.

    Returns:
        A ``NoveltyResult`` holding the normalized novelty curve, detected peak
        indices, and summary statistics.
    """

    size = len(ssm)
    if any(len(row) != size for row in ssm):
        raise ValueError("SSM must be a square 2D matrix")
    B = size
    kernel = _build_checkerboard_kernel(L)

    padded = _pad_matrix(ssm, L)
    novelty: list[float] = [0.0 for _ in range(B)]

    for i in range(B):
        window_sum = 0.0
        for r in range(2 * L + 1):
            for c in range(2 * L + 1):
                window_sum += padded[i + r][i + c] * kernel[r][c]
        novelty[i] = window_sum

    if not novelty:
        normalized_novelty: list[float] = []
    else:
        min_val = min(novelty)
        max_val = max(novelty)
        if max_val > min_val:
            normalized_novelty = [(value - min_val) / (max_val - min_val) for value in novelty]
        else:
            normalized_novelty = [0.0 for _ in novelty]

    peak_dist = min_distance if min_distance is not None else L
    peaks = _find_peaks(normalized_novelty, min_distance=peak_dist, prominence=prominence)
    prominences = [_peak_prominence(normalized_novelty, idx) for idx in peaks]
    intervals = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]

    prom_mean = sum(prominences) / len(prominences) if prominences else 0.0
    prom_median = _median(prominences) if prominences else 0.0
    interval_mean = sum(intervals) / len(intervals) if intervals else 0.0
    interval_cv = (_std(intervals, interval_mean) / interval_mean) if intervals and interval_mean else 0.0

    stats = {
        "peak_rate": float(len(peaks) / B) if B else 0.0,
        "prom_mean": float(prom_mean),
        "prom_median": float(prom_median),
        "interval_mean": float(interval_mean),
        "interval_cv": float(interval_cv),
    }

    return NoveltyResult(novelty=normalized_novelty, peaks=peaks, stats=stats)


def _pad_matrix(matrix: Sequence[Sequence[float]], L: int) -> list[list[float]]:
    size = len(matrix)
    padded_size = size + 2 * L
    padded = [[0.0 for _ in range(padded_size)] for _ in range(padded_size)]
    for r in range(size):
        for c in range(size):
            padded[r + L][c + L] = float(matrix[r][c])
    return padded


def _find_peaks(values: Sequence[float], min_distance: int, prominence: float) -> list[int]:
    peaks: list[int] = []
    last_peak = -min_distance - 1
    for i in range(1, len(values) - 1):
        if i - last_peak < min_distance:
            continue
        left = values[i - 1]
        right = values[i + 1]
        current = values[i]
        if current > left and current >= right and _peak_prominence(values, i) >= prominence:
            peaks.append(i)
            last_peak = i
    return peaks


def _peak_prominence(values: Sequence[float], index: int) -> float:
    current = values[index]
    left_min = min(values[: index + 1]) if index > 0 else current
    right_min = min(values[index:]) if index < len(values) - 1 else current
    baseline = max(left_min, right_min)
    return current - baseline


def _median(values: Sequence[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


def _std(values: Sequence[float], mean_val: float) -> float:
    if not values:
        return 0.0
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    return variance ** 0.5
