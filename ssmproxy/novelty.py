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
    raw_novelty: list[float] = [0.0 for _ in range(B)]

    for i in range(B):
        window_sum = 0.0
        for r in range(2 * L + 1):
            for c in range(2 * L + 1):
                window_sum += padded[i + r][i + c] * kernel[r][c]
        raw_novelty[i] = window_sum

    # Define valid range logic
    valid_start = L
    valid_end = B - L
    if valid_end < valid_start:
        valid_start = 0
        valid_end = 0
    valid_len = max(0, valid_end - valid_start)

    normalized_novelty = [0.0 for _ in range(B)]
    if valid_len > 0:
        valid_slice = raw_novelty[valid_start:valid_end]
        min_val = min(valid_slice)
        max_val = max(valid_slice)
        if max_val > min_val:
            for i in range(valid_start, valid_end):
                normalized_novelty[i] = (raw_novelty[i] - min_val) / (max_val - min_val)
        else:
            # If flat within valid range, leave as 0.0
            pass
    # Edges remain 0.0

    peak_dist = min_distance if min_distance is not None else L
    
    # Peak detection only in valid range
    # We slice normalized_novelty, find peaks, then adjust indices
    # Or just run on valid_slice and add valid_start
    # But _find_peaks runs on sequence.
    
    peaks = []
    if valid_len > 0:
        valid_slice_norm = normalized_novelty[valid_start:valid_end]
        # min_distance applies to indices inside the slice
        found_peaks_rel = _find_peaks(valid_slice_norm, min_distance=peak_dist, prominence=prominence)
        peaks = [p + valid_start for p in found_peaks_rel]

    prominences = [_peak_prominence(normalized_novelty, idx) for idx in peaks]
    intervals = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]

    prom_mean = sum(prominences) / len(prominences) if prominences else 0.0
    prom_median = _median(prominences) if prominences else 0.0
    interval_mean = sum(intervals) / len(intervals) if intervals else 0.0
    interval_cv = (_std(intervals, interval_mean) / interval_mean) if intervals and interval_mean else 0.0

    peak_rate = float(len(peaks) / valid_len) if valid_len > 0 else 0.0
    peak_rate_raw = float(len(peaks) / B) if B > 0 else 0.0

    stats = {
        "peak_rate": peak_rate,
        "peak_rate_raw": peak_rate_raw,
        "prom_mean": float(prom_mean),
        "prom_median": float(prom_median),
        "interval_mean": float(interval_mean),
        "interval_cv": float(interval_cv),
        "valid_start": valid_start,
        "valid_end": valid_end,
        "valid_len": valid_len,
        "L": L,
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



def compute_novelty_multiscale(
    ssm: Sequence[Sequence[float]], 
    Ls: Sequence[int], 
    *, 
    prominence: float = 0.10, 
    min_distance: int | None = None,
    mode: str = "integrated",
    consensus_min_scales: int = 2,
    consensus_tolerance: int = 2,
    keep_lmax: bool = True,
    fallback: bool = True,
) -> NoveltyResult:
    """Compute multi-scale novelty by integrating novelty curves from multiple kernel sizes.

    For each kernel size `L` in `Ls`, a novelty curve is computed.
    
    Modes:
    - 'integrated': Curves are max-integrated, then peaks are detected on the result.
    - 'consensus': Peaks are detected on each scale independently (with adaptive min_distance),
      then clustered across scales. Peaks supported by `min_scales` are kept.

    Args:
        ssm: Square self-similarity matrix.
        Ls: Sequence of kernel half-sizes.
        prominence: Peak detection prominence threshold.
        min_distance: Minimum distance between peaks. Defaults to max(Ls).
        mode: 'integrated' or 'consensus'.
        consensus_min_scales: Minimum number of scales required to agree on a peak.
        consensus_tolerance: Tolerance (in bars) for matching peaks across scales.
        keep_lmax: If True, always keep peaks found at the largest scale L_max.
        fallback: If True, fallback to 'integrated' peaks if consensus yields nothing.

    Returns:
        A ``NoveltyResult`` for the integrated novelty curve (and consensus peaks if selected).
    """
    if any(L <= 0 for L in Ls):
         raise ValueError("All kernel sizes must be positive")
    
    size = len(ssm)
    B = size
    if any(len(row) != size for row in ssm):
        raise ValueError("SSM must be a square 2D matrix")

    # Compute individual results
    individual_results = []
    curves = []
    for L in Ls:
        # In consensus mode, apply scale-dependent min_distance to suppress noise in small scales
        dist_arg = min_distance
        if mode == "consensus":
            base_md = min_distance if min_distance is not None else L
            # Ensure we don't pick too many peaks in small scales
            dist_arg = max(base_md, L // 2)

        res = compute_novelty(ssm, L=L, prominence=prominence, min_distance=dist_arg)
        individual_results.append(res)
        curves.append(res.novelty)

    # Combine: max over L (used for return value and integrated mode)
    integrated_novelty = [0.0] * B
    if curves:
        for i in range(B):
             integrated_novelty[i] = max(c[i] for c in curves)

    # Enforce common valid range
    max_L = max(Ls) if Ls else 0
    valid_start = max_L
    valid_end = B - max_L
    if valid_end < valid_start:
        valid_start = 0
        valid_end = 0
    valid_len = max(0, valid_end - valid_start)

    for i in range(B):
        if i < valid_start or i >= valid_end:
            integrated_novelty[i] = 0.0

    peaks = []
    peak_dist = min_distance if min_distance is not None else max_L

    if mode == "consensus":
        # Collect all peaks with their source scale
        # candidates: list of (position, L_val)
        candidates = []
        for res, L_val in zip(individual_results, Ls):
            for p in res.peaks:
                # Filter out peaks outside expected valid range?
                # compute_novelty returns peaks valid for ITS L.
                # But we only care about common valid range for the final result.
                if valid_start <= p < valid_end:
                    candidates.append((p, L_val))
        
        candidates.sort(key=lambda x: x[0])
        
        # Single-linkage clustering with tolerance
        clusters = []
        if candidates:
            current_cluster = [candidates[0]]
            for i in range(1, len(candidates)):
                curr_p = candidates[i]
                prev_p = current_cluster[-1]
                if (curr_p[0] - prev_p[0]) <= consensus_tolerance:
                    current_cluster.append(curr_p)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [curr_p]
            clusters.append(current_cluster)
            
        # Filter clusters
        final_peaks = []
        for cluster in clusters:
            unique_scales = set(c[1] for c in cluster)
            has_lmax = any(c[1] == max_L for c in cluster)
            
            keep = False
            if len(unique_scales) >= consensus_min_scales:
                keep = True
            elif keep_lmax and has_lmax:
                keep = True
                
            if keep:
                # Select representative peak: the one from the largest L in the cluster
                # If tie, pick median of those? Arbitrary pick is fine.
                # Sort cluster by L desc
                cluster.sort(key=lambda x: x[1], reverse=True)
                best_p = cluster[0][0]
                final_peaks.append(best_p)
                
        # Deduplicate final peaks (just in case clusters overlapped weirdly, though single linkage shouldn't)
        # But sorting by position is good.
        peaks = sorted(list(set(final_peaks)))
        
        # Fallback
        if not peaks and fallback:
             if valid_len > 0:
                valid_slice_norm = integrated_novelty[valid_start:valid_end]
                found_peaks_rel = _find_peaks(valid_slice_norm, min_distance=peak_dist, prominence=prominence)
                peaks = [p + valid_start for p in found_peaks_rel]

    else:
        # Integrated Mode (Legacy)
        if valid_len > 0:
            valid_slice_norm = integrated_novelty[valid_start:valid_end]
            found_peaks_rel = _find_peaks(valid_slice_norm, min_distance=peak_dist, prominence=prominence)
            peaks = [p + valid_start for p in found_peaks_rel]

    prominences = [_peak_prominence(integrated_novelty, idx) for idx in peaks]
    intervals = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]

    prom_mean = sum(prominences) / len(prominences) if prominences else 0.0
    prom_median = _median(prominences) if prominences else 0.0
    interval_mean = sum(intervals) / len(intervals) if intervals else 0.0
    interval_cv = (_std(intervals, interval_mean) / interval_mean) if intervals and interval_mean else 0.0

    peak_rate = float(len(peaks) / valid_len) if valid_len > 0 else 0.0
    peak_rate_raw = float(len(peaks) / B) if B > 0 else 0.0

    stats = {
        "peak_rate": peak_rate,
        "peak_rate_raw": peak_rate_raw,
        "prom_mean": float(prom_mean),
        "prom_median": float(prom_median),
        "interval_mean": float(interval_mean),
        "interval_cv": float(interval_cv),
        "valid_start": valid_start,
        "valid_end": valid_end,
        "valid_len": valid_len,
        "L": max_L, # Representative L
    }
    
    return NoveltyResult(novelty=integrated_novelty, peaks=peaks, stats=stats)


def _std(values: Sequence[float], mean_val: float) -> float:
    if not values:
        return 0.0
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    return variance ** 0.5
