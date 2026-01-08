"""Lag energy computation from self-similarity matrices."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple


def compute_lag_energy(
    ssm: List[List[float]],
    *,
    min_lag: int = 4,
    top_k: int = 1,
    return_full: bool = False,
    max_lag: int | None = None,
    min_support: int | None = None,
    best_lag_mode: str = "mean",
    best_lag_lcb_z: float = 1.0,
) -> Tuple[float, Optional[int], Optional[List[Optional[float]]]]:
    """Compute lag energies from a self-similarity matrix.

    For each lag ``k`` such that ``min_lag <= k <= B - min_lag`` (``B`` is the
    number of bars / size of the SSM), the mean value of the diagonal offset by
    ``k`` is computed. The function returns the sum of the top ``k`` energies,
    the lag with the maximum energy, and optionally the full array of lag
    energies indexed by lag value.

    Args:
        ssm: A square self-similarity matrix.
        min_lag: Minimum lag (inclusive) to consider.
        top_k: Number of top lag energies to sum when computing the returned
            energy value.
        return_full: Whether to return the full list of lag energies. Positions
            for lags that are not evaluated are filled with ``None``.
        max_lag: Optional absolute upper bound for the lag (inclusive).
        min_support: Optional minimum number of points (bars) required to
            calculate the lag energy. This effectively limits the maximum lag
            to ``B - min_support``.

    Returns:
        A tuple ``(energy_sum, best_lag, lag_energies)`` where ``energy_sum`` is
        the sum of the ``top_k`` highest lag energies, ``best_lag`` is the lag
        with the maximum energy (or ``None`` if no lags are valid), and
        ``lag_energies`` is the list of lag energies when ``return_full`` is
        ``True`` otherwise ``None``.
    """

    if min_lag < 1:
        raise ValueError("min_lag must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    num_rows = len(ssm)
    if any(len(row) != num_rows for row in ssm):
        raise ValueError("SSM must be square")

    if num_rows == 0:
        return 0.0, None, [] if return_full else None

    # Prepare the container for lag energies aligned by lag index.
    lag_energies: List[Optional[float]] = [None] * num_rows

    valid_energies: List[Tuple[int, float]] = []
    
    # Calculate the effective maximum lag based on constraints.
    # Default upper bound: B - min_lag
    limit_lag = num_rows - min_lag
    
    if max_lag is not None:
        limit_lag = min(limit_lag, max_lag)
        
    if min_support is not None:
        # Prevent lag from exceeding B - min_support
        limit_lag = min(limit_lag, num_rows - min_support)

    best_lag_scores: List[Tuple[int, float]] = []

    for lag in range(min_lag, limit_lag + 1):
        entries = [ssm[i][i + lag] for i in range(num_rows - lag)]
        
        # Calculate stats
        n = len(entries)
        if n == 0:
            continue
            
        mean_val = sum(entries) / n
        lag_energies[lag] = mean_val
        valid_energies.append((lag, mean_val))
        
        # Calculate score for best_lag selection
        score = mean_val
        if best_lag_mode in ("lcb", "zscore"):
            if n > 1:
                variance = sum((x - mean_val) ** 2 for x in entries) / (n - 1)
                std_dev = variance ** 0.5
            else:
                std_dev = 0.0
                
            std_err = std_dev / (n ** 0.5) if n > 0 else 0.0
            
            if best_lag_mode == "lcb":
                score = mean_val - best_lag_lcb_z * std_err
            elif best_lag_mode == "zscore":
                score = mean_val / (std_err + 1e-9)
                
        best_lag_scores.append((lag, score))

    if not valid_energies:
        return 0.0, None, lag_energies if return_full else None

    # Determine the lag with the highest SCORE (for best_lag).
    best_lag, _ = max(best_lag_scores, key=lambda item: item[1])

    # Sum the top_k energies (based on MEAN energy, for consistency).
    energies_sorted = sorted((energy for _, energy in valid_energies), reverse=True)
    energy_sum = sum(energies_sorted[:top_k])

    return energy_sum, best_lag, lag_energies if return_full else None


def _mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count if count else 0.0



def estimate_base_period(
    lag_energies: list[float | None],
    min_lag: int,
    top_k: int = 5
) -> int | None:
    """Estimate base period by returning the smallest lag among the top peaks."""
    if not lag_energies:
        return None
        
    # Find valid (lag, energy) pairs
    valid = []
    N = len(lag_energies)
    for lag in range(min_lag, N):
        val = lag_energies[lag]
        if val is not None:
            # Simple local peak check
            is_peak = True
            if lag > min_lag and (lag_energies[lag-1] is None or lag_energies[lag-1] > val):
                is_peak = False # Not strictly checking neighbors, just using raw values if I could
            # Actually just sorting by energy is robust enough for "top peaks"
            # But let's check basic local maximality if neighbors exist
            prev_val = lag_energies[lag-1] if lag > 0 else None
            next_val = lag_energies[lag+1] if lag < N-1 else None
            
            if prev_val is not None and prev_val > val:
                continue
            if next_val is not None and next_val > val:
                continue
            
            valid.append((lag, val))
            
    if not valid:
        # Fallback to pure argmax if no local peaks found
        best_lag = None
        best_val = -1.0
        for lag in range(min_lag, N):
            val = lag_energies[lag]
            if val is not None and val > best_val:
                best_val = val
                best_lag = lag
        return best_lag
        
    # Sort by energy descending
    valid.sort(key=lambda x: x[1], reverse=True)
    
    # Take top K
    top_candidates = valid[:top_k]
    
    # Return smallest lag
    top_candidates.sort(key=lambda x: x[0])
    return top_candidates[0][0]



def compute_lag_prominence(
    lag_energies: list[float | None],
    *,
    window: int | str = "adaptive",
) -> list[float | None]:
    """Compute local prominence of lag energies.

    Prominence is defined as the deviation from the median of a local window.
    This helps normalize against global trends (like decay) and noise.

    Args:
        lag_energies: list of energy values (or None).
        window: Window size (one-sided) or "adaptive".
            If "adaptive", window = max(2, min(16, lag // 2)).

    Returns:
        List of prominence values of the same size. Invalid entries are None.
    """
    N = len(lag_energies)
    prominence = [None] * N

    # Identify valid indices/values for efficient lookup
    valid_indices = [i for i, v in enumerate(lag_energies) if v is not None]
    if not valid_indices:
        return prominence
    
    # We might need efficient search, but since len(lag_energies) ~ bars ~ 100-1000,
    # linear scan or simple neighbor lookup is fine.
    
    # Pre-compute valid values for fast median if window is very large, 
    # but here window is small.
    
    for idx in valid_indices:
        val = lag_energies[idx]
        assert val is not None
        
        # Determine window size
        if window == "adaptive":
            w = max(2, min(16, idx // 2))
        else:
            w = int(window)
            
        start = max(0, idx - w)
        end = min(N, idx + w + 1)
        
        # Collect neighborhood values (excluding self)
        neighbors = []
        for j in range(start, end):
            if j == idx:
                continue
            v = lag_energies[j]
            if v is not None:
                neighbors.append(v)
        
        if not neighbors:
            # If no neighbors, fallback to global median of valid entries??
            # Or just 0. Let's use 0 conservatively or global median?
            # User spec: "baseline = median(nearby) (if few, use valid median)"
            # Let's say if < 1 neighbor (should happen only if isolated), use valid global median
            import statistics
            all_valid = [x for x in lag_energies if x is not None]
            if len(all_valid) > 1:
                baseline = statistics.median(all_valid)
            else:
                baseline = val # Prominence 0
        else:
             import statistics
             baseline = statistics.median(neighbors)
             
        prominence[idx] = val - baseline

    return prominence


def estimate_base_period_comb(
    lag_energies: list[float | None],
    prominence: list[float | None],
    min_lag: int,
    *,
    harmonics: list[int] | None = None,
    weights: list[float] | None = None,
    tau: float = 32.0,
    min_hits: int = 2,
) -> tuple[int | None, bool]: # returns (L0, is_fallback)
    """Estimate base period using a harmonic comb filter on prominence.

    Args:
        lag_energies: Raw lag energies.
        prominence: Lag prominence values.
        min_lag: Minimum lag to consider.
        harmonics: List of harmonic multipliers (e.g. [1, 2, 4, 8]).
        weights: Weights corresponding to harmonics.

    Returns:
        Tuple (estimated_l0, is_fallback).
    """
    import math

    if harmonics is None:
        harmonics = [1, 2, 4, 8]
    if weights is None:
        weights = [1.0, 0.8, 0.6, 0.5]
    
    # Identify candidates: local peaks in prominence
    # (User spec mentions candidates should be local peaks in prom OR energy)
    # Let's use local peaks in prominence for candidates.
    
    candidates = set()
    N = len(prominence)
    
    # Helper to clean get
    def get_p(i): return prominence[i] if 0 <= i < N and prominence[i] is not None else -float('inf')
    
    # 1. Find local peaks
    peak_indices = []
    for i in range(min_lag, N):
        p_val = get_p(i)
        if p_val == -float('inf'): continue
        
        # Check neighbors
        if p_val >= get_p(i-1) and p_val > get_p(i+1):
            peak_indices.append(i)
            
    # Always include global max energy lag as candidate
    best_lag_energy = -1.0
    best_lag_idx = None
    for i in range(min_lag, N):
        e = lag_energies[i]
        if e is not None and e > best_lag_energy:
            best_lag_energy = e
            best_lag_idx = i
            
    if best_lag_idx is not None:
        candidates.add(best_lag_idx)

    # Add top N peaks from prominence
    peak_indices.sort(key=lambda i: get_p(i), reverse=True)
    for p in peak_indices[:30]:
        candidates.add(p)
        
    if not candidates:
        return best_lag_idx, True  # Fallback to best energy lag

    # Score candidates
    best_score = -float('inf')
    best_l0 = best_lag_idx
    
    # Memoize max index
    max_idx = N - 1

    for cand in candidates:
        score = 0.0
        positive_peaks_hit = 0
        
        for h, w in zip(harmonics, weights):
            target = cand * h
            if target > max_idx:
                continue
            
            val = prominence[target]
            if val is not None:
                # Use max(0, val) as per spec
                contribution = max(0.0, val)
                score += w * contribution
                if contribution > 0:
                    positive_peaks_hit += 1
        
        # Apply penalty for large candidate lags: 1 / (1 + cand/tau)
        score *= (1.0 / (1.0 + cand / tau))
        
        # Skip candidates with too few positive hits
        if positive_peaks_hit < min_hits:
            continue
        
        # Penalize single-hit wonders if possible?
        # User spec: "Prioritize if positive prom >= 2"
        # We can simulate this by boosting score or requiring it.
        # Let's add a small boost for hits > 1 to break ties
        if positive_peaks_hit >= 2:
            score *= 1.1 
        
        if score > best_score:
            best_score = score
            best_l0 = cand
            
    # If score is effectively zero/low, maybe fallback?
    # Spec says "if fails (candidates empty? or no valid score?), fallback"
    # Logic implies if we have candidates we return best Comb L0.
    
    # One edge case: if best_l0 differs from best_lag_idx drastically and has low score,
    # it might be noise. But user didn't specify threshold.
    # We report is_fallback=False if we used Comb logic.
    # Actually, if the score is 0, it means no harmonics aligned with prominence.
    # In that case, we should probably fallback to best_lag.
    
    # If best_score is still -inf (no candidates passed min_hits), fallback.
    if best_score == -float('inf'):
         # Prefer estimate_base_period (smallest of top peaks) over raw best_lag
         fallback_l0 = estimate_base_period(lag_energies, min_lag)
         if fallback_l0 is not None:
             return fallback_l0, True
         # If that fails (no peaks), use best_lag_idx (max energy)
         return best_lag_idx, True

    return best_l0, False


def compute_hierarchy_index_auto_slope(
    prominence: list[float | None],
    l0: int,
    max_levels: int = 5,
    normalize: bool = False,
) -> float | None:
    """Compute hierarchy index using regression slope of prominence vs log2(scale).
    
    Args:
        prominence: List of prominence values.
        l0: Base period.
        max_levels: Maximum multiplier to consider (e.g. 5 -> 1, 2, 4, 8, 16).
        normalize: If True, z-score normalize the y-values (not implemented yet).
        
    Returns:
        Slope (float) or None if insufficient points.
    """
    import math
    import numpy as np
    
    x = []
    y = []
    
    # Powers of 2: 1, 2, 4, ...
    multipliers = [2**i for i in range(max_levels)]
    
    for m in multipliers:
        idx = l0 * m
        if idx < len(prominence) and prominence[idx] is not None:
            x.append(math.log2(m))
            y.append(prominence[idx])
            
    if len(x) < 2:
        # Not enough points for slope
        return None  # Or 0.0? User said "conservatively 0.0 or None". None is safer to indicate calculation failure.
        
    # Fit line
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except Exception:
        return None


def compute_hierarchy_index_auto_adjdiff(
    prominence: list[float | None],
    l0: int,
    max_levels: int = 5,
) -> float | None:
    """Compute hierarchy index using mean difference of adjacent scale prominence."""
    vals = []
    multipliers = [2**i for i in range(max_levels)]
    
    for m in multipliers:
        idx = l0 * m
        if idx < len(prominence) and prominence[idx] is not None:
             vals.append(prominence[idx])
        else:
             # Stop if chain breaks? Or skip?
             # "diffs = [p2-p1, p4-p2]" implies existence.
             # If p1, p2, p8 exist but p4 missing: we can't do p4-p2 or p8-p4.
             # So strictly we need adjacent pairs.
             pass
             
    # Actually, we need to know WHICH multipliers correspond to these vals to ensure adjacency.
    # Let's re-loop properly.
    
    diffs = []
    
    for i in range(len(multipliers) - 1):
        m1 = multipliers[i]
        m2 = multipliers[i+1]
        
        idx1 = l0 * m1
        idx2 = l0 * m2
        
        if idx2 < len(prominence):
            v1 = prominence[idx1]
            v2 = prominence[idx2]
            
            if v1 is not None and v2 is not None:
                diffs.append(v2 - v1)
                
    if not diffs:
        return None
        
    return sum(diffs) / len(diffs)

