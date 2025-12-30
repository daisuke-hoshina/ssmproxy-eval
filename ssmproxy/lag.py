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
    lag_energies: List[Optional[float]] = [None for _ in range(num_rows)]

    valid_energies: List[Tuple[int, float]] = []
    
    # Calculate the effective maximum lag based on constraints.
    # Default upper bound: B - min_lag
    limit_lag = num_rows - min_lag
    
    if max_lag is not None:
        limit_lag = min(limit_lag, max_lag)
        
    if min_support is not None:
        # Prevent lag from exceeding B - min_support
        limit_lag = min(limit_lag, num_rows - min_support)

    for lag in range(min_lag, limit_lag + 1):
        entries = (ssm[i][i + lag] for i in range(num_rows - lag))
        energy = _mean(entries)
        lag_energies[lag] = energy
        valid_energies.append((lag, energy))

    if not valid_energies:
        return 0.0, None, lag_energies if return_full else None

    # Determine the lag with the highest energy.
    best_lag, best_energy = max(valid_energies, key=lambda item: item[1])

    # Sum the top_k energies (or fewer if fewer lags are valid).
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
    lag_energies: list[float | None], min_lag: int
) -> int | None:
    """Estimate the base period L0 from lag energies.

    Finds local peaks in the valid lag range and selects the smallest lag
    among the top 5 highest energy peaks. Falls back to the global maximum
    if no peaks are found.

    Args:
        lag_energies: List of lag energies where index corresponds to lag.
            Invalid lags should be None.
        min_lag: Minimum lag considered valid.

    Returns:
        The estimated base period L0, or None if no valid energies exist.
    """
    valid_lags = [
        (lag, energy)
        for lag, energy in enumerate(lag_energies)
        if energy is not None and lag >= min_lag
    ]

    if not valid_lags:
        return None

    peaks: list[tuple[int, float]] = []
    max_idx = len(lag_energies) - 1

    # Find local peaks
    # We iterate through valid lags. Note that valid lags might be contiguous
    # range [min_lag, B - min_lag].
    # Accessing lag_energies directly is safer for neighbors.

    for lag, energy in valid_lags:
        # Check Neighbors
        # Left neighbor
        is_peak = True
        
        # Check left
        if lag - 1 >= min_lag and lag_energies[lag - 1] is not None:
             if energy < lag_energies[lag - 1]: # type: ignore
                 is_peak = False
        
        # Check right
        if is_peak and lag + 1 <= max_idx and lag_energies[lag + 1] is not None:
             # Strict inequality for right side to handle plateaus (opt to pick first)
             # or simply strict > for pure peaks. Spec says:
             # "E[lag] >= E[lag-1] and E[lag] > E[lag+1]"
             if energy <= lag_energies[lag + 1]: # type: ignore
                 is_peak = False
        
        # Boundary conditions are implicitly handled:
        # If lag=min_lag, left check is skipped (or check against None/min_lag-1)
        # Spec: "lag=min_lag is peak if E[min_lag] > E[min_lag+1]"
        # If lag=max, right check is skipped.
        
        if is_peak:
            peaks.append((lag, energy))

    if not peaks:
        # Fallback to global max
        best_lag, _ = max(valid_lags, key=lambda x: x[1])
        return best_lag

    # Sort peaks by energy descending
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Take top 5
    top_candidates = peaks[:5]

    # Select smallest lag among top candidates
    l0, _ = min(top_candidates, key=lambda x: x[0])
    
    return l0
