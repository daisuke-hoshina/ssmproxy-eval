"""Lag energy computation from self-similarity matrices."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple


def compute_lag_energy(
    ssm: List[List[float]],
    *,
    min_lag: int = 1,
    top_k: int = 1,
    return_full: bool = False,
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
    max_lag = num_rows - min_lag
    for lag in range(min_lag, max_lag + 1):
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
