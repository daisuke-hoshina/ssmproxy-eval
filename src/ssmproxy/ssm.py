"""Self-similarity matrix computation utilities."""
from __future__ import annotations

import math
from typing import List, Sequence


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity handling zero vectors safely."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be the same length")

    norm_a = math.sqrt(sum(value * value for value in vec_a))
    norm_b = math.sqrt(sum(value * value for value in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)


def compute_ssm(
    pch: List[List[float]],
    onh: List[List[float]],
    *,
    weight_pch: float = 0.5,
    weight_onh: float = 0.5,
    map_to_unit_interval: bool = False,
) -> List[List[float]]:
    """Compute a bar-by-bar self-similarity matrix.

    Args:
        pch: Pitch class histograms per bar.
        onh: Onset histograms per bar.
        weight_pch: Weight for the pitch class similarity component.
        weight_onh: Weight for the onset histogram similarity component.
        map_to_unit_interval: If true, map cosine range [-1, 1] to [0, 1]
            via ``(S + 1) / 2``.

    Returns:
        A square similarity matrix where ``S[i][j]`` represents the similarity
        between bar ``i`` and bar ``j``.
    """
    if len(pch) != len(onh):
        raise ValueError("PCH and ONH must have the same number of bars")

    num_bars = len(pch)
    if num_bars == 0:
        return []

    weight_sum = weight_pch + weight_onh
    if weight_sum == 0:
        raise ValueError("At least one weight must be non-zero")

    normalized_weight_pch = weight_pch / weight_sum
    normalized_weight_onh = weight_onh / weight_sum

    matrix: List[List[float]] = [[0.0 for _ in range(num_bars)] for _ in range(num_bars)]

    for i in range(num_bars):
        # Diagonal elements can be filled directly to avoid redundant computation.
        matrix[i][i] = 1.0 if (sum(pch[i]) > 0 or sum(onh[i]) > 0) else 0.0
        for j in range(i + 1, num_bars):
            sim_pch = _cosine_similarity(pch[i], pch[j])
            sim_onh = _cosine_similarity(onh[i], onh[j])
            similarity = normalized_weight_pch * sim_pch + normalized_weight_onh * sim_onh
            if map_to_unit_interval:
                similarity = (similarity + 1.0) / 2.0

            matrix[i][j] = similarity
            matrix[j][i] = similarity

    return matrix
