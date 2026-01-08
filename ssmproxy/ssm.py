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
    
    if not (math.isfinite(norm_a) and math.isfinite(norm_b)):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    
    if not math.isfinite(dot):
        return 0.0
        
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


try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _compute_ssm_multi_numpy(
    features: dict[str, Sequence[Sequence[float]]],
    weights: dict[str, float],
    active_keys: list[str],
    norm_weights: dict[str, float],
    num_bars: int,
    map_to_unit_interval: bool,
) -> List[List[float]]:
    """NumPy-accelerated implementation of SSM computation."""
    
    # Initialize accumulator
    S_total = np.zeros((num_bars, num_bars), dtype=np.float32)
    bar_is_active = np.zeros(num_bars, dtype=bool)
    
    for key in active_keys:
        # Shape: (num_bars, feature_dim)
        # Convert to float32 for speed/memory tradeoff
        X = np.array(features[key], dtype=np.float32)
        if len(X) != num_bars:
             # Should be caught by validation before, but safety check
             raise ValueError(f"Feature {key} shape mismatch in numpy path")

        # Sanitize input: replace NaN/Inf with 0.0
        # This prevents initial garbage from propagating
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Track activity per bar (norm > epsilon)
        # Compute norms in float64 for stability
        norms_sq = np.sum(X.astype(np.float64)**2, axis=1)
        
        # Determine active rows
        # A row is active if its L2 norm is sufficiently non-zero
        current_active = (norms_sq > 1e-9)
        bar_is_active |= current_active
        
        # Normalize rows safely
        # 1. Prepare output array (init with 0.0)
        X_norm = np.zeros_like(X)
        
        # 2. Compute norms
        norms = np.sqrt(norms_sq)
        
        # 3. Divide safely: only where active. Implicitly 0.0 where inactive.
        np.divide(X, norms[:, np.newaxis], out=X_norm, where=current_active[:, np.newaxis])
        
        # 4. Final sanitizer on X_norm (paranoia for 0/0 leaks or other weirdness)
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute cosine similarity: X @ X.T
        # We perform multiplication in a permissive context because some BLAS implementations
        # (e.g. Accelerate on ARM64) can set zero-divide or invalid flags spuriously even 
        # on valid inputs. We rely on strict input sanitization and the final finiteness 
        # check to catch actual errors.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
             S_k = X_norm @ X_norm.T
             
             # Accumulate
             # norm_weights[key] is float. S_k is float32.
             S_total += (norm_weights[key] * S_k)
        
    if map_to_unit_interval:
        # map [-1, 1] -> [0, 1]
        S_total = (S_total + 1.0) / 2.0
    
    # Enforce Diagonal Logic (Task D + A)
    # 1.0 if bar is active, 0.0 else. UNMAPPED.
    # The map_to_unit_interval above affected diagonals too. We must overwrite them.
    
    # Create diagonal values
    diag_vals = np.where(bar_is_active, 1.0, 0.0).astype(np.float32)
    np.fill_diagonal(S_total, diag_vals)
    
    # Final check for output finiteness
    if not np.all(np.isfinite(S_total)):
         raise FloatingPointError("Non-finite values in output SSM")

    return S_total.tolist()


def compute_ssm_multi(
    features: dict[str, Sequence[Sequence[float]]],
    weights: dict[str, float],
    *,
    map_to_unit_interval: bool = False,
    strict: bool = True,
) -> List[List[float]]:
    """Compute weighted SSM from multiple features.
    
    Args:
        features: Dictionary mapping feature keys to list of feature vectors.
            All feature lists must have the same length (num_bars).
        weights: Dictionary mapping feature keys to weights.
        map_to_unit_interval: If true, map cosine [-1, 1] to [0, 1].
        strict: If True, raise ValueError for missing keys in features for weights > 0.
    
    Returns:
        Square similarity matrix.
    """
    if not features:
        return []

    # Validate lengths
    num_bars = -1
    active_keys = []
    total_weight = 0.0
    
    for key, weight in weights.items():
        if weight <= 0:
            continue
        vals = features.get(key)
        if vals is None:
            if strict:
                raise ValueError(f"Strict mode: Weights contain key '{key}' which is missing in features.")
            # Missing feature, skip if not strict
            continue
            
        if num_bars == -1:
            num_bars = len(vals)
        else:
            if len(vals) != num_bars:
                raise ValueError(f"Feature {key} has length {len(vals)}, expected {num_bars}")
        
        active_keys.append(key)
        total_weight += weight
    
    if num_bars <= 0 or not active_keys:
        return []
        
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")
    
    # Normalize weights
    norm_weights = {k: weights[k] / total_weight for k in active_keys}
    
    # Try NumPy fast path
    if HAS_NUMPY:
        try:
            return _compute_ssm_multi_numpy(
                features, weights, active_keys, norm_weights, num_bars, map_to_unit_interval
            )
        except Exception as e:
            # Fallback if numpy fails (e.g. memory error, shape mismatch inside, or floating point error)
            print(f"WARNING: NumPy SSM calculation failed (fallback to Python): {e}")
            pass

    matrix: List[List[float]] = [[0.0 for _ in range(num_bars)] for _ in range(num_bars)]
    
    # Pre-check non-zero vectors for diagonals?
    # Diagonal is 1.0 if ANY feature is non-zero?
    # Similarity(A, A) = 1.0 if A != 0.
    # If using weighted sum, and all features A_k are non-zero, then sum(w_k * 1.0) = 1.0.
    # If some feature is zero-vector, cos-sim is 0.0.
    # So diagonal might be < 1.0 if a feature is zero-vector for that bar.
    # To be consistent with existing logic: "1.0 if (sum(pch)>0 or sum(onh)>0)"
    # We will compute diagonal fully or use the same logic.
    
    for i in range(num_bars):
        # Optimization: compute diagonal
        # Task D: Ensure diagonal is 1.0 (if active) or 0.0 (if not active)
        # and DO NOT apply map_to_unit_interval to diagonal.
        diag_sim = 0.0
        active_in_bar = False
        for k in active_keys:
            vec = features[k][i]
            if sum(v*v for v in vec) > 1e-9:
                 active_in_bar = True
                 break
        
        # If any feature is active for this bar, diagonal is 1.0 (Self-Similarity)
        # If the bar is silence (all features 0), diagonal is 0.0.
        if active_in_bar:
             diag_sim = 1.0
        else:
             diag_sim = 0.0
             
        matrix[i][i] = diag_sim
        
        for j in range(i + 1, num_bars):
            # Weighted average similarity
            sim_sum = 0.0
            for k in active_keys:
                s = _cosine_similarity(features[k][i], features[k][j])
                sim_sum += norm_weights[k] * s
            
            if map_to_unit_interval:
                sim_sum = (sim_sum + 1.0) / 2.0
            
            matrix[i][j] = sim_sum
            matrix[j][i] = sim_sum
            
    return matrix
