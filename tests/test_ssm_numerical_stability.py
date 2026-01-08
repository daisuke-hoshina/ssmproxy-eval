
import pytest
import numpy as np
import warnings
from ssmproxy.ssm import compute_ssm_multi

def test_no_runtime_warnings_with_garbage_input():
    """
    Ensure that passing NaN/Inf inputs does not trigger RuntimeWarning from NumPy.
    The code should catch FloatingPointError internally or sanitize inputs.
    """
    
    # 1. Setup features with NaN/Inf
    num_bars = 4
    features = {
        "A": [[1.0, 0.0], [0.0, 1.0], [np.nan, 0.0], [np.inf, 1.0]]
    }
    weights = {"A": 1.0}
    
    # 2. Strict check for warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error", RuntimeWarning) # Turn warnings into errors
        
        # This will trigger NumPy path -> sanitized or raise
        ssm = compute_ssm_multi(features, weights, map_to_unit_interval=False)
        
        # Verify result is valid
        assert len(ssm) == num_bars
        # Check all values are finite (0.0 fallback for bad rows)
        s_arr = np.array(ssm)
        assert np.all(np.isfinite(s_arr))
        
        # A=[np.nan, 0] should be treated as silent (0 vector) -> diagonal 0.0 if logic holds?
        # But wait, logic says "active if norm > 1e-9".
        assert ssm[2][2] == 0.0 
        assert ssm[3][3] == 1.0 # Inf -> 0.0. But vector has [0.0, 1.0], so active! Diag 1.0.
        
    # Check no RuntimeWarning caught
    # We used "error" filter, so if warning occurred, it raised exception and failed test.
    
def test_zero_vector_handling():
    """Verify zero vectors result in diagonal 0.0 and no division errors."""
    features = {
        "B": [[0.0, 0.0], [1.0, 0.0]]
    }
    weights = {"B": 1.0}
    
    ssm = compute_ssm_multi(features, weights)
    
    # Bar 0: zero vector. inactive. diag 0.0.
    assert ssm[0][0] == 0.0
    
    # Bar 1: active. diag 1.0.
    assert ssm[1][1] == 1.0
    
    # Cross: 0 dot 1 = 0
    assert ssm[0][1] == 0.0
