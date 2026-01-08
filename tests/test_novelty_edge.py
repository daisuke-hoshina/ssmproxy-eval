import pytest
from ssmproxy.novelty import compute_novelty

def test_novelty_edge_artifacts_suppressed():
    """
    Test that edge artifacts (high novelty at start/end due to padding) are suppressed.
    We create a blank SSM. Standard Foote kernel with zero-padding usually creates
    high novelty at the edges because the kernel sees 'contrast' between data and padding.
    With the fix, these should be zeroed out or ignored, resulting in no peaks.
    """
    B = 100
    # Completely uniform SSM
    ssm = [[1.0 for _ in range(B)] for _ in range(B)]
    L = 10
    
    # In the old version, this might produce peaks at the edges due to padding.
    # In the new version, valid range is [L, B-L].
    # The edges should be 0.0, so no peaks there.
    result = compute_novelty(ssm, L=L)
    
    # 1) Check no peaks found
    assert len(result.peaks) == 0, f"Found unexpected peaks: {result.peaks}"
    
    # 2) Check peak_rate is 0.0
    assert result.stats["peak_rate"] == 0.0
    
    # 3) Check valid_len logic
    valid_len_expected = max(0, B - 2 * L)
    assert result.stats["valid_len"] == valid_len_expected
    assert result.stats["valid_start"] == L
    assert result.stats["valid_end"] == B - L

def test_novelty_peak_in_valid_range():
    """
    Test that a real peak inside the valid range is preserved and calculated correctly.
    """
    B = 100
    L = 10
    ssm = [[0.0 for _ in range(B)] for _ in range(B)]
    
    # Create a 'checkerboard' block transition in the middle to simulate a structure boundary.
    # This should yield a high novelty peak at mid_point.
    # To get a peak at index 50 with kernel size L, we need contrast.
    # Foote kernel:
    #     + -
    #     - +
    # So if we have:
    # Block A | Block B
    # --------+--------
    # Block B | Block A
    # It matches the kernel.
    
    mid = 50
    # Top-left (0:50, 0:50) -> 1.0 (matches + kernel part)
    for r in range(mid):
        for c in range(mid):
            ssm[r][c] = 1.0
            
    # Bottom-right (50:100, 50:100) -> 1.0 (matches + kernel part)
    for r in range(mid, B):
        for c in range(mid, B):
            ssm[r][c] = 1.0
            
    # Top-right and Bottom-left remain 0.0 (matches - kernel part effectively)
    
    result = compute_novelty(ssm, L=L)
    
    # Should find exactly 1 peak at 49 (boundary of 50-50 blocks)
    assert len(result.peaks) == 1
    assert result.peaks[0] == 49
    
    valid_len = B - 2 * L  # 100 - 20 = 80
    expected_rate = 1 / valid_len
    assert result.stats["peak_rate"] == pytest.approx(expected_rate)
    assert result.stats["peak_rate_raw"] == pytest.approx(1 / B)

def test_short_ssm_valid_range():
    """
    Test behavior when B < 2L (valid_len = 0).
    """
    B = 10
    ssm = [[0.0 for _ in range(B)] for _ in range(B)]
    L = 6 # 2*L = 12 > B
    
    result = compute_novelty(ssm, L=L)
    
    assert result.stats["valid_len"] == 0
    assert result.stats["peak_rate"] == 0.0
    assert len(result.peaks) == 0
