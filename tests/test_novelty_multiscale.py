
import pytest
from ssmproxy.novelty import compute_novelty, compute_novelty_multiscale
from ssmproxy.metrics import build_piece_metrics

def test_metrics_valid_range():
    """Verify that metrics use valid range."""
    B = 100
    L = 10
    
    # Create novelty with valid range stats
    novelty = compute_novelty([[0]*B]*B, L=L) # Dummy compute to get structure
    
    # Manually inject values
    # Edges (outside valid [10, 90]) are huge, center is small
    # Valid range: 10 to 90
    novelty.novelty = [100.0] * B # Default huge
    for i in range(10, 90):
        novelty.novelty[i] = 1.0 # Valid range small
        
    metrics = build_piece_metrics(
        piece_id="test", num_bars=B, novelty=novelty, lag_energy=0.0, best_lag=0, lag_min_lag=4
    )
    
    # std of value 1.0 is 0.0
    # if it included 100.0, it would be huge
    assert metrics.novelty_std == 0.0
    assert metrics.novelty_tv == 0.0
    assert metrics.novelty_topk_mean == 1.0

def test_multiscale_novelty_integration():
    """Verify multi-scale novelty computation."""
    B = 100
    Ls = [5, 10]
    ssm = [[0.0]*B for _ in range(B)]
    
    # Peak at 50, L=5 requires 5x5 blocks
    # Peak at 50, L=10 requires 10x10 blocks
    # Create 10x10 blocks around 50.
    # L=5 will see it perfectly. L=10 will see it perfectly.
    
    mid = 50
    # Top-left and bottom-right blocks of size 10
    block_size = 10
    for r in range(mid - block_size, mid):
        for c in range(mid - block_size, mid):
             ssm[r][c] = 1.0
    for r in range(mid, mid + block_size):
        for c in range(mid, mid + block_size):
             ssm[r][c] = 1.0
             
    res = compute_novelty_multiscale(ssm, Ls=Ls)
    
    # Valid range should be based on max L = 10
    assert res.stats["L"] == 10
    assert res.stats["valid_start"] == 10
    assert res.stats["valid_end"] == 90
    assert res.stats["valid_len"] == 80
    
    # Should detect peak at 50 (actually 49 in 0-indexed boundary logic usually)
    # The previous test found 49. It depends on exact kernel alignment.
    # Check near 49/50.
    peaks = res.peaks
    assert len(peaks) > 0
    assert any(48 <= p <= 51 for p in peaks)
    
    # Check values outside valid range are 0.0
    assert res.novelty[0] == 0.0
    assert res.novelty[9] == 0.0
    assert res.novelty[90] == 0.0
    assert res.novelty[99] == 0.0

def test_multiscale_dominance():
    """
    Test where one scale has a peak and another doesn't (or is smaller).
    Multi-scale (max) should pick it up.
    """
    B = 50
    Ls = [4, 16] 
    ssm = [[0.0]*B for _ in range(B)]
    
    # Create a small feature compatible with L=4 but "too fast" or high freq for L=16?
    # Or just a small checkerboard of size 4.
    mid = 25
    sz = 4
    for r in range(mid-sz, mid):
        for c in range(mid-sz, mid):
            ssm[r][c] = 1.0
    for r in range(mid, mid+sz):
        for c in range(mid, mid+sz):
            ssm[r][c] = 1.0
            
    # L=4 should see high normalized novelty.
    # L=16 might see something but likely lower or more spread out?
    # Actually L=16 kernel is huge (33x33), window sum might be diluted?
    # Normalized, L=4 peak should be ~1.0. 
    
    res = compute_novelty_multiscale(ssm, Ls=Ls)
    
    # Valid range: 16 to 34 (50-16).
    # Peak is at 25. 25 is inside valid range [16, 34].
    
    # Expect peak near 25
    assert any(abs(p - 25) <= 1 for p in res.peaks)
    
    # Confirm valid range 16->34
    assert res.stats["valid_start"] == 16
    assert res.stats["valid_end"] == 34
