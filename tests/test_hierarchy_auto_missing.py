
import pytest
from ssmproxy.pipeline import resolve_auto_hierarchy_metrics

def test_resolve_auto_missing_index_remains_none():
    """Verify that if slope/adjdiff returns None, the result is None (not 0.0)."""
    
    # 1. Setup prominence that yields None for slope (e.g. only 1 point)
    # L0 = 4. Points at 4*1(4). But 4*2(8) is None.
    # So x=[log2(1)], y=[...]. Not enough for slope.
    
    prominence = [None] * 20
    prominence[4] = 0.5
    # prominence[8] is None
    
    
    idx, mult, lag_used, valid, reason = resolve_auto_hierarchy_metrics(
        prominence=prominence,
        lag_base_period=4,
        fallback_mode=False,
        fallback_index=None,
        auto_mode="slope",
        max_levels=3
    )
    
    assert idx is None, f"Expected None for single point prominence, got {idx}"
    assert valid == 0
    assert reason == "insufficient_points"
    
    # Meta info: max_m should be 1 (since 4 exists)
    assert mult == 1
    assert lag_used == 4

def test_resolve_auto_fallback_missing_remains_none():
    """Verify fallback mode preserves None if fallback_index is None."""
    
    idx, mult, lag_used, valid, reason = resolve_auto_hierarchy_metrics(
        prominence=[],
        lag_base_period=4,
        fallback_mode=True,
        fallback_index=None, # e.g. E8 or E4 missing
        auto_mode="slope",
        max_levels=3
    )
    
    assert idx is None, f"Expected None in fallback if fallback_index is None, got {idx}"
    assert mult is None
    assert lag_used is None
    assert valid == 0
    assert reason == "fallback_no_fixed_index"

def test_resolve_auto_valid_calculation():
    """Verify valid calculation still works."""
    # L0=4. Points at 4, 8.
    prominence = [None] * 20
    prominence[4] = 1.0 # log2(1)=0, y=1
    prominence[8] = 0.5 # log2(2)=1, y=0.5
    # Slope = (0.5-1)/(1-0) = -0.5
    
    idx, mult, lag_used, valid, reason = resolve_auto_hierarchy_metrics(
        prominence=prominence,
        lag_base_period=4,
        fallback_mode=False,
        fallback_index=None,
        auto_mode="slope",
        max_levels=3
    )
    
    assert idx is not None
    assert abs(idx - (-0.5)) < 1e-6
    assert mult == 2 # 8 is valid
    assert lag_used == 8
    assert valid == 1
    assert reason == "ok"
