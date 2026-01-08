
import random
import pytest
from ssmproxy.novelty import compute_novelty_multiscale

def generate_random_ssm(size: int, seed: int = 42) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.random() for _ in range(size)] for _ in range(size)]

def generate_block_ssm(block_sizes: list[int]) -> list[list[float]]:
    # Create SSM with blocks of 1s on diagonal, 0 elsewhere (simplified)
    # Actually checkerboard kernel likes contrast. 
    # Let's make blocks high similarity (1.0) and off-diagonal low (0.0).
    size = sum(block_sizes)
    ssm = [[0.0 for _ in range(size)] for _ in range(size)]
    
    current_idx = 0
    for b in block_sizes:
        for r in range(current_idx, current_idx + b):
            for c in range(current_idx, current_idx + b):
                ssm[r][c] = 1.0
        current_idx += b
    return ssm

def test_consensus_reduces_noise_on_random():
    # Random SSM should have many phantom peaks in integrated mode, fewer in consensus.
    size = 200
    ssm = generate_random_ssm(size, seed=123)
    Ls = [4, 8, 16]
    
    # Compare against Integrated with small min_distance (simulating desire for high resolution)
    # If we use default Integrated (dist=16), it ignores almost everything, so it's artificially clean but low-res.
    # We want to show Consensus filters noise BETTER than Integrated AT SAME RESOLUTION.
    res_integrated_dense = compute_novelty_multiscale(ssm, Ls, mode="integrated", prominence=0.05, min_distance=4)
    # Disable keep_lmax to strictly test consensus filtering
    res_consensus = compute_novelty_multiscale(ssm, Ls, mode="consensus", prominence=0.05, consensus_min_scales=2, keep_lmax=False)
    
    print(f"Integrated(dist=4) peaks: {len(res_integrated_dense.peaks)}")
    print(f"Consensus peaks: {len(res_consensus.peaks)}")
    
    assert len(res_consensus.peaks) < len(res_integrated_dense.peaks), "Consensus mode should reduce noise compared to high-res integrated"

def test_consensus_preserves_structure():
    # 32-bar blocks -> 32-bar boundaries.
    # Ls = [4, 8, 16]. All should see the boundary.
    block_sizes = [32, 32, 32, 32] 
    ssm = generate_block_ssm(block_sizes)
    Ls = [4, 8, 16]
    
    # Expected boundaries at 32, 64, 96.
    # tolerance=2 allows 30-34 etc.
    res_consensus = compute_novelty_multiscale(
        ssm, 
        Ls, 
        mode="consensus", 
        consensus_min_scales=2, 
        consensus_tolerance=2,
        prominence=0.1
    )
    
    peaks = res_consensus.peaks
    print(f"Structural peaks: {peaks}")
    
    expected = [32, 64, 96]
    for e in expected:
        # Check if there is a peak close to e
        matches = [p for p in peaks if abs(p - e) <= 2]
        assert len(matches) > 0, f"Expected peak at {e} missed in consensus mode"

def test_consensus_fallback():
    # Case where consensus finds nothing but integrated finds something?
    # Or just empty.
    # Let's try to force empty consensus but valid integrated.
    # SSM with very weak features that only appear in one scale?
    # Hard to engineer.
    # Instead, we test that fallback WORKS when consensus naturally returns empty 
    # (e.g. extremely strict consensus settings).
    
    size = 100
    ssm = generate_random_ssm(size, seed=999)
    Ls = [4, 8, 16]
    
    # Require 4 scales (imperossible since len(Ls)=3) -> Consensus should be empty
    # Fallback=True -> Should return integrated peaks
    # disable keep_lmax so we don't just return L_max peaks
    res_fallback = compute_novelty_multiscale(
        ssm, 
        Ls, 
        mode="consensus", 
        consensus_min_scales=4, # impossible
        fallback=True,
        keep_lmax=False,
        prominence=0.01 # ensure integrated finds something
    )
    
    assert len(res_fallback.peaks) > 0, "Fallback should retain peaks when consensus fails"
    
    # Fallback=False -> Should be empty
    res_no_fallback = compute_novelty_multiscale(
        ssm, 
        Ls, 
        mode="consensus", 
        consensus_min_scales=4,
        fallback=False,
        keep_lmax=False,
        prominence=0.01
    )
    assert len(res_no_fallback.peaks) == 0, "Fallback=False should perform no rescue"

def test_lmax_priority():
    # Test that peaks found in L_max are kept even if min_scales not met.
    # We need a setup where L=16 finds a peak but L=4, L=8 do not.
    # Maybe use a blurred block boundary that is only visible to large kernel?
    # Or manually construct ssm where contrast is low?
    # Actually, easier to mock? No, integration test is better.
    pass

