
import random
import numpy as np
import pretty_midi
from ssmproxy.toy_generator import _generate_hierarchical_v2, BPM
from ssmproxy.bar_features import compute_bar_features
from ssmproxy.ssm import compute_ssm
from ssmproxy.lag import compute_lag_energy

def test_hierarchical_v2_bars():
    rng = random.Random(42)
    # Test varying bar counts
    for bars in [32, 64, 96, 128]:
        midi = _generate_hierarchical_v2(rng, bars=bars)
        # Verify length in bars
        # Last note end time
        max_end = 0.0
        for inst in midi.instruments:
            for note in inst.notes:
                max_end = max(max_end, note.end)
        
        # Duration in seconds = bars * 4 * (60/BPM)
        # BPM=120 -> 0.5s/beat -> 2.0s/bar
        expected_dur = bars * 2.0
        assert abs(max_end - expected_dur) < 3.0, f"Expected {bars} bars ({expected_dur}s), got {max_end}s"

def test_hierarchical_v2_structure_dominance():
    # Verify that E(32) > E(8)
    rng = random.Random(123)
    bars = 128 # 4 sections of 32
    midi = _generate_hierarchical_v2(rng, bars=bars)
    
    # Compute Features
    _, features = compute_bar_features(midi, feature_mode="enhanced")
    
    # Combine PCH and ONH
    pch = np.array(features["pch"])
    onh = np.array(features["onh_bin"])
    # Simple concat
    combined = np.hstack([pch, onh])
    
    # Compute SSM
    ssm = compute_ssm(combined, combined)
    
    # Compute Lag
    _, _, lag_energies = compute_lag_energy(ssm, min_lag=1, return_full=True)
    
    e8 = lag_energies[8] if 8 < len(lag_energies) else 0.0
    # Lag 32 should comprise:
    # 0->32, 1->33...
    # Since we have 128 bars, max lag is 127.
    # Lag 32 is valid.
    e32 = lag_energies[32] if 32 < len(lag_energies) else 0.0
    
    print(f"E(8)={e8}, E(32)={e32}")
    
    # E(32) should be significantly higher than E(8)
    assert e32 > e8, f"Lag 32 energy ({e32}) should be higher than Lag 8 energy ({e8})"
