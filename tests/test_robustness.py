
import pytest
import pretty_midi
import math
from ssmproxy.ssm import compute_ssm_multi
from ssmproxy.bar_features import compute_bar_features
from ssmproxy.config import _resolve_ssm_weights

def test_config_double_counting_prevention():
    # Case 1: Only legacy weight_onh
    cfg1 = {"weight_onh": 0.8}
    w1 = _resolve_ssm_weights(cfg1)
    # Expect "onh" to be excluded from dict, and use weight_onh for onh_bin?
    # Logic implemented: onh_bin uses default (0.1) if not in cfg1. weight_onh uses legacy key.
    # WAIT, logic in config.py was: onh_bin = ssm_cfg.get("weight_onh_bin", 0.1).
    # Legacy weight_onh is ignored for ssm_weights dict construction in current implementation?
    # Let's verify what we implemented.
    # Implementation: 
    # "onh_bin": float(ssm_cfg.get("weight_onh_bin", 0.1)),
    # No inheritance from weight_onh. This effectively deprecates weight_onh for new pipeline.
    # But ensures "onh" key is NOT in dict.
    assert "onh" not in w1
    assert "onh_bin" in w1
    assert w1["onh_bin"] == 0.8 # Legacy weight propagated
    
    # Case 2: Explicit onh_bin
    cfg2 = {"weight_onh_bin": 0.5, "weight_onh": 0.9}
    w2 = _resolve_ssm_weights(cfg2)
    assert w2["onh_bin"] == 0.5
    assert "onh" not in w2

def test_ssm_strict_mode():
    features = {"A": [[1.0], [1.0]], "B": [[0.5], [0.5]]}
    
    # Case 1: All keys present -> OK
    weights_ok = {"A": 1.0, "B": 1.0}
    compute_ssm_multi(features, weights_ok, strict=True)
    
    # Case 2: Missing key -> Error
    weights_bad = {"A": 1.0, "C": 1.0} # C missing
    with pytest.raises(ValueError, match="Strict mode"):
        compute_ssm_multi(features, weights_bad, strict=True)
        
    # Case 3: Missing key but weight 0 -> OK (should continue)
    weights_zero = {"A": 1.0, "C": 0.0}
    compute_ssm_multi(features, weights_zero, strict=True)

def test_chroma_roll_duration_overlap():
    # Use exact same PC, but different durations in same bar/step setup
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0) # 0.5s/beat, 2.0s/bar
    inst = pretty_midi.Instrument(program=0)
    
    # Bar 0: Short note C5 (60)
    # Start 0.0, End 0.1
    # Step 0 duration is 0.5 / 4 = 0.125.
    # Overlap should be 0.1 - 0.0 = 0.1
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
    
    # Bar 1: Long note C5 (60)
    # Start 2.0 (Bar 1 start), End 2.4
    # Covers multiple steps.
    # Step 0 (2.0-2.125): Overlap 0.125
    # Step 1 (2.125-2.25): Overlap 0.125
    # ...
    # Total chroma roll sum for Bar 1 should be higher than Bar 0 because overlap sum is higher?
    # Wait, chroma_roll is L1 normalized. Sum is 1.0.
    # The value distribution changes, but sum is normalized.
    # If Bar 0 has ONLY this note, it fills Step 0 with X, others 0. Normalized -> Step 0 is 1.0.
    # If Bar 1 has ONLY this note, it fills Step 0,1,2.. with Y. Normalized -> distributed.
    
    # To test overlap INTENSITY, we should check raw values or compare against another note in same bar.
    # Or, compare Bar 0 (short C) vs Bar 1 (Long C).
    # If I add another note D (62) with CONSTANT duration in both bars,
    # then Short C vs Long C will change the ratio of C vs D.
    
    # Bar 0: Short C (0.1s), Constant D (1.0s)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=62, start=0.5, end=1.5)) # D in Bar 0
    
    # Bar 1: Long C (1.0s), Constant D (1.0s)
    # D at 2.5-3.5
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=2.0, end=3.0)) # Long C
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=62, start=2.5, end=3.5)) # D in Bar 1
    
    pm.instruments.append(inst)
    
    pid, features = compute_bar_features(pm, feature_mode="enhanced")
    cr = features["chroma_roll"] # Normalized
    
    # Unnormalized check is not possible directly from output (it's normalized).
    # But we can check ratio.
    # Bar 0: C is short, D is long. D should have much higher values.
    # Bar 1: C is long, D is long. C and D should have comparable values.
    
    # Sum of C bins vs Sum of D bins in Bar 0
    # C is pc 0. D is pc 2.
    c_indices = [i for i in range(len(cr[0])) if (i // 16) == 0]
    d_indices = [i for i in range(len(cr[0])) if (i // 16) == 2]
    
    sum_c0 = sum(cr[0][i] for i in c_indices)
    sum_d0 = sum(cr[0][i] for i in d_indices)
    
    sum_c1 = sum(cr[1][i] for i in c_indices)
    sum_d1 = sum(cr[1][i] for i in d_indices)
    
    print(f"Bar 0 (Short C, Long D): Sum C={sum_c0:.3f}, Sum D={sum_d0:.3f}")
    print(f"Bar 1 (Long C, Long D):  Sum C={sum_c1:.3f}, Sum D={sum_d1:.3f}")
    
    # Expectation: In Bar 0, D dominates C significantly.
    assert sum_d0 > sum_c0 * 2.0 
    
    # Expectation: In Bar 1, C and D are similar (both 1.0s overlap duration).
    # Overlap math: C (2.0-3.0), D (2.5-3.5). Both 1.0s total.
    # Should be roughly equal sum.
    diff = abs(sum_c1 - sum_d1)
    assert diff < 0.1 # Tolerance

def test_beat_extrapolation_for_last_note():
    # Make a note that goes WAY beyond initial beats
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0) # 0.5s/beat. 
    # Beats: 0, 0.5, ...
    # Add note at 100.0s
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=100.0, end=102.0))
    pm.instruments.append(inst)
    
    # This triggers beat extrapolation loop
    # If not extrapolated, it might crash or index error or infinite loop if logic wrong.
    # We added while beats[-1] < last_end loop.
    
    import time
    t0 = time.time()
    _, features = compute_bar_features(pm, feature_mode="enhanced")
    dur = time.time() - t0
    
    assert len(features["pch"]) > 0
    assert dur < 2.0 # Should be fast enough
