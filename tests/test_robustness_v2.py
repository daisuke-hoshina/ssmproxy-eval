
import pytest
import pretty_midi
import math
from pathlib import Path
from dataclasses import dataclass
from ssmproxy.pipeline import RunConfig, run_evaluation
from ssmproxy.bar_features import compute_bar_features, get_beat_times
from ssmproxy.dataset_utils import _compute_bars_fast

def test_run_config_mutable_defaults():
    """Task A: Ensure RunConfig defaults are not shared mutable objects."""
    c1 = RunConfig(input_dir=Path("."))
    c2 = RunConfig(input_dir=Path("."))
    
    # Modify list in c1
    if c1.lag_hierarchy_auto_harmonics is not None:
        c1.lag_hierarchy_auto_harmonics.append(999)
        
    assert 999 not in c2.lag_hierarchy_auto_harmonics
    
    # Modify dict in c1
    c1.ssm_weights["new_key"] = 1.0
    assert "new_key" not in c2.ssm_weights

def test_density_feature_structure():
    """Task C: Ensure density is 2D [log1p(d), 1.0]."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    # Add 3 notes in bar 0
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=62, start=0.5, end=0.6))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=1.0, end=1.1))
    pm.instruments.append(inst)
    
    pid, features = compute_bar_features(pm, feature_mode="enhanced")
    density = features["density"]
    
    # Check structure
    assert len(density) > 0
    assert isinstance(density[0], list)
    assert len(density[0]) == 2
    assert density[0][1] == 1.0
    
    # Value check: 3 notes. log1p(3) ≈ 1.386
    # Note: density calculation in compute_bar_features simply counts note starts in that bar.
    # Bar 0 (0.0-2.0s). All 3 notes start there.
    assert density[0][0] == pytest.approx(math.log1p(3.0))

def test_bar_count_consistency():
    """Task B: Ensure bar_features and dataset_utils agree on bar count with short beats."""
    # Scenario: Beats are [0.0, 0.5]. Note is at 10.0s.
    # get_beat_times should extrapolate.
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=10.0, end=10.5))
    pm.instruments.append(inst)
    
    # Verify get_beat_times behavior
    beats = get_beat_times(pm, target_end_time=10.5)
    assert beats[-1] >= 10.5
    
    # Verify dataset_utils
    bars_ds = _compute_bars_fast(pm)
    
    # Verify bar_features
    _, features = compute_bar_features(pm)
    bars_feat = len(features["pch"])
    
    assert bars_ds == bars_feat
    assert bars_ds > 5 # Should be around 10s / 2s/bar = 5 bars. 

def test_config_safety_validation():
    """Task D: Ensure incompatible config raises ValueError."""
    config = RunConfig(
        input_dir=Path("."),
        feature_mode="enhanced",
        quantize_mode="legacy_fixed_tempo"
    )
    
    # Mocking _iter_midi_files to avoid empty loop or file IO?
    # Actually run_evaluation checks config at START.
    # So even with empty dir it should raise if check is first.
    
    # We need to make sure it doesn't fail on input_dir not existing (it might rglobi).
    # But check is before loop.
    
    with pytest.raises(ValueError, match="requires quantize_mode='beat_grid'"):
        run_evaluation(config)

def test_analysis_beats_per_bar_consistency():
    """Task 1: Ensure analysis_beats_per_bar consistency between dataset_utils and bar_features."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    # Be sure to create enough length to have differing bar counts for different beats_per_bar
    # Tempo 120 -> 0.5s/beat.
    # Case 1: 4 beats/bar -> 2.0s/bar. Note at 10.0-10.5 -> ~5 or 6 bars.
    # Case 2: 3 beats/bar -> 1.5s/bar. Note at 10.0-10.5 -> 10.0 / 1.5 = 6.66 -> 7 bars.
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=10.0, end=10.5))
    pm.instruments.append(inst)
    
    # Check 4 beats per bar (default)
    bars_ds_4 = _compute_bars_fast(pm, analysis_beats_per_bar=4)
    _, features_4 = compute_bar_features(pm, analysis_beats_per_bar=4)
    assert bars_ds_4 == len(features_4["pch"])
    
    # Check 3 beats per bar
    bars_ds_3 = _compute_bars_fast(pm, analysis_beats_per_bar=3)
    _, features_3 = compute_bar_features(pm, analysis_beats_per_bar=3)
    assert bars_ds_3 == len(features_3["pch"])
    
    # Verify they are indeed different (sanity check on test setup)
    assert bars_ds_4 != bars_ds_3

def test_empty_notes_safety_v3():
    """Task 2: Ensure empty song returns valid dict for strict SSM."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    # No notes
    
    pid, features = compute_bar_features(pm, feature_mode="enhanced")
    
    # Verify required keys exist
    expected_keys = ["pch", "onh", "onh_bin", "onh_count", "chroma_roll", "density"]
    for k in expected_keys:
        assert k in features, f"Missing key {k} in empty features"
        assert features[k] == []
        
    # Verify strict SSM doesn't crash given these empty lists
    from ssmproxy.ssm import compute_ssm_multi
    weights = {"pch":0.3,"onh_bin":0.1,"onh_count":0.2,"chroma_roll":0.4,"density":0.0}
    
    # ssm.py logic: if features are empty, check ssm logic.
    # compute_ssm_multi usually loops over bars. If rows=0, it should handle gracefully.
    # Let's see if ssm.py handles empty lists by returning empty list.
    ssm = compute_ssm_multi(features, weights, strict=True)
    assert ssm == []

