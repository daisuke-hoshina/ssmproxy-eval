
import pytest
import pretty_midi
from ssmproxy.dataset_utils import _compute_bars_fast
from ssmproxy.bar_features import compute_bar_features

def test_compute_bars_fast_match():
    """Verify _compute_bars_fast matches compute_bar_features bar count."""
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0) # 0.5s/beat
    inst = pretty_midi.Instrument(program=0)
    # Note from 0.0 to 10.0 (20 beats) -> 5 bars exactly?
    # beat 0..19. beat 20 starts at 10.0.
    # time_to_beat_index(10.0) -> max index? 
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=10.0))
    pm.instruments.append(inst)
    
    # 1. Feature computation
    pid, features = compute_bar_features(pm, analysis_beats_per_bar=4)
    bars_feat = len(features["pch"])
    
    # 2. Fast computation
    bars_fast = _compute_bars_fast(pm)
    
    assert bars_feat == bars_fast
    assert bars_feat > 0
    # 10.0s = 20 beats. start=10.0 is not included in note interval [0, 10).
    # Wait, max(start) for bar count uses start time. 
    # Max start is 0.0. 
    # If only one note at 0.0, last_start=0. beat_idx=0. num_bars=1.
    
    # Let's add note at end.
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=9.5, end=10.0))
    # 9.5s = beat 19.
    # beat 19 // 4 = 4. bars = 5 (indices 0,1,2,3,4).
    
    _, features = compute_bar_features(pm, analysis_beats_per_bar=4)
    bars_feat = len(features["pch"])
    bars_fast = _compute_bars_fast(pm)
    
    assert bars_feat == bars_fast
    assert bars_feat == 5
