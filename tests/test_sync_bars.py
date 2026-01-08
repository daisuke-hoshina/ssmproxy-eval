
import pytest
import pretty_midi
from ssmproxy.bar_features import compute_bar_features
from ssmproxy.dataset_utils import _compute_bars_fast

def create_note_midi(start: float, end: float, tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(
        velocity=100, pitch=60, start=start, end=end
    ))
    pm.instruments.append(inst)
    return pm

def test_bar_count_sync_case_a_overlap():
    """
    Case A: Note overlaps bar boundary.
    Tempo 120 (0.5s/beat), 4 beats/bar -> 2.0s/bar.
    Note: 1.9s to 2.1s.
    Should be 2 bars (Bar 0 and Bar 1).
    """
    pm = create_note_midi(1.9, 2.1)
    
    # 1. Check compute_bar_features
    _, features = compute_bar_features(pm, analysis_beats_per_bar=4)
    pch = features["pch"]
    assert len(pch) == 2, f"Expected 2 bars for overlapping note, got {len(pch)}"
    
    # 2. Check dataset_utils
    bars_utils = _compute_bars_fast(pm, analysis_beats_per_bar=4)
    assert bars_utils == 2, f"Expected 2 bars from utils, got {bars_utils}"
    
    # Consistency
    assert len(pch) == bars_utils

def test_bar_count_sync_case_b_exact_boundary():
    """
    Case B: Note ends exactly on bar boundary.
    Tempo 120 (0.5s/beat), 4 beats/bar -> 2.0s/bar.
    Note: 0.0s to 2.0s.
    Should be 1 bar (Bar 0 only). It ends AT the start of Bar 1 (2.0s), so exclusive logic applies.
    """
    pm = create_note_midi(0.0, 2.0)
    
    # 1. Check compute_bar_features
    _, features = compute_bar_features(pm, analysis_beats_per_bar=4)
    pch = features["pch"]
    assert len(pch) == 1, f"Expected 1 bar for exact boundary note, got {len(pch)}"
    
    # 2. Check dataset_utils
    bars_utils = _compute_bars_fast(pm, analysis_beats_per_bar=4)
    assert bars_utils == 1, f"Expected 1 bar from utils, got {bars_utils}"
    
    # Consistency
    assert len(pch) == bars_utils

def test_bar_count_sync_case_c_just_over_boundary():
    """
    Case C: Note goes just over boundary.
    Note: 0.0s to 2.000001s
    Should be 2 bars.
    """
    pm = create_note_midi(0.0, 2.000001)
    
    _, features = compute_bar_features(pm, analysis_beats_per_bar=4)
    assert len(features["pch"]) == 2
    
    bars_utils = _compute_bars_fast(pm, analysis_beats_per_bar=4)
    assert bars_utils == 2

def test_bar_count_sync_beats_per_bar_change():
    """
    Verify analysis_beats_per_bar argument is respected.
    If beats_per_bar = 2 (1.0s/bar).
    Note: 0.0s to 2.0s.
    Bar 0: 0.0-1.0
    Bar 1: 1.0-2.0
    Ends at 2.0 (start of Bar 2).
    Should be 2 bars.
    """
    pm = create_note_midi(0.0, 2.0)
    
    _, features = compute_bar_features(pm, analysis_beats_per_bar=2)
    assert len(features["pch"]) == 2
    
    bars_utils = _compute_bars_fast(pm, analysis_beats_per_bar=2)
    assert bars_utils == 2
