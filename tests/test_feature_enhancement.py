
import pytest
import pretty_midi
from ssmproxy.bar_features import compute_bar_features, get_beat_times
from ssmproxy.ssm import compute_ssm_multi, compute_ssm

def test_beat_grid_logic():
    # Create MIDI with tempo change
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0) # 0.5s per beat
    # Add tempo change at 2.0s to 60bpm (1.0s per beat)
    pm.instruments.append(pretty_midi.Instrument(program=0))
    # Be careful, pretty_midi stores tempo changes in _tempo_change_times and _tempo_changes
    # But usually we use .add_tempo_change? (Not available in some versions?)
    # Let's verify what we can mock or construct.
    # Actually pretty_midi usually parses. Constructing tempo changes manually:
    # pm._tick_to_time is complex.
    # Let's rely on get_beat_times which uses pm.get_beats().
    
    # If we can't easily construct tempo changes programmatically without writing file, 
    # we can mock pm.get_beats().
    
    class MockMIDI:
        def get_beats(self):
            # 0, 0.5, 1.0, 1.5, 2.0 (change), 3.0, 4.0
            return [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        def get_tempo_changes(self):
            return ([0.0], [120.0])
        instruments = []
    
    midi = MockMIDI()
    beats = get_beat_times(midi)
    # expect one appended beat at 5.0 (diff 1.0)
    assert len(beats) == 8
    assert beats[-1] == 5.0 
    assert beats[5] == 3.0

def test_enhanced_features_structure():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, is_drum=False)
    # Add minimal notes: bar 0 beat 0
    note = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5)
    inst.notes.append(note)
    pm.instruments.append(inst)
    
    pid, features = compute_bar_features(
        pm, 
        piece_id="test", 
        quantize_mode="beat_grid", 
        feature_mode="enhanced",
        analysis_beats_per_bar=4,
        steps_per_beat=4
    )
    
    assert isinstance(features, dict)
    assert "pch" in features
    assert "onh" in features # legacy
    assert "chroma_roll" in features
    assert "onh_count" in features
    assert "density" in features
    
    # Check shapes
    num_bars = len(features["pch"])
    assert num_bars >= 1
    assert len(features["chroma_roll"][0]) == 12 * 16
    assert len(features["onh_count"][0]) == 16
    assert len(features["density"]) == num_bars
    assert isinstance(features["density"][0], list) # density is [[val], [val]]

def test_chroma_roll_order_sensitivity():
    """Verify that chroma_roll distinguishes order of notes."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0) # 0.5 sec/beat. bar=2.0s
    inst = pretty_midi.Instrument(program=0)
    
    # Bar 0: C then E
    # C: beat 0 (0.0), E: beat 2 (1.0)
    inst.notes.append(pretty_midi.Note(velocity=10, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=10, pitch=64, start=1.0, end=1.5))
    
    # Bar 1: E then C
    # E: beat 0 (2.0), C: beat 2 (3.0)
    inst.notes.append(pretty_midi.Note(velocity=10, pitch=64, start=2.0, end=2.5))
    inst.notes.append(pretty_midi.Note(velocity=10, pitch=60, start=3.0, end=3.5))
    
    pm.instruments.append(inst)
    
    _, features = compute_bar_features(pm, feature_mode="enhanced")
    bars_pch = features["pch"]
    bars_cr = features["chroma_roll"]
    
    # PCH should be identical (C and E present in both bars)
    # sim(pch[0], pch[1]) ~ 1.0
    from ssmproxy.ssm import _cosine_similarity
    
    sim_pch = _cosine_similarity(bars_pch[0], bars_pch[1])
    assert sim_pch > 0.99
    
    # Chroma Roll should be different (C->E vs E->C)
    sim_cr = _cosine_similarity(bars_cr[0], bars_cr[1])
    assert sim_cr < 0.9 # Should be significantly less than 1.0
    
def test_compute_ssm_multi():
    features = {
        "A": [[1.0, 0.0], [1.0, 0.0]], # Identical
        "B": [[1.0, 0.0], [0.0, 1.0]], # Orthogonal
    }
    
    # Case 1: Only A (weight B=0) -> SSM 1.0
    weights_A = {"A": 1.0, "B": 0.0}
    ssm_A = compute_ssm_multi(features, weights_A)
    assert abs(ssm_A[0][1] - 1.0) < 1e-6
    
    # Case 2: Only B (weight A=0) -> SSM 0.0
    weights_B = {"A": 0.0, "B": 1.0}
    ssm_B = compute_ssm_multi(features, weights_B)
    assert abs(ssm_B[0][1] - 0.0) < 1e-6
    
    # Case 3: Mixed 0.5, 0.5 -> SSM 0.5
    weights_mix = {"A": 0.5, "B": 0.5}
    ssm_mix = compute_ssm_multi(features, weights_mix)
    assert abs(ssm_mix[0][1] - 0.5) < 1e-6
