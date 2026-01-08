
import pytest
import numpy as np
import math
from ssmproxy.bar_features import compute_bar_features
import pretty_midi

def test_density_is_2d_and_cosine_sensitive():
    """
    Verify that the 'density' feature is returning a 2D vector for each bar,
    and that cosine similarity calculation honors magnitude differences.
    """
    # 1. Create a dummy MIDI with 2 bars:
    # Bar 0: Low density (1 note)
    # Bar 1: High density (10 notes)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(0)
    
    # Bar 0 (0.0 to 2.0s): 1 note
    inst.notes.append(pretty_midi.Note(60, 60, 0.0, 0.5))
    
    # Bar 1 (2.0 to 4.0s): 10 notes
    for i in range(10):
        t = 2.0 + (i * 0.1)
        inst.notes.append(pretty_midi.Note(60+i, 60, t, t+0.05))
        
    pm.instruments.append(inst)
    
    # 2. Extract features
    pid, features = compute_bar_features(
        pm, 
        piece_id="density_test",
        # Use simple config
        analysis_beats_per_bar=4, 
        steps_per_beat=4, 
        feature_mode="enhanced"
    )
    
    density = features["density"]
    assert len(density) == 2, "Expected 2 bars"
    
    vec0 = density[0]
    vec1 = density[1]
    
    # 3. Check shape: expected [log1p(d), 1.0]
    assert len(vec0) == 2, f"Expected 2D vector, got {vec0}"
    assert vec0[1] == 1.0, "Expected second component to be 1.0 (padding)"
    
    # 4. Check values
    # Bar 0: 1 note. roughly 1 onset if not duration weighted... 
    # Actually bar_features calculates density as sum of durations or count? 
    # Let's check bar_features.py logic broadly: overlap sum.
    # Note 1: 0.5s duration. 
    # Total duration 2.0s per bar (120bpm, 4 beats).
    # so density ~ 0.5/2.0 ? Or raw sum?
    # Usually it's raw sum of overlap or coverage.
    # Regardless, vec0[0] should be log1p(d0).
    
    print(f"Vec0: {vec0}, Vec1: {vec1}")
    
    # 5. Verify Cosine Similarity is NOT 1.0
    # cos(u, v) = (u . v) / (|u| |v|)
    # If they were 1D scalar [k*d], cos would be 1.0.
    # with [x, 1], angle changes.
    
    def cosine_sim(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    sim = cosine_sim(vec0, vec1)
    print(f"Similarity: {sim}")
    assert sim < 0.999, f"Cosine similarity should not be 1.0, got {sim}. Density feature is failing to distinguish differences."
    
