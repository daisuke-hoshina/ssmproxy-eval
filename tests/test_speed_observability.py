import pytest
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from unittest.mock import patch
from ssmproxy import ssm
from ssmproxy.ssm import compute_ssm_multi

@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
def test_numpy_ssm_consistency():
    """Verify NumPy fast path matches pure Python fallback strictly."""
    
    # 1. Synthesize features
    # Bar 0: Active
    # Bar 1: Active
    # Bar 2: Silent
    # Bar 3: Active
    features = {
        "f1": [
            [1.0, 0.0], # 0
            [0.0, 1.0], # 1
            [0.0, 0.0], # 2 (Silent)
            [1.0, 1.0], # 3
        ],
        "f2": [
            [0.5, 0.5], # 0
            [0.5, 0.5], # 1
            [0.0, 0.0], # 2
            [0.0, 0.0], # 3
        ]
    }
    weights = {"f1": 0.6, "f2": 0.4}
    
    # Run with NumPy (Default if installed)
    with patch("ssmproxy.ssm.HAS_NUMPY", True):
        res_np = compute_ssm_multi(features, weights, map_to_unit_interval=True)
    
    # Run with pure Python (Force fallback)
    with patch("ssmproxy.ssm.HAS_NUMPY", False):
        res_py = compute_ssm_multi(features, weights, map_to_unit_interval=True)
        
    # Check shape
    assert len(res_np) == 4
    assert len(res_py) == 4
    
    # Convert to array for easy comparison
    arr_np = np.array(res_np)
    arr_py = np.array(res_py)
    
    # Assert closeness
    assert np.allclose(arr_np, arr_py, atol=1e-7)
    
    # Verify Diagonals Logic (V4 Spec)
    # Bar 0: Active (f1, f2) -> 1.0
    # Bar 1: Active (f1, f2) -> 1.0
    # Bar 2: Silent (f1, f2) -> 0.0
    # Bar 3: Active (f1) -> 1.0
    diagonals = np.diag(arr_np)
    expected_diagonals = np.array([1.0, 1.0, 0.0, 1.0])
    assert np.allclose(diagonals, expected_diagonals)
    
    # Verify Mapping didn't mess up diagonals
    # If mapping applied to diags, 1.0 -> 1.0, but 0.0 -> 0.5.
    # We expect 0.0.
    assert diagonals[2] == 0.0
    
    # Verify off-diagonal mapping
    # Bar 0 vs Bar 1:
    # f1: [1,0] vs [0,1] -> cos=0.0
    # f2: [0.5,0.5] vs [0.5,0.5] -> cos=1.0 (vector is identical)
    # Norm weights: f1=0.6, f2=0.4
    # Weighted Sum = 0.6*0 + 0.4*1 = 0.4.
    # Mapped: (0.4 + 1) / 2 = 0.7.
    assert np.isclose(arr_np[0, 1], 0.7)


def test_config_validation_robustness(tmp_path):
    """Test that invalid configurations raise ValueError eagerly."""
    from ssmproxy.pipeline import RunConfig, run_evaluation
    
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # Base valid config
    base_config = RunConfig(
        input_dir=input_dir,
        output_root=tmp_path / "output",
        run_id="test_val"
    )
    
    # 1. Invalid beats_per_bar
    base_config.analysis_beats_per_bar = 0
    with pytest.raises(ValueError, match="analysis_beats_per_bar"):
        run_evaluation(base_config)
        
    # 2. Invalid steps_per_beat
    base_config.analysis_beats_per_bar = 4 # restore
    base_config.steps_per_beat = 0
    with pytest.raises(ValueError, match="steps_per_beat"):
        run_evaluation(base_config)
        
    # 3. Invalid Weights (all zero)
    base_config.steps_per_beat = 4 # restore
    base_config.ssm_weights = {"pch": 0.0, "onh": 0.0}
    with pytest.raises(ValueError, match="Total SSM weights must be positive"):
        run_evaluation(base_config)


def test_pipeline_observability(tmp_path):
    """Verify counters and summary JSON are generated."""
    from ssmproxy.pipeline import RunConfig, run_evaluation
    import json
    
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # Create 1 valid MIDI (Long enough to ensure >0 bars)
    # 120 BPM, 4/4 -> 2s per bar. 
    # Add notes for 4 bars (8s).
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(0)
    # Add notes
    for i in range(16): # 16 quarter notes
        inst.notes.append(pretty_midi.Note(100, 60, i*0.5, i*0.5 + 0.5))
    pm.instruments.append(inst)
    pm.write(str(input_dir / "valid.mid"))
    
    output_dir = tmp_path / "output"
    config = RunConfig(
        input_dir=input_dir,
        output_root=output_dir,
        run_id="test_obs"
    )
    
    run_dir = run_evaluation(config)
    
    # Check summary
    summary_path = run_dir / "metrics" / "run_summary.json"
    assert summary_path.exists()
    
    with open(summary_path) as f:
        summary = json.load(f)
        
    print(f"DEBUG SUMMARY: {summary}")
    
    if summary["counts"]["failed"] > 0:
        errors_path = run_dir / "metrics" / "errors.jsonl"
        if errors_path.exists():
             with open(errors_path) as f:
                 for line in f:
                     print(f"ERROR LOG: {line.strip()}")
    
    assert summary["config"]["run_id"] == "test_obs"
    assert summary["counts"]["total_files"] == 1
    # Check if processed or skipped/failed
    # If 0 processed, one of the others must be 1.
    assert summary["counts"]["processed"] == 1, f"Processed 0. Skipped: {summary['counts']['skipped']}, Failed: {summary['counts']['failed']}"
    assert summary["counts"]["failed"] == 0
    assert summary["counts"]["skipped"] == 0
    assert "output_dir" in summary

