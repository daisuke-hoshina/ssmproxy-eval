import pytest
from pathlib import Path
import pretty_midi
import json
import logging
from ssmproxy.pipeline import RunConfig, run_evaluation
from ssmproxy.ssm import compute_ssm_multi
from ssmproxy.cli import app
from typer.testing import CliRunner

runner = CliRunner()

def test_eval_error_resilience(tmp_path):
    """Test that eval continues despite bad MIDI files and logs errors."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # 1. Valid MIDI
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 0, 1))
    pm.instruments.append(inst)
    pm.write(str(input_dir / "valid.mid"))
    
    # 2. Bad MIDI (garbage content)
    with open(input_dir / "bad.mid", "wb") as f:
        f.write(b"NOT A MIDI FILE")
        
    output_dir = tmp_path / "output"
    
    config = RunConfig(
        input_dir=input_dir,
        output_root=output_dir,
        run_id="test_run",
        fail_fast=False # Default behavior ensuring resilience
    )
    
    # Run pipeline
    run_dir = run_evaluation(config)
    
    # Verify outputs
    # 1. Config snapshot exists
    assert (run_dir / "config.yaml").exists()
    
    # 2. Metrics CSV exists and contains 1 valid entry
    # (Checking basic file existence first)
    # Canonical path: output/metrics/ssm_proxy.csv
    assert (run_dir / "metrics" / "ssm_proxy.csv").exists()
    
    # Check content roughly
    with open(run_dir / "metrics" / "ssm_proxy.csv") as f:
         lines = f.readlines()
         # Header + 1 entry if valid file succeeded
         # If valid file failed (e.g. empty bars, 0 bars), then only header.
         # But minimal note should yield bars. 120bpm, 1s note -> 2 beats -> <1 bar?
         # 4 beats/bar. 2 beats = 0.5 bars. 
         # get_beat_times ensures beat grid covers last note.
         # last_end=1.0. beats=[0.0, 0.5, 1.0]. last_beat_idx=2.
         # num_bars = 2 // 4 + 1 = 0 + 1 = 1 bar.
         # So it should be valid.
         assert len(lines) >= 2
    
    # 3. errors.jsonl exists and contains 1 error
    errors_path = run_dir / "metrics" / "errors.jsonl"
    assert errors_path.exists()
    
    with open(errors_path) as f:
        lines = f.readlines()
        assert len(lines) == 1
        err = json.loads(lines[0])
        assert err["piece_id"] == "bad"
        assert "valid.mid" not in err["midi_path"] # confirm valid one didn't error
        
def test_ssm_diagonal_logic_enhanced():
    """Verify SSM diagonals are strictly 1.0/0.0 and unmapped in enhanced mode."""
    # Create synthetic features: 2 bars
    # Bar 0: active
    # Bar 1: silent (all zero)
    features = {
        "f1": [[1.0, 0.0], [0.0, 0.0]], 
        "f2": [[0.0, 1.0], [0.0, 0.0]]
    }
    weights = {"f1": 0.5, "f2": 0.5}
    
    # Case 1: No mapping
    ssm = compute_ssm_multi(features, weights, map_to_unit_interval=False)
    assert ssm[0][0] == 1.0 # Active
    assert ssm[1][1] == 0.0 # Silent
    
    # Case 2: With mapping
    # Diagonals should STILL be 1.0 and 0.0 (NOT mapped to 1.0 and 0.5)
    ssm_mapped = compute_ssm_multi(features, weights, map_to_unit_interval=True)
    assert ssm_mapped[0][0] == 1.0 
    assert ssm_mapped[1][1] == 0.0
    
    # Verify off-diagonal mapping logic applies
    # Let's make Bar 1 non-zero but opposite to Bar 0 to test mapping
    # Bar 0: [1, 0]
    # Bar 1: [-1, 0] -> cos sim = -1.0
    features_opp = {
        "f1": [[1.0, 0.0], [-1.0, 0.0]]
    }
    weights_opp = {"f1": 1.0}
    
    ssm_opp = compute_ssm_multi(features_opp, weights_opp, map_to_unit_interval=True)
    # Diagonal 0: active -> 1.0
    assert ssm_opp[0][0] == 1.0
    # Diagonal 1: active -> 1.0
    assert ssm_opp[1][1] == 1.0
    # Off-diagonal: -1.0 -> mapped to ( -1 + 1 ) / 2 = 0.0
    assert abs(ssm_opp[0][1] - 0.0) < 1e-6
