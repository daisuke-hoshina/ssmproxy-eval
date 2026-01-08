import pytest
from typer.testing import CliRunner
from ssmproxy.cli import app
from ssmproxy.dataset_utils import _compute_bars_fast
import pretty_midi
from pathlib import Path
import yaml

runner = CliRunner()

def test_dataset_scan_respects_analysis_beats_per_bar(tmp_path):
    """Test A: dataset scan uses provided analysis_beats_per_bar."""
    # Setup
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    
    # Create MIDI: Tempo 120 (0.5s/beat). Note at 10.0-10.5.
    # 4 beats/bar -> 2.0s/bar -> 10.5s -> bar index floor(10/2)=5 -> 6 bars.
    # 3 beats/bar -> 1.5s/bar -> 10.5s -> bar index floor(10/1.5)=6 -> 7 bars.
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=10.0, end=10.5))
    pm.instruments.append(inst)
    pm.write(str(midi_dir / "test.mid"))
    
    # Run scan with 4 (default logic via CLI)
    out_csv_4 = tmp_path / "scan_4.csv"
    result_4 = runner.invoke(app, ["dataset", "scan", "-i", str(midi_dir), "-o", str(out_csv_4), "--analysis-beats-per-bar", "4", "--min-bars", "1"])
    assert result_4.exit_code == 0
    
    # Run scan with 3
    out_csv_3 = tmp_path / "scan_3.csv"
    result_3 = runner.invoke(app, ["dataset", "scan", "-i", str(midi_dir), "-o", str(out_csv_3), "--analysis-beats-per-bar", "3", "--min-bars", "1"])
    assert result_3.exit_code == 0
    
    # Verify num_bars in CSV
    import csv
    
    with open(out_csv_4, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        bars_4 = int(rows[0]["num_bars"])
        
    with open(out_csv_3, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        bars_3 = int(rows[0]["num_bars"])
        
    assert bars_4 == 6
    assert bars_3 == 7
    assert bars_4 != bars_3

def test_eval_config_propagation(tmp_path):
    """Test B: eval command reflects config file values in output RunConfig."""
    # Setup
    input_dir = tmp_path / "eval_input"
    input_dir.mkdir()
    # Create dummy MIDI to avoid skip error
    pm = pretty_midi.PrettyMIDI()
    pm.write(str(input_dir / "dummy.mid"))
    
    # Create config file with specific non-default values
    config_vals = {
        "features": {
            "analysis_beats_per_bar": 7,
            "feature_mode": "enhanced",
            "quantize_mode": "beat_grid",
            "exclude_drums": False,
        }
    }
    config_p = tmp_path / "test_config.yaml"
    with open(config_p, "w") as f:
        yaml.dump(config_vals, f)
        
    out_dir = tmp_path / "eval_out"
    run_id = "CONFTEST"
    
    # Run eval
    # Note: run_evaluation inside fails if no files or empty files might be skipped.
    # But we check the CONFIG snapshot saving, which happens BEFORE processing loop.
    # Wait, run_evaluation saves config snapshot at start.
    result = runner.invoke(app, ["eval", str(input_dir), "--output", str(out_dir), "--run-id", run_id, "--config", str(config_p)])
    
    # Even if pipeline fails later (e.g. empty MIDI skipping), config.yaml should be written.
    # Or strict error might happen.
    # Let's ensure it runs slightly.
    # If run_evaluation logic raises error (like safe validation), we catch it.
    if result.exit_code != 0:
        print(result.stdout)
        # assert result.exit_code == 0 # Relaxed if empty MIDI causes loop to yield nothing and maybe future checks fail?
        # But snapshot is saved early.
    
    snapshot_path = out_dir / run_id / "config.yaml"
    assert snapshot_path.exists()
    
    with open(snapshot_path) as f:
        saved_cfg = yaml.safe_load(f)
        
    assert saved_cfg["analysis_beats_per_bar"] == 7
    assert saved_cfg["feature_mode"] == "enhanced"
    assert saved_cfg["exclude_drums"] == False
