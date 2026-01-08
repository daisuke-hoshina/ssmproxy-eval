
import json
import pytest
from ssmproxy.pipeline import RunConfig, run_evaluation
import pretty_midi

def test_pipeline_logging_correct_piece_id_on_failure(tmp_path):
    """
    Regression test for a bug where piece_id from a previous successful iteration
    leaked into the error log of a subsequent failed iteration.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # 1. Create a valid MIDI file (lexicographically first)
    # "a_valid.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 0, 1))
    pm.instruments.append(inst)
    pm.write(str(input_dir / "a_valid.mid"))
    
    # 2. Create a corrupted MIDI file (lexicographically second)
    # "z_bad.mid" - contains junk bytes
    bad_path = input_dir / "z_bad.mid"
    with open(bad_path, "wb") as f:
        f.write(b"NOT A MIDI FILE")
        
    output_dir = tmp_path / "output"
    config = RunConfig(
        input_dir=input_dir,
        output_root=output_dir,
        run_id="test_log_bug",
        fail_fast=False # Important: must continue after error
    )
    
    run_dir = run_evaluation(config)
    
    # 3. Check errors.jsonl
    errors_path = run_dir / "metrics" / "errors.jsonl"
    assert errors_path.exists()
    
    with open(errors_path, "r") as f:
        lines = [json.loads(line) for line in f]
        
    assert len(lines) == 1
    err = lines[0]
    
    assert "z_bad" in err["midi_path"]
    # The bug causes this to be "a_valid" because it reused the variable from the loop.
    # We want it to be "z_bad" (derived from filename if load failed).
    assert err["piece_id"] == "z_bad", f"Expected piece_id 'z_bad', got '{err['piece_id']}'"
