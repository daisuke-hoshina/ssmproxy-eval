import csv
from pathlib import Path

import pretty_midi
import yaml

from ssmproxy.pipeline import RunConfig, run_evaluation


def _make_simple_midi(output_path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=1.0, end=1.5))
    midi.instruments.append(instrument)
    midi.write(str(output_path))


def test_pipeline_produces_artifacts(tmp_path: Path):
    midi_path = tmp_path / "example.mid"
    _make_simple_midi(midi_path)

    output_root = tmp_path / "outputs"
    run_dir = run_evaluation(
        RunConfig(input_dir=tmp_path, output_root=output_root, run_id="test-run", novelty_L=1, lag_top_k=2)
    )

    config_snapshot = run_dir / "config.yaml"
    metrics_csv = run_dir / "metrics.csv"
    ssm_png = run_dir / "figures" / "ssm" / "example.png"
    novelty_png = run_dir / "figures" / "novelty" / "example.png"

    assert config_snapshot.is_file()
    assert metrics_csv.is_file()
    assert ssm_png.is_file()
    assert novelty_png.is_file()

    with metrics_csv.open(newline="") as fp:
        reader = list(csv.DictReader(fp))
    assert any(row["piece_id"] == "example" for row in reader)
    assert all("lag_min_lag" in row for row in reader)
    assert all("group" in row for row in reader)
    assert reader[0]["group"] == ""
    assert reader[0]["midi_path"] == "example.mid"
    assert reader[0]["bars"] == reader[0]["num_bars"]

    with config_snapshot.open() as fp:
        config_data = yaml.safe_load(fp)
    assert config_data["lag_min_lag"] == 4
    assert config_data["exclude_drums"] is True
