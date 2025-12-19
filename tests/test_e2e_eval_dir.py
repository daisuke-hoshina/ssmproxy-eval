import csv
from pathlib import Path

import pretty_midi
from typer.testing import CliRunner

from ssmproxy.cli import app


def _write_test_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    instrument = pretty_midi.Instrument(program=0)

    seconds_per_beat = 60.0 / 100.0
    for idx, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        start = idx * seconds_per_beat
        end = start + seconds_per_beat * 0.8
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))

    midi.instruments.append(instrument)
    midi.write(str(path))


def test_cli_eval_dir_generates_outputs(tmp_path: Path) -> None:
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    midi_path = midi_dir / "cli_piece.mid"
    _write_test_midi(midi_path)

    output_root = tmp_path / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            str(midi_dir),
            "--output",
            str(output_root),
            "--run-id",
            "cli-test",
            "--novelty-L",
            "2",
            "--lag-top-k",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output

    run_dir = output_root / "cli-test"
    config_snapshot = run_dir / "config.yaml"
    metrics_csv = run_dir / "metrics.csv"
    ssm_dir = run_dir / "figures" / "ssm"
    novelty_dir = run_dir / "figures" / "novelty"

    assert config_snapshot.is_file()
    assert metrics_csv.is_file()
    assert any(ssm_dir.glob("*.png"))
    assert any(novelty_dir.glob("*.png"))

    with metrics_csv.open(newline="") as fp:
        rows = list(csv.DictReader(fp))

    assert any(row.get("piece_id") == "cli_piece" for row in rows)
