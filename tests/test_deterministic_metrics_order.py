import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ssmproxy.cli import app

pretty_midi = pytest.importorskip("pretty_midi")


def _write_test_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    instrument = pretty_midi.Instrument(program=0)

    seconds_per_beat = 60.0 / 100.0
    for idx, pitch in enumerate([60, 62, 64, 65]):
        start = idx * seconds_per_beat
        end = start + seconds_per_beat * 0.8
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))

    midi.instruments.append(instrument)
    midi.write(str(path))


def test_metrics_rows_sorted_by_midi_path(tmp_path: Path) -> None:
    midi_dir = tmp_path / "inputs" / "group"
    midi_dir.mkdir(parents=True)
    b_path = midi_dir / "b.mid"
    a_path = midi_dir / "a.mid"

    _write_test_midi(b_path)
    _write_test_midi(a_path)

    output_root = tmp_path / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            str(tmp_path / "inputs"),
            "--output",
            str(output_root),
            "--run-id",
            "deterministic",
            "--novelty-L",
            "2",
            "--lag-top-k",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output

    metrics_csv = output_root / "deterministic" / "metrics" / "ssm_proxy.csv"
    assert metrics_csv.is_file()

    with metrics_csv.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    midi_paths = [row["midi_path"] for row in rows]
    assert midi_paths == ["group/a.mid", "group/b.mid"]
