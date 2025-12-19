import csv
from pathlib import Path

import pretty_midi
import yaml
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
    group_dir = midi_dir / "repeat"
    group_dir.mkdir(parents=True)
    midi_path = group_dir / "cli_piece.mid"
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
    metrics_csv = run_dir / "metrics" / "ssm_proxy.csv"
    legacy_metrics_csv = run_dir / "metrics.csv"
    ssm_dir = run_dir / "figures" / "ssm"
    novelty_dir = run_dir / "figures" / "novelty"
    summary_dir = run_dir / "summary"

    assert config_snapshot.is_file()
    assert metrics_csv.is_file()
    if legacy_metrics_csv.is_file():
        assert legacy_metrics_csv.read_text() == metrics_csv.read_text()
    assert any(ssm_dir.glob("*.png"))
    assert any(novelty_dir.glob("*.png"))
    assert not list((run_dir / "figures" / "lag").glob("*.png"))

    with config_snapshot.open() as fp:
        config_data = yaml.safe_load(fp)
    assert config_data["lag_min_lag"] == 4
    assert config_data["exclude_drums"] is True

    with metrics_csv.open(newline="") as fp:
        rows = list(csv.DictReader(fp))

    assert any(row.get("piece_id") == "cli_piece" for row in rows)
    assert all("lag_min_lag" in row for row in rows)
    assert all("group" in row for row in rows)
    assert rows[0]["group"] == "repeat"
    assert rows[0]["midi_path"] == "repeat/cli_piece.mid"
    assert rows[0]["bars"] == rows[0]["num_bars"]

    runner = CliRunner()
    report_result = runner.invoke(
        app,
        [
            "report",
            "run",
            "--eval-out",
            str(run_dir),
        ],
    )
    assert report_result.exit_code == 0, report_result.output

    joined_csv = summary_dir / "metrics_joined.csv"
    group_stats_csv = summary_dir / "metrics_group_stats.csv"
    figures_dir = summary_dir / "figures"

    assert joined_csv.is_file()
    assert group_stats_csv.is_file()
    assert any(figures_dir.glob("boxplot_*.png"))
    assert any(figures_dir.glob("bar_*.png"))
    assert (figures_dir / "scatter_novelty_vs_lag.png").is_file()
