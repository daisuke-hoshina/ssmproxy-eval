import csv
from pathlib import Path

import pretty_midi
from typer.testing import CliRunner

from ssmproxy.cli import app


def _write_grouped_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)

    seconds_per_beat = 60.0 / 120.0
    for idx, pitch in enumerate(pitches):
        start = idx * seconds_per_beat
        end = start + seconds_per_beat * 0.75
        instrument.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))

    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def test_report_run_end_to_end(tmp_path: Path) -> None:
    midi_root = tmp_path / "midi"
    repeat_dir = midi_root / "repeat"
    random_dir = midi_root / "random"

    _write_grouped_midi(repeat_dir / "a.mid", [60, 60, 67, 67])
    _write_grouped_midi(random_dir / "b.mid", [60, 62, 64, 65])

    output_root = tmp_path / "outputs"
    run_id = "report-e2e"
    runner = CliRunner()
    eval_result = runner.invoke(
        app,
        [
            "eval",
            str(midi_root),
            "--output",
            str(output_root),
            "--run-id",
            run_id,
            "--novelty-L",
            "2",
            "--lag-top-k",
            "2",
        ],
    )

    assert eval_result.exit_code == 0, eval_result.output

    run_dir = output_root / run_id
    metrics_candidates = [run_dir / "metrics.csv", run_dir / "metrics" / "ssm_proxy.csv"]
    metrics_path = next((path for path in metrics_candidates if path.is_file()), None)
    assert metrics_path is not None

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

    summary_dir = run_dir / "summary"
    joined_csv = summary_dir / "metrics_joined.csv"
    group_stats_csv = summary_dir / "metrics_group_stats.csv"
    figures_dir = summary_dir / "figures"

    assert joined_csv.is_file()
    assert group_stats_csv.is_file()
    assert any(figures_dir.glob("*.png"))

    with group_stats_csv.open(newline="") as fp:
        stats_rows = list(csv.DictReader(fp))
    groups = {row.get("group", "") for row in stats_rows}
    assert {"repeat", "random"}.issubset(groups)
