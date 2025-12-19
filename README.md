# ssmproxy-eval
Evaluation scaffold for running self-similarity metrics over MIDI corpora.

## Installation
Install the project (and developer tools for linting/testing) with:

```bash
python -m pip install -e ".[dev]"
```

The package depends on the real `pretty_midi`/`mido`/`PyYAML` stacks to read and
write standard MIDI files.

## Quickstart

1. Install the package: `python -m pip install -e ".[dev]"`
2. Run the test suite to validate the environment: `pytest`
3. Inspect CLI options: `ssmproxy --help`
4. Run an evaluation: `ssmproxy eval /path/to/midi/dir --output outputs`

## Assumptions

- **Time signature:** Fixed 4/4 with `beats_per_bar=4`.
- **Grid:** `steps_per_bar=16` (16th-note grid), with note-on counts projected to this grid.
- **Tempo:** The leading tempo event is used; mid-piece tempo changes are not fully modeled.
- **Feature basis:** Bar features are derived from note-on events.
- **Percussion:** Drum tracks (`pretty_midi.Instrument.is_drum`) are excluded from features by default and can be re-enabled via CLI/config.

## Usage
Inspect the CLI and subcommands:

```bash
ssmproxy --help
ssmproxy eval /path/to/midi/dir --output outputs --run-id demo --lag-min-lag 4
```

Generate toy MIDI examples:

```bash
ssmproxy toy generate --out-dir toy-output --variants 2 --seed 0
```

Key evaluation options:

- `--lag-min-lag`: Minimum bar lag to consider (defaults to the config value, 4).
- `--lag-top-k`: Number of lag energies to aggregate.
- `--exclude-drums/--include-drums`: Toggle whether drum tracks contribute to features (defaults to exclude).

## Outputs

Each run produces artifacts under `outputs/<run_id>/`:

- `config.yaml`: Snapshot of the resolved run configuration (including lag and drum settings).
- `metrics.csv`: Per-piece metrics (see column descriptions below).
- `figures/ssm/`: Self-similarity matrices as PNGs.
- `figures/novelty/`: Novelty curves with detected peaks.
- `figures/lag/`: (Reserved) Lag visualizations when enabled.

### Metrics columns

| Column | Description |
| --- | --- |
| `piece_id` | Stem of the MIDI filename. |
| `num_bars` | Bars inferred from note-on positions. |
| `num_novelty_peaks` | Count of novelty curve peaks. |
| `novelty_peak_rate` | Peak count normalized by bar count. |
| `novelty_prom_mean` / `novelty_prom_median` | Mean/median peak prominence. |
| `novelty_interval_mean` / `novelty_interval_cv` | Mean and CV of inter-peak distances. |
| `lag_energy` | Sum of the top-`k` lag diagonal energies. |
| `lag_best` | Lag index with the strongest energy. |
| `lag_min_lag` | Minimum lag threshold applied during lag computation. |

## Run the tests

```bash
pytest
```
