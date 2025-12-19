# ssmproxy-eval
Evaluation scaffold for running self-similarity metrics over MIDI corpora.

## Installation
Install the project (and developer tools for linting/testing) with:

```bash
python -m pip install -e ".[dev]"
```

The package depends on the real `pretty_midi`/`mido`/`PyYAML` stacks to read and
write standard MIDI files.

## Usage
Inspect the CLI and subcommands:

```bash
ssmproxy --help
ssmproxy eval /path/to/midi/dir --output outputs --run-id demo
```

Generate toy MIDI examples:

```bash
ssmproxy toy generate --out-dir toy-output --variants 2 --seed 0
```

Run the tests:

```bash
pytest
```
