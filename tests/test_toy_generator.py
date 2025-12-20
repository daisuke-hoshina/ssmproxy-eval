import csv
from pathlib import Path

import pytest
import pretty_midi

from ssmproxy import toy_generator


def test_generate_corpus_creates_midi_and_manifest(tmp_path: Path) -> None:
    variants = 2
    pieces = toy_generator.generate_corpus(tmp_path, variants=variants, seed=123)

    expected_count = len(toy_generator._GENERATORS) * variants
    assert len(pieces) == expected_count

    manifest_path = tmp_path / "manifest.csv"
    assert manifest_path.is_file()

    with manifest_path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == expected_count

    for piece, row in zip(pieces, rows):
        # Default behavior: nested directories
        midi_path = tmp_path / piece.pattern / f"{piece.piece_id}.mid"
        assert midi_path.is_file()
        
        # Verify relative path in manifest matches structure and is POSIX
        expected_rel = f"{piece.pattern}/{piece.piece_id}.mid"
        assert row["relative_path"] == expected_rel
        assert piece.relative_path == expected_rel
        assert "\\" not in row["relative_path"]  # Strictly POSIX

        loaded = pretty_midi.PrettyMIDI(midi_path)
        _, tempi = loaded.get_tempo_changes()
        assert pytest.approx(float(tempi[0]), rel=1e-5) == toy_generator.BPM
        assert piece.piece_id == row["piece_id"]
        assert row["pattern"] in toy_generator._GENERATORS
        assert int(row["variant"]) == piece.variant
        assert int(row["seed"]) == piece.seed
        
        # New columns
        assert int(row["bars"]) == toy_generator.BARS_PER_PIECE
        assert float(row["bpm"]) == toy_generator.BPM

        max_time = max((note.end for inst in loaded.instruments for note in inst.notes), default=0.0)
        total_duration = toy_generator.BARS_PER_PIECE * toy_generator.BEATS_PER_BAR * toy_generator.SECONDS_PER_BEAT
        assert 0.0 < max_time <= total_duration


def test_generate_corpus_flat_structure(tmp_path: Path) -> None:
    variants = 1
    pieces = toy_generator.generate_corpus(tmp_path, variants=variants, seed=456, flat=True)

    manifest_path = tmp_path / "manifest.csv"
    assert manifest_path.is_file()

    with manifest_path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    
    for piece, row in zip(pieces, rows):
        # Flat behavior: root directory
        midi_path = tmp_path / f"{piece.piece_id}.mid"
        assert midi_path.is_file()

        # Verify relative path is just filename
        expected_rel = f"{piece.piece_id}.mid"
        assert row["relative_path"] == expected_rel
        assert piece.relative_path == expected_rel
