import math

import pretty_midi

from ssmproxy.bar_features import compute_bar_features
from ssmproxy.midi_io import extract_note_on_events, load_midi


def _make_synthetic_midi() -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)

    # Bar 0 events (beats 0-3).
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.25))  # C4, step 0
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.3, end=0.55))  # E4, step 2
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=0.6, end=0.85))  # C5, step 4

    # Bar 1 event (beats 4-7).
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=67, start=2.1, end=2.35))  # G4, step 0 of bar 1

    midi.instruments.append(instrument)
    return midi


def test_extract_note_on_events(tmp_path):
    midi = _make_synthetic_midi()
    midi_path = tmp_path / "test_piece.mid"
    midi.write(str(midi_path))

    piece_id, loaded = load_midi(midi_path)
    assert piece_id == "test_piece"

    events = extract_note_on_events(loaded)
    assert [pitch for _, pitch in events] == [60, 64, 72, 67]

    times = [t for t, _ in events]
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))


def test_compute_bar_features_bins_and_normalization():
    midi = _make_synthetic_midi()
    piece_id, pch, onh = compute_bar_features(midi, piece_id="synthetic")

    assert piece_id == "synthetic"
    assert len(pch) == 2
    assert len(onh) == 2
    assert len(pch[0]) == 12
    assert len(onh[0]) == 16

    # Bar 0: pitch classes C and E with C appearing twice.
    assert math.isclose(pch[0][0], 2 / 3)
    assert math.isclose(pch[0][4], 1 / 3)
    assert math.isclose(sum(pch[0]), 1.0)

    # Bar 1: single G note.
    assert math.isclose(pch[1][7], 1.0)

    # Onset steps should be normalized and only mark steps with notes.
    expected_onh_bar0 = [0.0] * 16
    expected_onh_bar0[0] = expected_onh_bar0[2] = expected_onh_bar0[4] = 1 / 3
    assert all(math.isclose(o, e) for o, e in zip(onh[0], expected_onh_bar0))
    assert math.isclose(sum(onh[0]), 1.0)

    expected_onh_bar1 = [0.0] * 16
    expected_onh_bar1[0] = 1.0
    assert all(math.isclose(o, e) for o, e in zip(onh[1], expected_onh_bar1))
