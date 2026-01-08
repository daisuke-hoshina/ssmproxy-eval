"""MIDI loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pretty_midi


NoteOnEvent = Tuple[float, int]


def load_midi(path: Path | str) -> tuple[str, pretty_midi.PrettyMIDI]:
    """Load a MIDI file and return its identifier and PrettyMIDI object."""

    midi_path = Path(path)
    piece_id = midi_path.stem
    return piece_id, pretty_midi.PrettyMIDI(str(midi_path))


def extract_note_on_events(midi: pretty_midi.PrettyMIDI, *, exclude_drums: bool = True) -> List[NoteOnEvent]:
    """Return a sorted list of (start_time_seconds, pitch) tuples from a MIDI object."""

    note_ons: List[NoteOnEvent] = []
    for instrument in midi.instruments:
        if exclude_drums and instrument.is_drum:
            continue

        for note in instrument.notes:
            note_ons.append((note.start, note.pitch))

    note_ons.sort(key=lambda event: event[0])
    return note_ons


NoteEvent = Tuple[float, float, int, int]


def extract_note_events(midi: pretty_midi.PrettyMIDI, *, exclude_drums: bool = True) -> List[NoteEvent]:
    """Return a sorted list of (start, end, pitch, velocity) tuples."""
    notes: List[NoteEvent] = []
    for instrument in midi.instruments:
        if exclude_drums and instrument.is_drum:
            continue

        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))

    notes.sort(key=lambda event: event[0])
    return notes
