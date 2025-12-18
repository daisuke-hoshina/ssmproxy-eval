"""A lightweight fallback implementation of the :mod:`pretty_midi` API.

This stub is intentionally minimal and only implements the pieces of the
``pretty_midi`` interface required by the local feature extraction utilities
and tests. It supports programmatic construction of notes/instruments and a
lossless JSON-based serialization for round-tripping within the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


class Note:
    def __init__(self, velocity: int, pitch: int, start: float, end: float) -> None:
        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end


class Instrument:
    def __init__(self, program: int, is_drum: bool = False) -> None:
        self.program = program
        self.is_drum = is_drum
        self.notes: List[Note] = []


class PrettyMIDI:
    def __init__(self, midi_file: str | None = None, initial_tempo: float = 120.0) -> None:
        self.initial_tempo = initial_tempo
        self.instruments: List[Instrument] = []
        self._tempo_changes = ([0.0], [float(initial_tempo)])

        if midi_file:
            self._load(midi_file)

    def get_tempo_changes(self) -> Tuple[List[float], List[float]]:
        times, tempi = self._tempo_changes
        return times.copy(), tempi.copy()

    def write(self, file_path: str | bytes | "PathLike[str]") -> None:
        path = Path(file_path)
        payload = {
            "initial_tempo": self.initial_tempo,
            "instruments": [
                {
                    "program": inst.program,
                    "is_drum": inst.is_drum,
                    "notes": [
                        {
                            "velocity": note.velocity,
                            "pitch": note.pitch,
                            "start": note.start,
                            "end": note.end,
                        }
                        for note in inst.notes
                    ],
                }
                for inst in self.instruments
            ],
        }
        path.write_text(json.dumps(payload))

    def _load(self, file_path: str | bytes | "PathLike[str]") -> None:
        path = Path(file_path)
        payload = json.loads(path.read_text())
        self.initial_tempo = float(payload.get("initial_tempo", 120.0))
        self._tempo_changes = ([0.0], [self.initial_tempo])
        self.instruments = []
        for inst_data in payload.get("instruments", []):
            inst = Instrument(program=int(inst_data.get("program", 0)), is_drum=bool(inst_data.get("is_drum", False)))
            for note_data in inst_data.get("notes", []):
                note = Note(
                    velocity=int(note_data["velocity"]),
                    pitch=int(note_data["pitch"]),
                    start=float(note_data["start"]),
                    end=float(note_data["end"]),
                )
                inst.notes.append(note)
            self.instruments.append(inst)
