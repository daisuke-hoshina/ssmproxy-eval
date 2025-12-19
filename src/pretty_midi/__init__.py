"""Lightweight stand-in for the ``pretty_midi`` package used in tests.

This module provides the minimal surface needed by the ssmproxy evaluation
pipeline without relying on native dependencies or internet access. It stores
MIDI-like data as JSON for round-tripping within tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class Note:
    velocity: int
    pitch: int
    start: float
    end: float


class Instrument:
    def __init__(self, program: int = 0, is_drum: bool = False):
        self.program = program
        self.is_drum = is_drum
        self.notes: List[Note] = []


class PrettyMIDI:
    def __init__(self, midi_file: str | None = None, *, initial_tempo: float = 120.0):
        self.initial_tempo = float(initial_tempo)
        self.instruments: List[Instrument] = []
        if midi_file:
            path = Path(midi_file)
            if path.is_file():
                self._load_json(path)

    def _load_json(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.initial_tempo = float(data.get("initial_tempo", self.initial_tempo))
        for inst_data in data.get("instruments", []):
            instrument = Instrument(
                program=int(inst_data.get("program", 0)),
                is_drum=bool(inst_data.get("is_drum", False)),
            )
            for note_data in inst_data.get("notes", []):
                instrument.notes.append(
                    Note(
                        velocity=int(note_data.get("velocity", 0)),
                        pitch=int(note_data.get("pitch", 0)),
                        start=float(note_data.get("start", 0.0)),
                        end=float(note_data.get("end", 0.0)),
                    )
                )
            self.instruments.append(instrument)

    def get_tempo_changes(self) -> Tuple[list[float], list[float]]:
        return [0.0], [self.initial_tempo]

    def write(self, filename: str | Path) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "initial_tempo": self.initial_tempo,
            "instruments": [
                {
                    "program": instrument.program,
                    "is_drum": instrument.is_drum,
                    "notes": [
                        {
                            "velocity": note.velocity,
                            "pitch": note.pitch,
                            "start": note.start,
                            "end": note.end,
                        }
                        for note in instrument.notes
                    ],
                }
                for instrument in self.instruments
            ],
        }
        path.write_text(json.dumps(data))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"PrettyMIDI(instruments={len(self.instruments)}, tempo={self.initial_tempo})"
