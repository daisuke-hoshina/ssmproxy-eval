"""Utilities for generating toy MIDI corpora."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pretty_midi

BPM = 120.0
BEATS_PER_BAR = 4
BARS_PER_PIECE = 32
SECONDS_PER_BEAT = 60.0 / BPM


@dataclass
class ToyPiece:
    piece_id: str
    pattern: str
    variant: int
    seed: int
    midi: pretty_midi.PrettyMIDI
    relative_path: str = ""
    bars: int = BARS_PER_PIECE
    bpm: float = BPM


NoteTemplate = tuple[float, float, int]
BarTemplate = list[NoteTemplate]


def _note_time(beat: float) -> float:
    return beat * SECONDS_PER_BEAT


def _build_bar_template(rng: random.Random, base_pitch: int, pattern: str = "repeat") -> BarTemplate:
    offsets = [-5, -2, 0, 2, 4, 7]
    notes: BarTemplate = []
    
    # Rhythm implementation:
    # repeat/default: [0.0, 2.0]
    # random: random subset of 16th notes
    # AABA/ABAB: distinct rhythmic motifs for sections (handled by caller passing different base params, but here we can toggle logic)
    # partial_copy: standard
    # hierarchical: standard
    
    if pattern == "random":
        # Select 2-4 positions from 16th beat grid
        grid = [i * 0.25 for i in range(int(BEATS_PER_BAR * 4))]
        num_notes = rng.randint(2, 4)
        beats = sorted(rng.sample(grid, num_notes))
    elif pattern == "hierarchical":
        # Use off-beats for differentiation
        beats = [0.0, 1.5, 3.0]
    else:
        # Standard distinct rhythm for basic patterns could be varied by variant, but here we keep simple defaults
        # To differentiate groups by ONH, we need distinct onset patterns.
        # Let's map patterns to static distinct rhythms for consistency within group.
        if pattern == "AABA":
             beats = [0.0, 1.0, 2.0, 3.0] # Quarter notes
        elif pattern == "ABAB":
             beats = [0.5, 1.5, 2.5, 3.5] # Off-beats
        else:
             beats = [0.0, 2.0] # Half notes (legacy default)

    for beat in beats:
        pitch = base_pitch + rng.choice(offsets)
        notes.append((beat, 0.25, pitch)) # Shorten duration to avoid overlap
    return notes


def _build_phrase_template(rng: random.Random, bars: int, base_pitch: int, pattern: str = "repeat") -> list[BarTemplate]:
    return [_build_bar_template(rng, base_pitch, pattern=pattern) for _ in range(bars)]


def _write_phrase(instrument: pretty_midi.Instrument, template: Sequence[BarTemplate], start_bar: int) -> None:
    for bar_index, bar in enumerate(template):
        base_beat = (start_bar + bar_index) * BEATS_PER_BAR
        for beat_offset, duration_beats, pitch in bar:
            start = _note_time(base_beat + beat_offset)
            end = _note_time(base_beat + beat_offset + duration_beats)
            instrument.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))


def _mutate_phrase(template: Sequence[BarTemplate], rng: random.Random, change_prob: float = 0.25) -> list[BarTemplate]:
    mutated: list[BarTemplate] = []
    for bar in template:
        if rng.random() < change_prob:
            base_pitch = rng.choice([58, 60, 62, 65, 67])
            mutated.append(_build_bar_template(rng, base_pitch, pattern="partial_copy"))
        else:
            mutated.append([note for note in bar])
    return mutated


def _generate_repeat(rng: random.Random) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    instrument = pretty_midi.Instrument(program=0)
    phrase = _build_phrase_template(rng, bars=4, base_pitch=60, pattern="repeat")
    for repeat in range(0, BARS_PER_PIECE, 4):
        _write_phrase(instrument, phrase, repeat)
    midi.instruments.append(instrument)
    return midi


def _generate_random(rng: random.Random) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    instrument = pretty_midi.Instrument(program=0)
    for bar in range(BARS_PER_PIECE):
        base_pitch = rng.choice([55, 58, 60, 62, 65, 67, 69])
        _write_phrase(instrument, [_build_bar_template(rng, base_pitch, pattern="random")], bar)
    midi.instruments.append(instrument)
    return midi


def _generate_aaba(rng: random.Random) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    instrument = pretty_midi.Instrument(program=0)
    phrase_a = _build_phrase_template(rng, bars=8, base_pitch=60, pattern="AABA")
    phrase_b = _build_phrase_template(rng, bars=8, base_pitch=67, pattern="AABA")
    sections = [phrase_a, phrase_a, phrase_b, phrase_a]
    for idx, phrase in enumerate(sections):
        _write_phrase(instrument, phrase, idx * 8)
    midi.instruments.append(instrument)
    return midi


def _generate_abab(rng: random.Random) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    instrument = pretty_midi.Instrument(program=0)
    instrument = pretty_midi.Instrument(program=0)
    phrase_a = _build_phrase_template(rng, bars=8, base_pitch=62, pattern="ABAB")
    phrase_b = _build_phrase_template(rng, bars=8, base_pitch=69, pattern="ABAB")
    sections = [phrase_a, phrase_b, phrase_a, phrase_b]
    for idx, phrase in enumerate(sections):
        _write_phrase(instrument, phrase, idx * 8)
    midi.instruments.append(instrument)
    return midi


def _generate_partial_copy(rng: random.Random) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    instrument = pretty_midi.Instrument(program=0)
    instrument = pretty_midi.Instrument(program=0)
    base_phrase = _build_phrase_template(rng, bars=8, base_pitch=64, pattern="partial_copy")
    altered = _mutate_phrase(base_phrase, rng, change_prob=0.35)
    sections = [base_phrase, altered, base_phrase, altered]
    for idx, phrase in enumerate(sections):
        _write_phrase(instrument, phrase, idx * 8)
    midi.instruments.append(instrument)
    return midi


def _generate_hierarchical(rng: random.Random) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    instrument = pretty_midi.Instrument(program=0)
    instrument = pretty_midi.Instrument(program=0)
    motif = _build_phrase_template(rng, bars=2, base_pitch=60, pattern="hierarchical")
    phrase: list[BarTemplate] = []
    for i in range(4):
        shift = rng.choice([0, 2, 5]) if i % 2 else 0
        shifted: list[BarTemplate] = []
        for bar in motif:
            shifted.append([(beat, dur, pitch + shift) for beat, dur, pitch in bar])
        phrase.extend(shifted)
    for idx in range(0, BARS_PER_PIECE, len(phrase)):
        _write_phrase(instrument, phrase, idx)
    midi.instruments.append(instrument)
    return midi


_GENERATORS = {
    "repeat": _generate_repeat,
    "random": _generate_random,
    "AABA": _generate_aaba,
    "ABAB": _generate_abab,
    "partial_copy": _generate_partial_copy,
    "hierarchical": _generate_hierarchical,
}


def generate_piece(pattern: str, variant: int, seed: int) -> ToyPiece:
    if pattern not in _GENERATORS:
        raise ValueError(f"Unknown pattern: {pattern}")
    rng = random.Random(seed)
    midi = _GENERATORS[pattern](rng)
    piece_id = f"{pattern}_{variant:03d}"
    return ToyPiece(
        piece_id=piece_id,
        pattern=pattern,
        variant=variant,
        seed=seed,
        midi=midi,
        bars=BARS_PER_PIECE,
        bpm=BPM,
    )


def save_manifest(rows: Iterable[ToyPiece], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["piece_id", "pattern", "variant", "seed", "relative_path", "bars", "bpm"])
        for piece in rows:
            writer.writerow(
                [
                    piece.piece_id,
                    piece.pattern,
                    piece.variant,
                    piece.seed,
                    piece.relative_path,
                    piece.bars,
                    piece.bpm,
                ]
            )


def generate_corpus(output_dir: Path, variants: int = 1, seed: int = 0, flat: bool = False) -> list[ToyPiece]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pieces: list[ToyPiece] = []
    patterns = list(_GENERATORS.keys())
    for pattern in patterns:
        for variant in range(variants):
            piece_seed = rng.randint(0, 1_000_000)
            piece = generate_piece(pattern, variant, piece_seed)

            if flat:
                filename = f"{piece.piece_id}.mid"
                output_path = output_dir / filename
                piece.relative_path = filename
            else:
                group_dir = output_dir / pattern
                group_dir.mkdir(exist_ok=True)
                filename = f"{piece.piece_id}.mid"
                output_path = group_dir / filename
                # Ensure POSIX-style path for manifest compatibility
                piece.relative_path = f"{pattern}/{filename}"

            piece.midi.write(str(output_path))
            pieces.append(piece)
    save_manifest(pieces, output_dir / "manifest.csv")
    return pieces
