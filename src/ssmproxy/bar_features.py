"""Bar-aligned feature extraction utilities."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import pretty_midi

from .midi_io import extract_note_on_events, load_midi

BEATS_PER_BAR = 4
STEPS_PER_BAR = 16
STEPS_PER_BEAT = STEPS_PER_BAR // BEATS_PER_BAR


def _ensure_piece_id(piece_id: Optional[str]) -> str:
    return piece_id if piece_id else "unknown"


def _allocate_bar_arrays(num_bars: int) -> Tuple[List[List[float]], List[List[float]]]:
    return ([ [0.0] * 12 for _ in range(num_bars) ], [ [0.0] * STEPS_PER_BAR for _ in range(num_bars) ])


def compute_bar_features(
    midi: pretty_midi.PrettyMIDI, piece_id: Optional[str] = None, *, exclude_drums: bool = True
) -> tuple[str, List[List[float]], List[List[float]]]:
    """Compute bar-wise pitch class and onset histograms.

    Args:
        midi: Loaded PrettyMIDI object.
        piece_id: Optional identifier for the piece.

    Args:
        exclude_drums: When True, skip percussion instruments flagged with
            ``is_drum``.

    Returns:
        A tuple of (piece_id, PCH array [bars, 12], ONH array [bars, 16]).
    """

    tempos_times, tempos = midi.get_tempo_changes()
    tempo = float(tempos[0]) if len(tempos) else 120.0
    if tempo <= 0:
        tempo = 120.0

    seconds_per_beat = 60.0 / tempo
    seconds_per_step = seconds_per_beat / STEPS_PER_BEAT

    note_on_events = extract_note_on_events(midi, exclude_drums=exclude_drums)
    if not note_on_events:
        return _ensure_piece_id(piece_id), [], []

    # Determine number of bars based on the latest note onset.
    last_start = max(event[0] for event in note_on_events)
    last_beat_index = int(math.floor(last_start / seconds_per_beat))
    num_bars = last_beat_index // BEATS_PER_BAR + 1

    pch, onh = _allocate_bar_arrays(num_bars)

    for start, pitch in note_on_events:
        beat_float = start / seconds_per_beat
        beat_index = int(math.floor(beat_float))
        bar_index = beat_index // BEATS_PER_BAR

        if bar_index >= num_bars:
            # Expand arrays if calculation underestimated number of bars.
            extra_pch, extra_onh = _allocate_bar_arrays(bar_index - num_bars + 1)
            pch.extend(extra_pch)
            onh.extend(extra_onh)
            num_bars = len(pch)

        pitch_class = pitch % 12
        pch[bar_index][pitch_class] += 1.0

        step_within_beat = int(math.floor((start - beat_index * seconds_per_beat) / seconds_per_step))
        step_within_beat = min(step_within_beat, STEPS_PER_BEAT - 1)
        step_index = (beat_index % BEATS_PER_BAR) * STEPS_PER_BEAT + step_within_beat
        onh[bar_index][step_index] = 1.0

    # L1-normalize per bar.
    for bar in range(num_bars):
        p_sum = sum(pch[bar])
        if p_sum:
            pch[bar] = [value / p_sum for value in pch[bar]]

        o_sum = sum(onh[bar])
        if o_sum:
            onh[bar] = [value / o_sum for value in onh[bar]]

    return _ensure_piece_id(piece_id), pch, onh


def compute_bar_features_from_path(
    path: str | bytes | "PathLike[str]", *, exclude_drums: bool = True
) -> tuple[str, List[List[float]], List[List[float]]]:
    """Load a MIDI file and compute bar features."""

    piece_id, midi = load_midi(path)
    return compute_bar_features(midi, piece_id, exclude_drums=exclude_drums)
