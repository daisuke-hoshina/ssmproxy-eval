"""Bar-aligned feature extraction utilities."""

from __future__ import annotations

import bisect
import math
from typing import List, Optional, Tuple, Dict, Any

import pretty_midi

from .midi_io import extract_note_on_events, load_midi, extract_note_events

BEATS_PER_BAR = 4
STEPS_PER_BAR = 16
STEPS_PER_BEAT = STEPS_PER_BAR // BEATS_PER_BAR


def _ensure_piece_id(piece_id: Optional[str]) -> str:
    return piece_id if piece_id else "unknown"


def get_beat_times(midi: pretty_midi.PrettyMIDI, target_end_time: float | None = None) -> List[float]:
    """Get beat times from MIDI, protecting against empty beats and ensuring coverage."""
    beats = list(midi.get_beats())
    
    # Fallback if no beats
    if not beats:
        tempos_times, tempos = midi.get_tempo_changes()
        tempo = tempos[0] if len(tempos) > 0 else 120.0
        if tempo <= 0: tempo = 120.0
        spb = 60.0 / tempo
        beats = [0.0, spb] # Initial beats
        
    # Determine last interval for extrapolation
    if len(beats) > 1:
        last_interval = beats[-1] - beats[-2]
    else:
        last_interval = 0.5 

    # Ensure coverage of target_end_time
    if target_end_time is not None:
        if beats[-1] < target_end_time:
            # Extrapolate
            current_last = beats[-1]
            if last_interval > 0:
                needed = target_end_time - current_last
                steps = math.ceil(needed / last_interval)
                for _ in range(steps):
                    current_last += last_interval
                    beats.append(current_last)
                    if current_last >= target_end_time:
                        break
            else:
                 beats.append(target_end_time + 1.0)
    else:
        # Legacy/Default behavior: append one beat
        beats.append(beats[-1] + last_interval)

    return beats


def time_to_beat_index(t: float, beats: List[float]) -> int:
    """Map time to beat index 0..len-2."""
    idx = bisect.bisect_right(beats, t) - 1
    if idx < 0:
        return 0
    if idx >= len(beats) - 1:
        return len(beats) - 2
    return idx


def time_to_beat_index_end(t: float, beats: List[float]) -> int:
    """Map time to beat index using end-exclusive logic (start <= t < end).
    
    If t falls exactly on a beat boundary, it belongs to the PREVIOUS beat.
    """
    # bisect_left returns insertion point left of equal elements.
    # If t matches beats[i], returns i. We want i-1.
    # If t is between beats[i] and beats[i+1], returns i+1. We want i.
    # So idx - 1 is generally correct.
    idx = bisect.bisect_left(beats, t) - 1
    if idx < 0:
        return 0
    if idx >= len(beats) - 1:
        return len(beats) - 2
    return idx


def _allocate_bar_arrays(num_bars: int) -> Tuple[List[List[float]], List[List[float]]]:
    return ([ [0.0] * 12 for _ in range(num_bars) ], [ [0.0] * STEPS_PER_BAR for _ in range(num_bars) ])


def compute_bar_features(
    midi: pretty_midi.PrettyMIDI,
    piece_id: Optional[str] = None,
    *,
    exclude_drums: bool = True,
    max_bars: Optional[int] = None,
    quantize_mode: str = "beat_grid",
    analysis_beats_per_bar: int = 4,
    steps_per_beat: int = 4,
    feature_mode: str = "enhanced",
) -> tuple[str, Dict[str, Any]]:
    """Compute bar-wise features.

    Args:
        midi: Loaded PrettyMIDI object.
        piece_id: Optional identifier for the piece.
        exclude_drums: When True, skip percussion instruments.
        max_bars: Maximum number of bars to process.
        quantize_mode: "beat_grid" or "legacy_fixed_tempo".
        analysis_beats_per_bar: Number of beats per bar (for beat_grid).
        steps_per_beat: Number of steps per beat.
        feature_mode: "enhanced" or "basic".

    Returns:
        A tuple of (piece_id, features_dict).
        features_dict contains "pch" and "onh" (legacy keys) and others.
    """
    pid = _ensure_piece_id(piece_id)
    
    if quantize_mode == "legacy_fixed_tempo":
        # Legacy logic wrapper
        _, pch, onh = _compute_bar_features_legacy(midi, pid, exclude_drums=exclude_drums, max_bars=max_bars)
        return pid, {"pch": pch, "onh": onh}

    # Beat Grid Logic
    notes = extract_note_events(midi, exclude_drums=exclude_drums)
    
    if not notes:
        # Task 2: Return complete keys for strict SSM safety
        return pid, {
            "pch": [], 
            "onh": [], 
            "onh_bin": [], 
            "onh_count": [], 
            "density": [],
            "chroma_roll": [] 
        }

    # Determine num_bars
    # Ensure beats cover the very last note end (important for chroma roll overlap)
    last_end = max(n[1] for n in notes)
    beats = get_beat_times(midi, target_end_time=last_end)
    
    # Use end-exclusive logic for bar counting
    # This prevents an extra bar if the note ends exactly on measure boundary
    last_beat_idx = time_to_beat_index_end(last_end, beats)
    num_bars = (last_beat_idx // analysis_beats_per_bar) + 1
    
    if max_bars is not None:
        num_bars = min(num_bars, max_bars)
        
    steps_per_bar = analysis_beats_per_bar * steps_per_beat
    
    # Allocations
    # pch: [bars, 12]
    # onh_bin: [bars, steps_per_bar] (legacy onh)
    # onh_count: [bars, steps_per_bar]
    # density: [bars]
    # chroma_roll: [bars, 12 * steps_per_bar]
    
    pch = [[0.0] * 12 for _ in range(num_bars)]
    onh_bin = [[0.0] * steps_per_bar for _ in range(num_bars)]
    onh_count = [[0.0] * steps_per_bar for _ in range(num_bars)]
    density = [0.0] * num_bars
    chroma_roll = [[0.0] * (12 * steps_per_bar) for _ in range(num_bars)]
    
    enhanced = (feature_mode == "enhanced")
    
    for start, end, pitch, velocity in notes:
        # 1. Map Start to Beat/Step
        start_beat_idx = time_to_beat_index(start, beats)
        bar_idx = start_beat_idx // analysis_beats_per_bar
        
        if bar_idx >= num_bars:
            continue
            
        # PCH (Onset based)
        pc = pitch % 12
        pch[bar_idx][pc] += 1.0
        
        # ONH / Step calculation
        # Fraction within beat
        beat_start_t = beats[start_beat_idx]
        beat_end_t = beats[start_beat_idx + 1]
        beat_dur = max(beat_end_t - beat_start_t, 1e-6)
        
        frac = (start - beat_start_t) / beat_dur
        step_within_beat = int(math.floor(frac * steps_per_beat))
        step_within_beat = max(0, min(step_within_beat, steps_per_beat - 1))
        
        step_in_bar = (start_beat_idx % analysis_beats_per_bar) * steps_per_beat + step_within_beat
        
        onh_bin[bar_idx][step_in_bar] = 1.0
        onh_count[bar_idx][step_in_bar] += 1.0
        density[bar_idx] += 1.0
        
        if enhanced:
            # Chroma Roll (Duration based with exact overlap)
            # Iterate through covered steps
            note_end = min(end, beats[-1]) # Clip to covered range
            if note_end <= start:
                continue
                
            # Find step range
            # Global step: bar_idx * steps_per_bar + step_in_bar
            # Calculate start/end step indices
            start_global_step = start_beat_idx * steps_per_beat + step_within_beat
            
            # For end step, we need to be careful about strict upper bound.
            # We can iterate from start step until step's start time >= note_end
            
            current_global_step = start_global_step
            
            while True:
                # Compute current step time range
                # beat index = step // steps_per_beat
                # step within beat = step % steps_per_beat
                s_beat_idx = current_global_step // steps_per_beat
                s_within = current_global_step % steps_per_beat
                
                if s_beat_idx >= len(beats) - 1:
                    break
                    
                # Step time range
                b_start = beats[s_beat_idx]
                b_end = beats[s_beat_idx+1]
                b_dur = b_end - b_start
                if b_dur <= 0: # Should not happen with validated beats
                    current_global_step += 1
                    continue
                    
                step_dur = b_dur / steps_per_beat
                step_t0 = b_start + (s_within * step_dur)
                step_t1 = step_t0 + step_dur
                
                # Overlap
                overlap = max(0.0, min(note_end, step_t1) - max(start, step_t0))
                
                if overlap > 0:
                    # Add to matrix
                    # Map global step to bar/local
                    c_bar = current_global_step // steps_per_bar
                    c_step = current_global_step % steps_per_bar
                    
                    if c_bar < num_bars:
                        idx = (pc * steps_per_bar) + c_step
                        chroma_roll[c_bar][idx] += overlap
                
                # Check termination
                if step_t1 >= note_end:
                    break
                    
                current_global_step += 1

    # Normalization
    for bar in range(num_bars):
        # PCH
        s = sum(pch[bar])
        if s > 0:
             pch[bar] = [v / s for v in pch[bar]]
        
        # ONH Bin (Legacy: sum to 1? Or binary?)
        # Legacy: L1 normalized per bar.
        s_onh = sum(onh_bin[bar])
        if s_onh > 0:
            onh_bin[bar] = [v / s_onh for v in onh_bin[bar]]
            
        # ONH Count: L1 normalize
        s_cnt = sum(onh_count[bar])
        if s_cnt > 0:
            onh_count[bar] = [v / s_cnt for v in onh_count[bar]]
            
        # Chroma Roll: L1 normalize
        if enhanced:
            s_cr = sum(chroma_roll[bar])
            if s_cr > 0:
                chroma_roll[bar] = [v / s_cr for v in chroma_roll[bar]]
    
    # Cast density to list of vectors [ [log1p(d), 1.0] ]
    # This makes cosine similarity sensitive to magnitude differences.
    # log1p is used to dampen extreme values.
    density_vec = [[math.log1p(d), 1.0] for d in density]

    features = {
        "pch": pch,
        "onh": onh_bin, # Legacy name
        "onh_bin": onh_bin,
        "onh_count": onh_count,
        "density": density_vec,
        "chroma_roll": chroma_roll,
    }
    
    return pid, features


def _compute_bar_features_legacy(
    midi: pretty_midi.PrettyMIDI,
    piece_id: Optional[str] = None,
    *,
    exclude_drums: bool = True,
    max_bars: Optional[int] = None,
) -> tuple[str, List[List[float]], List[List[float]]]:
    """Legacy implementation of compute_bar_features."""
    # (Copied existing logic)
    tempos_times, tempos = midi.get_tempo_changes()
    tempo = float(tempos[0]) if len(tempos) else 120.0
    if tempo <= 0:
        tempo = 120.0

    seconds_per_beat = 60.0 / tempo
    seconds_per_step = seconds_per_beat / STEPS_PER_BEAT

    note_on_events = extract_note_on_events(midi, exclude_drums=exclude_drums)
    if not note_on_events:
        return _ensure_piece_id(piece_id), [], []

    last_start = max(event[0] for event in note_on_events)
    last_beat_index = int(math.floor(last_start / seconds_per_beat))
    num_bars = last_beat_index // BEATS_PER_BAR + 1
    if max_bars is not None:
        num_bars = min(num_bars, max_bars)

    pch, onh = _allocate_bar_arrays(num_bars)

    for start, pitch in note_on_events:
        beat_float = start / seconds_per_beat
        beat_index = int(math.floor(beat_float))
        bar_index = beat_index // BEATS_PER_BAR
        if bar_index >= num_bars:
            continue

        pitch_class = pitch % 12
        pch[bar_index][pitch_class] += 1.0

        step_within_beat = int(math.floor((start - beat_index * seconds_per_beat) / seconds_per_step))
        step_within_beat = min(step_within_beat, STEPS_PER_BEAT - 1)
        step_index = (beat_index % BEATS_PER_BAR) * STEPS_PER_BEAT + step_within_beat
        onh[bar_index][step_index] = 1.0

    for bar in range(num_bars):
        p_sum = sum(pch[bar])
        if p_sum:
            pch[bar] = [value / p_sum for value in pch[bar]]

        o_sum = sum(onh[bar])
        if o_sum:
            onh[bar] = [value / o_sum for value in onh[bar]]

    return _ensure_piece_id(piece_id), pch, onh


def compute_bar_features_from_path(
    path: str | bytes | "PathLike[str]",
    *,
    exclude_drums: bool = True,
    max_bars: Optional[int] = None,
    quantize_mode: str = "beat_grid",
    analysis_beats_per_bar: int = 4,
    steps_per_beat: int = 4,
    feature_mode: str = "enhanced",
) -> tuple[str, Dict[str, Any]]:
    """Load a MIDI file and compute bar features."""
    piece_id, midi = load_midi(path)
    return compute_bar_features(
        midi, 
        piece_id, 
        exclude_drums=exclude_drums, 
        max_bars=max_bars,
        quantize_mode=quantize_mode,
        analysis_beats_per_bar=analysis_beats_per_bar,
        steps_per_beat=steps_per_beat,
        feature_mode=feature_mode
    )
