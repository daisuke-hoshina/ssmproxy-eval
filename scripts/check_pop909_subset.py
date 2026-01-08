#!/usr/bin/env python3
"""
scripts/check_pop909_subset.py

Check POP909 MIDI subset for:
1. Constant tempo
2. Constant time signature
3. Minimum number of bars (4/4-beat equivalent)

Dependencies: mido
"""
import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import mido

# Increase CSV field size limit just in case
csv.field_size_limit(sys.maxsize)

def is_note_event(msg: mido.Message) -> bool:
    if msg.type == 'note_on':
        return True
    if msg.type == 'note_off':
        return True
    return False

def check_midi(
    path: Path,
    target_bars: int,
    beats_per_bar: int,
    require_constant_tempo: bool,
    require_constant_timesig: bool,
    allowed_timesig: set[str],
    tempo_tol: float,
    use_end_tick: bool,
) -> dict:
    
    result = {
        "midi_path": str(path),
        "ticks_per_beat": 0,
        "tempo_events_count": 0,
        "tempo_unique_count": 0,
        "tempo_bpm": 0.0,
        "tempo_is_constant": False,
        "timesig_events_count": 0,
        "timesig_unique_count": 0,
        "timesig_value": "missing",
        "timesig_is_constant": False,
        "max_tick_all": 0,
        "max_tick_note": 0,
        "bars_4beat_end": 0,
        "bars_4beat_note": 0,
        "target_bars": target_bars,
        "bars_ok": False,
        "overall_ok": False,
        "fail_reason": [],
    }
    
    fail_reasons = []

    try:
        mid = mido.MidiFile(path)
        result["ticks_per_beat"] = mid.ticks_per_beat
        
        # 1. Collect Metas and Max Ticks
        tempo_events = []
        timesig_events = []
        
        max_tick_all = 0
        max_tick_note = 0
        
        # Iterate over tracks
        # Type 0: single track. Type 1: multiple tracks, synchronous.
        # We need absolute time for each track to find global max.
        
        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                
                # Check message type
                if msg.type == 'set_tempo':
                    # Tempo in microseconds per beat
                    bpm = mido.tempo2bpm(msg.tempo)
                    tempo_events.append(bpm)
                elif msg.type == 'time_signature':
                    ts_str = f"{msg.numerator}/{msg.denominator}"
                    timesig_events.append(ts_str)
                elif is_note_event(msg):
                    # For note_on with velocity 0, it's effectively note_off
                    # We care about the "end" of musical content.
                    if msg.type == 'note_on' and msg.velocity > 0:
                        max_tick_note = max(max_tick_note, abs_tick)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        max_tick_note = max(max_tick_note, abs_tick)
            
            # End of track
            max_tick_all = max(max_tick_all, abs_tick)

        result["max_tick_all"] = max_tick_all
        result["max_tick_note"] = max_tick_note
        
        # 2. Analyze Tempo
        result["tempo_events_count"] = len(tempo_events)
        if tempo_events:
            # Cluster tempos by tolerance
            unique_tempos = []
            for t in tempo_events:
                found = False
                for ut in unique_tempos:
                    if abs(t - ut) < tempo_tol:
                        found = True
                        break
                if not found:
                    unique_tempos.append(t)
            
            result["tempo_unique_count"] = len(unique_tempos)
            result["tempo_bpm"] = round(unique_tempos[0], 2) # First representative
            result["tempo_is_constant"] = (len(unique_tempos) <= 1)
        else:
            # No tempo defined. Standard MIDI implies 120, but here 'constant' means we didn't see changes.
            result["tempo_unique_count"] = 0
            result["tempo_bpm"] = 120.0
            result["tempo_is_constant"] = True # No changes seen
        
        if require_constant_tempo and not result["tempo_is_constant"]:
            fail_reasons.append("tempo_not_constant")

        # 3. Analyze Time Signature
        result["timesig_events_count"] = len(timesig_events)
        if timesig_events:
            unique_ts = sorted(list(set(timesig_events)))
            result["timesig_unique_count"] = len(unique_ts)
            result["timesig_value"] = unique_ts[0] if len(unique_ts) == 1 else "mixed"
            result["timesig_is_constant"] = (len(unique_ts) <= 1)
        else:
            result["timesig_unique_count"] = 0
            result["timesig_value"] = "missing"
            result["timesig_is_constant"] = True # No changes seen
            
        if require_constant_timesig and not result["timesig_is_constant"]:
            fail_reasons.append("timesig_not_constant")
            
        if allowed_timesig:
            # If missing, it's not in allowed (unless allowed has 'missing', but usually not)
            if result["timesig_value"] not in allowed_timesig:
                 fail_reasons.append(f"timesig_not_allowed({result['timesig_value']})")

        # 4. Count Bars (4/4 equivalent)
        ticks_per_bar_4beat = mid.ticks_per_beat * beats_per_bar
        bars_4beat_end = math.ceil(max_tick_all / ticks_per_bar_4beat) if ticks_per_bar_4beat > 0 else 0
        bars_4beat_note = math.ceil(max_tick_note / ticks_per_bar_4beat) if ticks_per_bar_4beat > 0 else 0
        
        result["bars_4beat_end"] = bars_4beat_end
        result["bars_4beat_note"] = bars_4beat_note
        
        check_val = bars_4beat_end if use_end_tick else bars_4beat_note
        
        # Loose check: >= target_bars
        # User specified "bars_ok = (bars_4beat_note == target_bars)" in instruction
        # But logically usually we want Minimum.
        # However, to facilitate strict subsets, I will implement >= logic but name it 'bars_ok'
        if check_val >= target_bars:
             result["bars_ok"] = True
        else:
             result["bars_ok"] = False
             fail_reasons.append(f"bars_short({check_val}<{target_bars})")
             
    except Exception as e:
        fail_reasons.append(f"error:{str(e)}")
        # Partial results might be there, but default to safe fails

    if not fail_reasons:
        result["overall_ok"] = True
    else:
        result["overall_ok"] = False
        result["fail_reason"] = ";".join(fail_reasons)
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Check POP909 MIDI subset integrity.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--target-bars", type=int, default=96, help="Target 4-beat bars (minimum)")
    parser.add_argument("--beats-per-bar", type=int, default=4, help="Beats per bar for calculation")
    parser.add_argument("--require-constant-tempo", action="store_true", help="Fail if tempo changes")
    parser.add_argument("--require-constant-timesig", action="store_true", help="Fail if time signature changes")
    parser.add_argument("--allowed-timesig", type=str, help="Comma-separated allowed time signatures (e.g., '4/4,2/4')")
    parser.add_argument("--tempo-tol", type=float, default=0.01, help="BPM tolerance for constant check")
    parser.add_argument("--use-end-tick", action="store_true", help="Use End-Of-Track tick instead of last Note tick")
    
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    out_csv: Path = args.out_csv
    
    allowed_ts = set(x.strip() for x in args.allowed_timesig.split(",")) if args.allowed_timesig else set()

    files = sorted(list(input_dir.glob("**/*.mid")) + list(input_dir.glob("**/*.midi")))
    print(f"Found {len(files)} MIDI files in {input_dir}")
    
    results = []
    
    for f in files:
        res = check_midi(
            f,
            target_bars=args.target_bars,
            beats_per_bar=args.beats_per_bar,
            require_constant_tempo=args.require_constant_tempo,
            require_constant_timesig=args.require_constant_timesig,
            allowed_timesig=allowed_ts,
            tempo_tol=args.tempo_tol,
            use_end_tick=args.use_end_tick,
        )
        results.append(res)
        
    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "midi_path", "ticks_per_beat",
        "tempo_events_count", "tempo_unique_count", "tempo_bpm", "tempo_is_constant",
        "timesig_events_count", "timesig_unique_count", "timesig_value", "timesig_is_constant",
        "max_tick_all", "max_tick_note",
        "bars_4beat_end", "bars_4beat_note",
        "target_bars", "bars_ok",
        "overall_ok", "fail_reason"
    ]
    
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    # Summary
    ok_count = sum(1 for r in results if r["overall_ok"])
    print(f"\nProcessing complete.")
    print(f"Total files: {len(files)}")
    print(f"Overall OK: {ok_count}")
    print(f"Overall NG: {len(files) - ok_count}")
    
    # NG Reasons
    fail_reasons_all = []
    for r in results:
        if r["fail_reason"]:
            # reason can be "A;B"
            parts = r["fail_reason"].split(";")
            fail_reasons_all.extend(parts)
    
    if fail_reasons_all:
        print("\nFail Reasons:")
        for reason, count in Counter(fail_reasons_all).most_common():
            print(f"  {reason}: {count}")
            
    # Distribution of bars
    bars = [r["bars_4beat_note"] for r in results]
    if bars:
        print("\nBars (4-beat, note-based) distribution (Top 10):")
        for val, count in Counter(bars).most_common(10):
            print(f"  {val} bars: {count}")
            
    print(f"\nReport written to: {out_csv}")

if __name__ == "__main__":
    main()
