#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import pretty_midi

BEATS_PER_BAR = 4  # ssmproxy match

def get_time_signature_summary(midi: pretty_midi.PrettyMIDI) -> str:
    """
    Return summary of time signatures: 'all_4_4', 'all_2_4', 'mixed', 'unknown'.
    """
    tsc = getattr(midi, "time_signature_changes", None) or []
    if not tsc:
        # Some older pretty_midi versions or files might not have it.
        # Fallback to checking manually if possible, or unknown.
        return "unknown"
    
    numerators = set(ts.numerator for ts in tsc)
    denominators = set(ts.denominator for ts in tsc)
    
    if len(numerators) == 0:
        return "unknown"
    
    if len(numerators) == 1 and len(denominators) == 1:
        n = list(numerators)[0]
        d = list(denominators)[0]
        return f"all_{n}_{d}"
    
    # Check if mostly one type? For now just 'mixed'
    # We can be more detailed: e.g. "mixed_4_4_2_4"
    return "mixed"

def estimate_num_bars_like_ssmproxy(midi: pretty_midi.PrettyMIDI, *, exclude_drums: bool) -> tuple[int, float, int, int]:
    _, tempos = midi.get_tempo_changes()
    tempo0 = float(tempos[0]) if len(tempos) else 120.0
    if tempo0 <= 0:
        tempo0 = 120.0
    seconds_per_beat = 60.0 / tempo0

    last_start = None
    note_count = 0
    inst_count = 0

    for inst in midi.instruments:
        if exclude_drums and inst.is_drum:
            continue
        inst_count += 1
        for note in inst.notes:
            note_count += 1
            s = note.start
            if last_start is None or s > last_start:
                last_start = s

    if last_start is None:
        return 0, tempo0, note_count, inst_count

    last_beat_index = int(math.floor(last_start / seconds_per_beat))
    num_bars = last_beat_index // BEATS_PER_BAR + 1
    return num_bars, tempo0, note_count, inst_count

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True, help="Path to index.xlsx")
    ap.add_argument("--pop909-root", type=Path, required=True, help="Root dir containing year/ subdirs with MIDI files")
    ap.add_argument("--out-csv", type=Path, required=True, help="Output scan CSV path")
    ap.add_argument("--exclude-drums", action="store_true", help="Match ssmproxy default exclude_drums=True")
    args = ap.parse_args()

    meta_path: Path = args.metadata
    root: Path = args.pop909_root
    out_csv: Path = args.out_csv
    exclude_drums: bool = bool(args.exclude_drums)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Read metafile using pandas
    try:
        df = pd.read_excel(meta_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    results = []

    print(f"Scanning {len(df)} files from metadata...")

    # Suppress pretty_midi warnings for cleaner output
    warnings.filterwarnings("ignore")

    for _, row in df.iterrows():
        pop909_id = str(row.get('song_id') if 'song_id' in row else (row.get('pop909_id') if 'pop909_id' in row else row.get('id', '')))
        pop909_id = pop909_id.zfill(3)
        
        if not pop909_id:
            continue

        midi_rel = f"{pop909_id}/{pop909_id}.mid"
        midi_path = (root / midi_rel).resolve()
        
        out_row = {
            "pop909_id": pop909_id,
            "artist": row.get('artist', ''),
            "name": row.get('name', ''),
            "midi_filename": midi_rel,
            "midi_abs_path": str(midi_path),
        }

        try:
            if not midi_path.exists():
                raise FileNotFoundError(f"{midi_path} does not exist")

            midi = pretty_midi.PrettyMIDI(str(midi_path))
            out_row["load_ok"] = "1"
            out_row["error"] = ""

            ts_summary = get_time_signature_summary(midi)
            out_row["time_sig_summary"] = ts_summary

            num_bars, tempo0, note_count, inst_count = estimate_num_bars_like_ssmproxy(midi, exclude_drums=exclude_drums)
            _, tempos = midi.get_tempo_changes()
            out_row["num_bars"] = num_bars
            out_row["tempo0"] = tempo0
            out_row["num_tempo_changes"] = len(tempos)
            out_row["note_count"] = note_count
            out_row["instrument_count"] = inst_count

        except Exception as e:
            out_row["load_ok"] = "0"
            out_row["error"] = str(e)
            out_row["num_bars"] = -1
            out_row["tempo0"] = -1
            out_row["num_tempo_changes"] = -1
            out_row["time_sig_summary"] = "error"
            out_row["note_count"] = -1
            out_row["instrument_count"] = -1
        
        results.append(out_row)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote scan CSV: {out_csv} ({len(out_df)} rows)")

if __name__ == "__main__":
    main()
