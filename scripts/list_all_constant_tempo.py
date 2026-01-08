#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-csv", type=Path, required=True, help="Full check report CSV")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.check_csv)
    print(f"Loaded {len(df)} rows.")

    # Filter: tempo_is_constant == True
    # We ignore bars_ok, timesig_ok etc.
    if df['tempo_is_constant'].dtype == object:
        constant_tempo_df = df[df['tempo_is_constant'].astype(str) == 'True'].copy()
    else:
        constant_tempo_df = df[df['tempo_is_constant']].copy()

    print(f"Constant Tempo Files: {len(constant_tempo_df)}")

    # Add Song ID
    def get_song_id(path_str):
        p = Path(path_str)
        if p.parent.name.isdigit() and len(p.parent.name) == 3:
             return p.parent.name
        if p.parent.name == 'versions' and p.parent.parent.name.isdigit():
             return p.parent.parent.name
        return "unknown"

    constant_tempo_df['song_id'] = constant_tempo_df['midi_path'].apply(get_song_id)
    
    # Sort for readability
    constant_tempo_df = constant_tempo_df.sort_values(by=['song_id', 'midi_path'])
    
    # Count unique
    unique_ids = constant_tempo_df['song_id'].unique()
    unique_ids = [u for u in unique_ids if u != "unknown"]
    print(f"Unique Song IDs with Constant Tempo: {len(unique_ids)}")

    # Save
    cols = ['song_id', 'midi_path', 'tempo_bpm', 'bars_4beat_note', 'ticks_per_beat', 'fail_reason']
    # Keep fail_reason just in case user wants to see why it failed strict checks (e.g. length)
    
    constant_tempo_df[cols].to_csv(args.out_csv, index=False)
    print(f"Saved report to {args.out_csv}")

if __name__ == "__main__":
    main()
