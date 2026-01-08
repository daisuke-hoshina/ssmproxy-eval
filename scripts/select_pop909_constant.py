#!/usr/bin/env python3
import argparse
import shutil
import random
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-csv", type=Path, required=True, help="Full check report CSV")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for selected MIDI")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output summary CSV")
    parser.add_argument("--n", type=int, default=100, help="Number of songs to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.check_csv)
    print(f"Loaded {len(df)} rows.")

    # Filter overall_ok
    # Note: 'overall_ok' corresponds to the strict criteria used in the check run:
    # constant tempo AND >= target bars (96).
    # The output from check_pop909_subset.py has string 'True'/'False' for overall_ok if using pandas default, or boolean.
    # Let's handle both.
    
    # Try converting to boolean
    if df['overall_ok'].dtype == object:
         ok_df = df[df['overall_ok'].astype(str) == 'True'].copy()
    else:
         ok_df = df[df['overall_ok']].copy()

    print(f"Candidates (overall_ok): {len(ok_df)}")
    
    def get_song_id(path_str):
        # Very simple heuristic: find the 3-digit folder or filename
        p = Path(path_str)
        # Check parent folder name
        if p.parent.name.isdigit() and len(p.parent.name) == 3:
             return p.parent.name
        # Check grandparent if parent is 'versions'
        if p.parent.name == 'versions' and p.parent.parent.name.isdigit():
             return p.parent.parent.name
        return "unknown"

    ok_df['song_id'] = ok_df['midi_path'].apply(get_song_id)
    
    # Group by song_id
    # We want unique songs.
    unique_songs = ok_df['song_id'].unique()
    # Filter out 'unknown' if any
    unique_songs = [s for s in unique_songs if s != "unknown"]
    
    print(f"Unique song IDs: {len(unique_songs)}")
    
    random.seed(args.seed)
    
    selected_ids = []
    if len(unique_songs) < args.n:
        print(f"Warning: Only {len(unique_songs)} unique songs available. Taking all.")
        selected_ids = list(unique_songs)
    else:
        selected_ids = random.sample(list(unique_songs), args.n)
    
    print(f"Selected {len(selected_ids)} IDs.")
    
    # Collect files
    # For each ID, pick one file. Prefer the main one (001.mid inside 001/) over versions if available in ok_df.
    
    final_selection = []
    
    for sid in selected_ids:
        # Get all candidates for this song
        candidates = ok_df[ok_df['song_id'] == sid]
        
        # Priority: filename == sid.mid
        # e.g. 001.mid
        # We need to be careful about path separator or just match name
        main_ver = candidates[candidates['midi_path'].apply(lambda p: Path(p).name == f"{sid}.mid")]
        
        if not main_ver.empty:
            chosen = main_ver.iloc[0]
        else:
            # Pick any
            chosen = candidates.iloc[0]
            
        final_selection.append(chosen)
        
    final_df = pd.DataFrame(final_selection)
    
    # Clean output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying {len(final_df)} files to {args.out_dir} ...")
    count = 0
    for _, row in final_df.iterrows():
        src = Path(row['midi_path'])
        dst_name = f"pop909_{row['song_id']}.mid"
        dst = args.out_dir / dst_name
        try:
            shutil.copy2(src, dst)
            count += 1
        except Exception as e:
            print(f"Error copying {src}: {e}")
            
    final_df.to_csv(args.out_csv, index=False)
    print(f"Saved selection list to {args.out_csv}")

if __name__ == "__main__":
    main()
