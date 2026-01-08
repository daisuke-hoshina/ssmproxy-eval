#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input scan CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output selected CSV")
    parser.add_argument("--collect-dir", type=Path, default=None, help="Directory to copy selected MIDI files to")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--min-bars", type=int, default=100, help="Minimum number of bars")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows.")

    # Filter
    # 1. load_ok == 1
    # 2. time_sig_summary IN ['all_4_4', 'all_2_4']
    # 3. num_bars >= min_bars
    
    cond_load = df['load_ok'] == 1
    # Allow 4/4 and 2/4. Note: scan script outputs 'all_4_4', 'all_2_4', etc.
    cond_ts = df['time_sig_summary'].isin(['all_4_4', 'all_2_4'])
    cond_bars = df['num_bars'] >= args.min_bars
    
    filtered = df[cond_load & cond_ts & cond_bars].copy()
    print(f"Filtered: {len(filtered)} / {len(df)} satisfy criteria (4/4 or 2/4, >={args.min_bars} bars).")
    
    # Breakdown of time signatures
    print("Time signature breakdown in filtered set:")
    print(filtered['time_sig_summary'].value_counts())

    if len(filtered) < args.n:
        print(f"Warning: Only {len(filtered)} available, taking all.")
        sampled = filtered
    else:
        sampled = filtered.sample(n=args.n, random_state=args.seed)
    
    print(f"Selected {len(sampled)} tracks.")
    
    sampled.to_csv(args.output, index=False)
    print(f"Saved selected list to {args.output}")

    # Copy files if requested
    if args.collect_dir:
        dest_dir = args.collect_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Copying files to {dest_dir} ...")
        
        count = 0
        for _, row in sampled.iterrows():
            src = Path(row['midi_abs_path'])
            # Naming: pop909_001.mid
            dst_name = f"pop909_{row['pop909_id']}.mid"
            dst = dest_dir / dst_name
            try:
                shutil.copy2(src, dst)
                count += 1
            except Exception as e:
                print(f"Error copying {src}: {e}")
        
        print(f"Copied {count} files.")

if __name__ == "__main__":
    main()
