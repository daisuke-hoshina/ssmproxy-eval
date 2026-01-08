import csv
import random
import shutil
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-csv", required=True, help="Path to scan CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to copy/symlink files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n", type=int, default=100, help="Number of songs to sample")
    args = parser.parse_args()
    
    scan_csv = Path(args.scan_csv)
    output_dir = Path(args.output_dir)
    seed = args.seed
    n_sample = args.n
    
    if not scan_csv.exists():
        print(f"Error: {scan_csv} not found.")
        return

    random.seed(seed)
    
    candidates = []
    
    with scan_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check load error
            if row.get("load_ok") != "1":
                continue
            
            # Check 4/4
            if row.get("time_sig_status") != "all_4_4":
                continue
                
            # Check bars
            try:
                bars = int(row.get("num_bars", "0"))
            except ValueError:
                bars = 0
            
            if bars >= 128:
                candidates.append(row)
                
    print(f"Found {len(candidates)} candidates matching criteria (4/4, >=128 bars).")
    
    if len(candidates) < n_sample:
        print(f"Warning: requested {n_sample} but only found {len(candidates)}. Taking all.")
        selected = candidates
    else:
        selected = random.sample(candidates, n_sample)
        
    print(f"Selected {len(selected)} songs.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected metadata
    out_meta = output_dir / "selected_meta.csv"
    if selected:
        fieldnames = list(selected[0].keys())
        with out_meta.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
            
    # Copy/Link
    count = 0
    for row in selected:
        src = Path(row["midi_abs_path"])
        if not src.exists():
            print(f"Warning: Source file not found: {src}")
            continue
            
        dst_name = src.name
        dst = output_dir / dst_name
        
        # Avoid overwrite collision
        if dst.exists():
             # Basic collision handling
             stem = src.stem
             suffix = src.suffix
             # Use parent folder name as prefix? e.g. 2018_fname.midi
             # src.parent.name is usually year
             prefix = src.parent.name
             dst = output_dir / f"{prefix}_{stem}{suffix}"
             
        try:
            if dst.exists():
                dst.unlink() # remove existing link/file
            os.symlink(src, dst)
            count += 1
        except OSError:
            try:
                shutil.copy2(src, dst)
                count += 1
            except Exception as e:
                print(f"Failed to copy {src}: {e}")
                
    print(f"Successfully collected {count} files to {output_dir}")

if __name__ == "__main__":
    main()
