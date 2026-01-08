#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import pretty_midi
from tqdm import tqdm

def merge_onsets(onsets, merge_sec=0.03):
    """
    Merge onsets that are closer than merge_sec.
    onsets: sorted list of floats
    """
    if not onsets:
        return []
    
    # Ensure sorted
    onsets = np.sort(np.array(onsets))
    
    merged = [onsets[0]]
    last = onsets[0]
    
    for t in onsets[1:]:
        if t - last >= merge_sec:
            merged.append(t)
            last = t
    return np.array(merged)

def estimate_tempo_ioi_v2(path, exclude_drums=True, 
                          merge_sec=0.03, k_max=8,
                          ioi_min=0.08, ioi_max=2.0,
                          bpm_min=40, bpm_max=240, bin_width=1.0):
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        # load failure
        return None, None, None

    raw_onsets = []
    for inst in midi.instruments:
        if exclude_drums and inst.is_drum:
            continue
        raw_onsets.extend([n.start for n in inst.notes])

    if len(raw_onsets) < 2:
        return None, None, None

    # A) Onset Merge
    onsets = merge_onsets(raw_onsets, merge_sec=merge_sec)
    
    if len(onsets) < 2:
        return None, None, None

    # B) k-hop diffs
    all_iois = []
    # k=1 to k_max
    # limit k_max by length
    valid_k_max = min(k_max, len(onsets) - 1)
    
    for k in range(1, valid_k_max + 1):
        # diff between i and i+k
        diffs = onsets[k:] - onsets[:-k]
        # filter
        valid_diffs = diffs[(diffs >= ioi_min) & (diffs <= ioi_max)]
        all_iois.append(valid_diffs)
        
    if not all_iois:
        return None, None, None
        
    iois = np.concatenate(all_iois)
    
    if len(iois) == 0:
        return None, None, None

    # Convert to BPM
    bpms = 60.0 / iois

    # Fold to range
    folded = []
    for b in bpms:
        curr = b
        # Avoid infinite loop if logic is weird, but usually simple while works
        # logic: fold into [bpm_min, bpm_max]
        # If b is very small, multiply by 2 until >= bpm_min
        # If b is very large, divide by 2 until <= bpm_max
        
        # Optimization: use log2 to jump? simplistic loop is fine for MIDI
        while curr < bpm_min and curr * 2 < 10000: # safety break
             curr *= 2.0
             if curr < bpm_min and curr * 2 >= bpm_min: # check if next double is in range
                 pass
             
        while curr > bpm_max and curr > 0.1: # safety
             curr /= 2.0
             
        # double check range (sometimes it might still be out if min/max gap is narrow)
        if bpm_min <= curr <= bpm_max:
             folded.append(curr)
        # If it's slightly out due to folding logic gaps (e.g. min=100, max=120, val=90->180->90), currently ignoring
             
    folded = np.array(folded)
    if len(folded) == 0:
        return None, None, None

    # Histogram
    # bins: bpm_min, bpm_min+1, ... bpm_max+1
    bins = np.arange(bpm_min, bpm_max + bin_width + 1e-9, bin_width)
    hist, edges = np.histogram(folded, bins=bins)
    
    if np.sum(hist) == 0:
        return None, None, None

    # Mode
    peak_idx = int(np.argmax(hist))
    bpm_hat = float(edges[peak_idx])

    # C) Confidence
    count_top1 = hist[peak_idx]
    
    # second peak
    # mask top1 to find top2
    hist_masked = hist.copy()
    hist_masked[peak_idx] = -1
    count_top2 = np.max(hist_masked)
    if count_top2 < 0: count_top2 = 0 # if only 1 bin had values
    
    total_count = np.sum(hist)
    
    # Metrics
    if total_count > 0:
        confidence = count_top1 / total_count
    else:
        confidence = 0.0
        
    if count_top2 > 0:
        peak_ratio = count_top1 / count_top2
    else:
        # infinite ratio if no second peak. Just set to a high number or top1
        # user said: peak_ratio = top1_count / (top2_count + 1e-9)
        peak_ratio = count_top1 / (count_top2 + 1e-9)

    return bpm_hat, confidence, peak_ratio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True, help="Input CSV")
    parser.add_argument("--out-csv", required=True, help="Output CSV")
    
    # Parameters
    parser.add_argument("--merge-sec", type=float, default=0.03)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--ioi-min", type=float, default=0.08)
    parser.add_argument("--ioi-max", type=float, default=2.0)
    parser.add_argument("--bpm-min", type=float, default=40.0)
    parser.add_argument("--bpm-max", type=float, default=240.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)

    if not in_csv.exists():
        print(f"Error: {in_csv} not found")
        sys.exit(1)

    with in_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    fieldnames = list(reader[0].keys())
    # Add new columns
    for col in ["estimated_bpm_v2", "tempo_confidence", "tempo_peak_ratio"]:
        if col not in fieldnames:
            fieldnames.append(col)

    total = len(reader)
    print(f"Estimating tempo v2 for {total} files...")
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader):
            path = row.get("midi_abs_path")
            
            # Defaults
            row["estimated_bpm_v2"] = ""
            row["tempo_confidence"] = ""
            row["tempo_peak_ratio"] = ""

            if path and row.get("load_ok") == "1":
                bpm, conf, ratio = estimate_tempo_ioi_v2(
                    path, 
                    merge_sec=args.merge_sec,
                    k_max=args.k_max,
                    ioi_min=args.ioi_min,
                    ioi_max=args.ioi_max,
                    bpm_min=args.bpm_min,
                    bpm_max=args.bpm_max,
                    bin_width=args.bin_width
                )
                
                if bpm is not None:
                    row["estimated_bpm_v2"] = f"{bpm:.1f}"
                    row["tempo_confidence"] = f"{conf:.4f}"
                    row["tempo_peak_ratio"] = f"{ratio:.4f}"
            
            writer.writerow(row)

    print(f"Done. Wrote to {out_csv}")

if __name__ == "__main__":
    main()
