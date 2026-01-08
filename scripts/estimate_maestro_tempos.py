#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import pretty_midi
from tqdm import tqdm

def estimate_tempo_ioi(path, exclude_drums=True, bpm_min=40, bpm_max=240):
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None

    onsets = []
    for inst in midi.instruments:
        if exclude_drums and inst.is_drum:
            continue
        onsets.extend([n.start for n in inst.notes])

    if len(onsets) < 2:
        return None

    onsets = np.sort(np.array(onsets))
    iois = np.diff(onsets)

    # Note: IOI logic from user
    # 極端に短い装飾音/長すぎる休符を除外（適宜調整）
    iois = iois[(iois >= 0.05) & (iois <= 2.0)]
    if len(iois) == 0:
        return None

    bpms = 60.0 / iois

    # bpmをレンジに折り畳む（×2/÷2）
    folded = []
    for b in bpms:
        # Simple folding to [min, max]
        # logic: while b < min: b*=2; while b > max: b/=2
        # But doing it once might not be enough if it's very far off, so loop
        
        # User defined logic:
        # while b < bpm_min: b *= 2.0
        # while b > bpm_max: b /= 2.0
        curr = b
        while curr < bpm_min:
            curr *= 2.0
        while curr > bpm_max:
            curr /= 2.0
        folded.append(curr)
        
    folded = np.array(folded)

    if len(folded) == 0:
        return None

    # 1BPM刻みのヒストグラムで最頻値
    bins = np.arange(bpm_min, bpm_max + 1, 1.0)
    hist, edges = np.histogram(folded, bins=bins)
    peak_idx = int(np.argmax(hist))
    bpm_hat = float(edges[peak_idx])  # bin左端
    return bpm_hat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True, help="Input scan CSV")
    parser.add_argument("--out-csv", required=True, help="Output CSV with tempo")
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)

    with in_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    fieldnames = list(reader[0].keys())
    if "estimated_bpm" not in fieldnames:
        fieldnames.append("estimated_bpm")

    total = len(reader)
    print(f"Estimating tempo for {total} files...")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader):
            path = row.get("midi_abs_path")
            bpm = ""
            if path and row.get("load_ok") == "1":
                try:
                    val = estimate_tempo_ioi(path)
                    if val is not None:
                        bpm = f"{val:.1f}"
                except Exception as e:
                    pass # ignore errors, leave empty
            
            row["estimated_bpm"] = bpm
            writer.writerow(row)

    print(f"Done. Wrote to {out_csv}")

if __name__ == "__main__":
    main()
