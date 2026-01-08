"""Dataset utilities for scanning, collecting, and sampling MIDI files."""

import csv
import hashlib
import math
import random
import shutil
from pathlib import Path
from typing import List, Optional

import pretty_midi

from .bar_features import get_beat_times, time_to_beat_index, time_to_beat_index_end, BEATS_PER_BAR
from .midi_io import extract_note_events


def _compute_bars_fast(midi: pretty_midi.PrettyMIDI, exclude_drums: bool = True, analysis_beats_per_bar: int = 4) -> int:
    """Compute number of bars using beat-grid logic."""
    notes = extract_note_events(midi, exclude_drums=exclude_drums)
    
    if not notes:
        return 0
        
    # Must match compute_bar_features logic exactly
    last_end = max(n[1] for n in notes)
    beats = get_beat_times(midi, target_end_time=last_end)
    
    # Use end-exclusive logic matching compute_bar_features
    last_beat_idx = time_to_beat_index_end(last_end, beats)
    num_bars = (last_beat_idx // analysis_beats_per_bar) + 1
    return num_bars

def scan_dataset(
    input_dir: Path,
    out_csv: Path,
    min_bars: int = 128,
    require_4_4: bool = False,
    unknown_ts_is_4_4: bool = True,
    exclude_drums: bool = True,
    max_files: Optional[int] = None,
    seed: Optional[int] = None,
    write_all: bool = False,
    analysis_beats_per_bar: int = 4,
) -> None:
    """Scan a directory for MIDI files and write metadata to a CSV."""
    
    files = sorted(list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi")))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(files)
        
    if max_files is not None:
        files = files[:max_files]
        
    rows = []
    
    print(f"Scanning {len(files)} files in {input_dir}...")
    
    for fpath in files:
        rel_path = fpath.relative_to(input_dir)
        try:
            midi = pretty_midi.PrettyMIDI(str(fpath))
        except Exception as e:
            # If we can't parse, we probably skip or mark as failed if write_all
            if write_all:
                 rows.append({
                    "midi_rel_path": str(rel_path.as_posix()),
                    "midi_abs_path": str(fpath.resolve()),
                    "num_bars": -1,
                    "is_all_4_4": "",
                    "selected": False,
                    "error": str(e)
                })
            continue

        # 1. 4/4 Check
        if not midi.time_signature_changes:
            # Unknown
            is_4_4 = unknown_ts_is_4_4
            is_all_4_4_str = "" # Unknown
        else:
            # Check if ALL TS changes are 4/4
            all_4_4 = all(ts.numerator == 4 and ts.denominator == 4 for ts in midi.time_signature_changes)
            is_4_4 = all_4_4
            is_all_4_4_str = str(all_4_4)
            
        if require_4_4 and not is_4_4:
            if not write_all:
                continue
            is_selected_ts = False
        else:
            is_selected_ts = True

        # 2. Bar Estimate & Tempo Check
        tempos_times, tempos = midi.get_tempo_changes()
        if not len(tempos):
            is_constant_tempo = True
        else:
            is_constant_tempo = (max(tempos) - min(tempos) < 1e-3)
            
        num_bars = _compute_bars_fast(midi, exclude_drums=exclude_drums, analysis_beats_per_bar=analysis_beats_per_bar)
            
        # 3. Min Bars filter
        if num_bars < min_bars:
            is_selected_bars = False
        else:
            is_selected_bars = True
            
        final_selected = is_selected_ts and is_selected_bars
        
        if final_selected or write_all:
             rows.append({
                "midi_rel_path": str(rel_path.as_posix()),
                "midi_abs_path": str(fpath.resolve()),
                "num_bars": num_bars,
                "is_all_4_4": is_all_4_4_str,
                "is_constant_tempo": is_constant_tempo,
                "selected": final_selected,
                "error": ""
            })

    # Write CSV
    header = ["midi_rel_path", "midi_abs_path", "num_bars", "is_all_4_4", "is_constant_tempo", "selected", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Scanned {len(files)} files. Wrote {len(rows)} entries to {out_csv}.")


def collect_dataset(
    in_csv: Path,
    out_dir: Path,
    mode: str = "symlink",
    flatten: bool = False,
    name_from: str = "rel",
    dry_run: bool = False,
    only_selected: bool = True,
    limit: Optional[int] = None,
) -> None:
    """Collect files based on CSV."""
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    count = 0
    for row in rows:
        if limit is not None and count >= limit:
            break
            
        if only_selected:
            # Handle 'True'/'False' strings
            sel = row.get("selected", "True").lower() == "true"
            if not sel:
                continue
        
        src_path = Path(row["midi_abs_path"])
        if not src_path.exists():
            # Try resolving relative path if abs path is empty or wrong? 
            # The user spec says "scanner makes midi_abs_path".
            print(f"Warning: Source not found {src_path}")
            continue
            
        rel_path = row["midi_rel_path"] # e.g. "composer/piece.mid"
        
        if flatten:
            if name_from == "hash":
                # Hash the relative path to get a unique name
                ext = src_path.suffix
                h = hashlib.md5(rel_path.encode("utf-8")).hexdigest()
                dest_name = f"{h}{ext}"
            else:
                # rel path with __ replacement
                dest_name = rel_path.replace("/", "__").replace("\\", "__")
            
            dest_path = out_dir / dest_name
        else:
            # preserve tree
            dest_path = out_dir / rel_path
            
        if not dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            if dest_path.exists():
                # Simple skip or overwrite? Let's overwrite or skip. 
                # For safety, remove if exists?
                if dest_path.is_symlink() or dest_path.is_file():
                    dest_path.unlink()

            if mode == "symlink":
                try:
                    dest_path.symlink_to(src_path)
                except OSError:
                    # Fallback
                    shutil.copy2(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)
        
        count += 1
        
    print(f"Collected {count} files to {out_dir} (dry_run={dry_run})")


def sample_dataset(
    in_csv: Path,
    out_csv: Path,
    n: int,
    seed: int = 0,
    only_selected: bool = True,
    shuffle_output: bool = True,
) -> None:
    """Sample N rows from the CSV."""
    
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    if only_selected:
        rows = [r for r in rows if r.get("selected", "True").lower() == "true"]
        
    if n > len(rows):
        raise ValueError(f"Requested {n} samples but only {len(rows)} available.")
        
    rng = random.Random(seed)
    # If we need random sample
    # We can just shuffle and take N
    sampled = list(rows)
    rng.shuffle(sampled)
    sampled = sampled[:n]
    
    if not shuffle_output:
        # Sort back to original order? Or just leave as is?
        # User spec says "shuffle (default True)". If False, maybe sort by path?
        # Let's simple sort by midi_rel_path if not shuffle
        sampled.sort(key=lambda x: x["midi_rel_path"])
        
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)
        
    print(f"Sampled {len(sampled)} items to {out_csv}")
