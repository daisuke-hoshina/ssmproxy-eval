"""Dataset utilities for scanning, collecting, and sampling MIDI files."""

import csv
import hashlib
import math
import random
import shutil
from pathlib import Path
from typing import List, Optional

import pretty_midi

BEATS_PER_BAR = 4
STEPS_PER_BAR = 16
STEPS_PER_BEAT = STEPS_PER_BAR // BEATS_PER_BAR


def _compute_bars_fast(midi: pretty_midi.PrettyMIDI) -> int:
    """Compute number of bars using the same logic as bar_features.compute_bar_features.
    
    Logic:
    - Use only the initial tempo (or 120 if invalid).
    - Find the last note-on event time.
    - Calculate last beat index.
    - num_bars = last_beat_index // 4 + 1
    """
    tempos_times, tempos = midi.get_tempo_changes()
    tempo = float(tempos[0]) if len(tempos) else 120.0
    if tempo <= 0:
        tempo = 120.0

    seconds_per_beat = 60.0 / tempo
    
    # pretty_midi.PrettyMIDI.instruments is a list of Instrument objects
    # We aggregate all note starts from non-drum instruments (usually we exclude drums)
    # But wait, logic in bar_features says:
    # note_on_events = extract_note_on_events(midi, exclude_drums=exclude_drums)
    # Let's inspect bar_features logic in memory... 
    # It uses extract_note_on_events. To avoid circular dependency or code duplication, 
    # we can try to be consistent. 
    # Since this function is for 'scan', and performance is key, we can do a quick gathering.
    # However, strict consistency is required.
    
    # Let's do a lightweight version of extract_note_on_events logic here
    # assuming we want to exclude drums by default or generally follow the rule.
    # The user requirement says: "scanは高速化のため、bar_features のロジック...を再実装しても良い"
    # "exclude_drums / --include-drums (default: True)" in scan command.
    
    # We will pass exclude_drums as arg to this internal function if needed, 
    # but for now let's assume we implement the logic here.
    pass

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

        # 2. Bar Estimate
        # Re-implement bar logic locally for speed avoiding full feature extraction overhead
        tempos_times, tempos = midi.get_tempo_changes()
        tempo = float(tempos[0]) if len(tempos) > 0 else 120.0
        if tempo <= 0:
            tempo = 120.0
        seconds_per_beat = 60.0 / tempo
        
        last_start = 0.0
        has_notes = False
        for instr in midi.instruments:
            if exclude_drums and instr.is_drum:
                continue
            for note in instr.notes:
                if note.start > last_start:
                    last_start = note.start
                has_notes = True
        
        if not has_notes:
            num_bars = 0
        else:
            last_beat_index = int(math.floor(last_start / seconds_per_beat))
            num_bars = last_beat_index // BEATS_PER_BAR + 1
            
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
                "selected": final_selected,
                "error": ""
            })

    # Write CSV
    header = ["midi_rel_path", "midi_abs_path", "num_bars", "is_all_4_4", "selected", "error"]
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
