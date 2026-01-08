#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import pretty_midi

BEATS_PER_BAR = 4  # ssmproxy と同じ前提

def is_all_4_4(midi: pretty_midi.PrettyMIDI) -> Optional[bool]:
    """Return True if all time signatures are 4/4, False if any is not, None if unknown."""
    tsc = getattr(midi, "time_signature_changes", None) or []
    if not tsc:
        return None
    return all(ts.numerator == 4 and ts.denominator == 4 for ts in tsc)

def estimate_num_bars_like_ssmproxy(midi: pretty_midi.PrettyMIDI, *, exclude_drums: bool) -> tuple[int, float, int, int]:
    """
    Match ssmproxy.bar_features.compute_bar_features's bar counting:
    - use first tempo
    - bars from last note-on start: last_beat_index // 4 + 1
    """
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
    ap.add_argument("--metadata", type=Path, required=True, help="Path to maestro-v3.0.0.csv")
    ap.add_argument("--maestro-root", type=Path, required=True, help="Root dir containing year/ subdirs with MIDI files")
    ap.add_argument("--out-csv", type=Path, required=True, help="Output scan CSV path")
    ap.add_argument("--exclude-drums", action="store_true", help="Match ssmproxy default exclude_drums=True")
    args = ap.parse_args()

    meta_path: Path = args.metadata
    root: Path = args.maestro_root
    out_csv: Path = args.out_csv
    exclude_drums: bool = bool(args.exclude_drums)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Read metafile
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fields = [
        # from metadata
        "canonical_composer",
        "canonical_title",
        "split",
        "year",
        "midi_filename",
        "duration",
        # scan results
        "midi_abs_path",
        "load_ok",
        "error",
        "num_bars",
        "tempo0",
        "num_tempo_changes",
        "time_sig_status",   # 'all_4_4' / 'has_non_4_4' / 'unknown'
        "note_count",
        "instrument_count",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()

        for r in rows:
            midi_rel = r.get("midi_filename", "")
            midi_path = (root / midi_rel).resolve()

            out_row = {k: r.get(k, "") for k in ["canonical_composer", "canonical_title", "split", "year", "midi_filename", "duration"]}
            out_row["midi_abs_path"] = str(midi_path)

            try:
                midi = pretty_midi.PrettyMIDI(str(midi_path))
                out_row["load_ok"] = "1"
                out_row["error"] = ""

                ts = is_all_4_4(midi)
                if ts is None:
                    out_row["time_sig_status"] = "unknown"
                elif ts:
                    out_row["time_sig_status"] = "all_4_4"
                else:
                    out_row["time_sig_status"] = "has_non_4_4"

                num_bars, tempo0, note_count, inst_count = estimate_num_bars_like_ssmproxy(midi, exclude_drums=exclude_drums)
                _, tempos = midi.get_tempo_changes()
                out_row["num_bars"] = str(num_bars)
                out_row["tempo0"] = f"{tempo0:.6f}"
                out_row["num_tempo_changes"] = str(len(tempos))
                out_row["note_count"] = str(note_count)
                out_row["instrument_count"] = str(inst_count)

            except Exception as e:
                out_row["load_ok"] = "0"
                out_row["error"] = repr(e)
                out_row["num_bars"] = ""
                out_row["tempo0"] = ""
                out_row["num_tempo_changes"] = ""
                out_row["time_sig_status"] = ""
                out_row["note_count"] = ""
                out_row["instrument_count"] = ""

            w.writerow(out_row)

    print(f"Wrote scan CSV: {out_csv}")

if __name__ == "__main__":
    main()
