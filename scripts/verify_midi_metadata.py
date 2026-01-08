import sys
from pathlib import Path
import pretty_midi
import mido

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify_midi_metadata.py <midi_path>")
        return

    path = sys.argv[1]
    print(f"--- Checking: {path} ---")

    print("\n[pretty_midi]")
    try:
        midi = pretty_midi.PrettyMIDI(path)

        # テンポ変化（time[], tempo[]）
        t, tempos = midi.get_tempo_changes()
        print("tempo changes:", list(zip(t, tempos))[:10], " ... total:", len(tempos))

        # 拍子変化（PrettyMIDIのTimeSignatureオブジェクト）
        ts = midi.time_signature_changes
        print("time signatures:", [(x.time, x.numerator, x.denominator) for x in ts])
    except Exception as e:
        print(f"Error loading with pretty_midi: {e}")

    print("\n[mido]")
    try:
        mid = mido.MidiFile(path)
        found_meta = False
        for ti, track in enumerate(mid.tracks):
            for msg in track:
                if msg.type in ("set_tempo", "time_signature"):
                    print("track", ti, msg)
                    found_meta = True
        if not found_meta:
            print("No set_tempo or time_signature messages found in any track.")
    except Exception as e:
        print(f"Error loading with mido: {e}")

if __name__ == "__main__":
    main()
