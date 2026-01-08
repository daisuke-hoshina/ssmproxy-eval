import pretty_midi
import mido
import sys
import os

def check_file(path):
    print(f"--- Checking {os.path.basename(path)} ---")
    
    # PrettyMIDI
    print("[PrettyMIDI]")
    try:
        midi = pretty_midi.PrettyMIDI(path)
        # テンポ変化（time[], tempo[]）
        t, tempos = midi.get_tempo_changes()
        print("  tempo changes:", list(zip(t, tempos))[:10], " ... total:", len(tempos))

        # 拍子変化（PrettyMIDIのTimeSignatureオブジェクト）
        ts = midi.time_signature_changes
        print("  time signatures:", [(x.time, x.numerator, x.denominator) for x in ts])
    except Exception as e:
        print(f"  Error with PrettyMIDI: {e}")

    # mido
    print("\n[mido]")
    try:
        mid = mido.MidiFile(path)
        for ti, track in enumerate(mid.tracks):
            # Print only first few relevant messages to avoid clutter
            count = 0
            has_relevant = False
            for msg in track:
                if msg.type in ("set_tempo", "time_signature"):
                    print(f"  track {ti}: {msg}")
                    has_relevant = True
                    count += 1
            if has_relevant:
                 print(f"  (Found {count} tempo/sig messages in track {ti})")
    except Exception as e:
        print(f"  Error with mido: {e}")
    print("\n")

def main():
    # Check a few files
    files = [
        "/Users/daisuke/Downloads/Research/POP909/001/001.mid",
        "/Users/daisuke/Downloads/Research/POP909/002/002.mid",
        "/Users/daisuke/Downloads/Research/POP909/003/003.mid"
    ]
    
    for f in files:
        if os.path.exists(f):
            check_file(f)
        else:
            print(f"File not found: {f}")

if __name__ == "__main__":
    main()
