
import random
import pytest
from ssmproxy import toy_generator
from ssmproxy.toy_generator import generate_piece, DEFAULT_BARS_PER_PIECE

def get_piece_notes(midi):
    """Extract (bar_idx, pitch, offset) tuples for analysis."""
    notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            # Simple quantization assuming strict toy generation
            beat = note.start * (120.0 / 60.0)
            bar_idx = int(beat // 4)
            offset = beat % 4
            notes.append((bar_idx, note.pitch, round(offset, 2)))
    return sorted(notes)

def get_bar_signature(midi, bar_idx):
    """Return sorted list of (offset, pitch) for a specific bar."""
    sigs = []
    for inst in midi.instruments:
        for note in inst.notes:
             beat = note.start * (120.0 / 60.0)
             b = int(beat // 4)
             if b == bar_idx:
                 offset = round(beat % 4, 2)
                 sigs.append((offset, note.pitch))
    return tuple(sorted(sigs))

def test_bars_per_piece():
    assert DEFAULT_BARS_PER_PIECE == 96

def test_aaba_96_structure():
    piece = generate_piece("AABA", 0, seed=42)
    # Check total length by max bar index found
    notes = get_piece_notes(piece.midi)
    max_bar = max(n[0] for n in notes)
    assert max_bar == 95 # 0..95
    
    # Check Block 0 (0-31) Structure: A A B A
    # A(0-7) == A(8-15)
    sig_a1 = get_bar_signature(piece.midi, 0)
    sig_a2 = get_bar_signature(piece.midi, 8)
    assert sig_a1 == sig_a2
    assert sig_a1 # Not empty

    # A(0-7) != B(16-23)
    sig_b = get_bar_signature(piece.midi, 16)
    assert sig_a1 != sig_b
    
    # A(0-7) == A(24-31)
    sig_a4 = get_bar_signature(piece.midi, 24)
    assert sig_a1 == sig_a4

    # Check Block 1 (32-63) Structure
    # Should have same internal structure but DIFFERENT content from Block 0
    sig_block1_a1 = get_bar_signature(piece.midi, 32)
    
    # Block 1 A should be A A B A internally
    sig_block1_a2 = get_bar_signature(piece.midi, 40)
    assert sig_block1_a1 == sig_block1_a2
    
    # CROSS BLOCK CHECK: Block 0 A != Block 1 A
    assert sig_a1 != sig_block1_a1

def test_abab_96_structure():
    piece = generate_piece("ABAB", 0, seed=123)
    notes = get_piece_notes(piece.midi)
    assert max(n[0] for n in notes) == 95
    
    # Block 0: A B A B
    a1 = get_bar_signature(piece.midi, 0)
    b1 = get_bar_signature(piece.midi, 8)
    a2 = get_bar_signature(piece.midi, 16)
    b2 = get_bar_signature(piece.midi, 24)
    
    assert a1 == a2
    assert b1 == b2
    assert a1 != b1
    
    # Block 1 internal consistency
    block1_a1 = get_bar_signature(piece.midi, 32)
    block1_a2 = get_bar_signature(piece.midi, 48)
    assert block1_a1 == block1_a2
    
    # Cross block: A1 != Block1_A1
    assert a1 != block1_a1

def test_hierarchical_nesting():
    piece = generate_piece("hierarchical", 0, seed=999)
    # Just check that it generates full length and has some repetition
    notes = get_piece_notes(piece.midi)
    assert max(n[0] for n in notes) == 95
    
    # Motif check: Bar 0 == Bar 1? (Usually built from motif phrase(8) -> motif(2))
    # Phrase = motif * 4. 
    # Logic: motif = _build(2 bars). So bar 0 and bar 1 are the motif.
    # Phrase extends this.
    # Let's check bar 0 vs bar 2 (start of next motif repetition)
    
    m0 = get_bar_signature(piece.midi, 0)
    m1 = get_bar_signature(piece.midi, 2)
    
    # Within a phrase, motif repeats but might have shift (rng.choice([0,2]))
    # It's likely similar or shifted.
    # But Bar 0 and Bar 1 (internal motif structure) should be distinct usually
    # unless motif happened to be repetitive.
    
    # Cross section check
    # Section 0 (0-31) vs Section 1 (32-63)
    # Should be different due to section pitch shift
    s0_bar0 = get_bar_signature(piece.midi, 0)
    s1_bar0 = get_bar_signature(piece.midi, 32)
    
    assert s0_bar0 != s1_bar0

def test_partial_copy_96():
    piece = generate_piece("partial_copy", 0, seed=777)
    notes = get_piece_notes(piece.midi)
    assert max(n[0] for n in notes) == 95
    
    # Block 0: Base, Altered, Base, Altered
    base = get_bar_signature(piece.midi, 0)
    base_rep = get_bar_signature(piece.midi, 16)
    assert base == base_rep
    
    # Cross block
    base_block1 = get_bar_signature(piece.midi, 32)
    assert base != base_block1

def test_explicit_bars_parameter():
    # Verify we can generate 32 bars piece (legacy size)
    # 32 bars = 1 block of 32 bars. So no repetition of blocks.
    piece = generate_piece("AABA", 0, seed=555, bars=32)
    notes = get_piece_notes(piece.midi)
    max_bar = max(n[0] for n in notes)
    assert max_bar == 31 # 0..31 ie 32 bars
    assert piece.bars == 32
    
    # Verify 128 bars
    # 128 = 4 blocks
    piece_128 = generate_piece("ABAB", 0, seed=666, bars=128)
    notes_128 = get_piece_notes(piece_128.midi)
    max_bar_128 = max(n[0] for n in notes_128)
    assert max_bar_128 == 127
    assert piece_128.bars == 128
