"""Plotting helpers for SSMProxy outputs."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Sequence


def _write_png(image: list[list[int]] | list[list[list[int]]], output_path: Path) -> None:
    """Minimal PNG writer for grayscale or RGB images using the standard library."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image:
        image = [[0]]

    if isinstance(image[0][0], list):
        # RGB image.
        height = len(image)
        width = len(image[0])
        color_type = 2
        raw_rows = []
        for row in image:
            flat = bytes(int(channel) & 0xFF for pixel in row for channel in pixel)
            raw_rows.append(b"\x00" + flat)
    else:
        # Grayscale image.
        height = len(image)
        width = len(image[0])
        color_type = 0
        raw_rows = [b"\x00" + bytes(int(pixel) & 0xFF for pixel in row) for row in image]

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    idat = zlib.compress(b"".join(raw_rows))
    png_bytes = signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    output_path.write_bytes(png_bytes)


def save_ssm_plot(ssm: Sequence[Sequence[float]], output_path: Path) -> None:
    """Save a heatmap-like visualization of the self-similarity matrix."""

    if not ssm:
        ssm = [[0.0]]

    max_val = max((max(row) for row in ssm), default=1.0)
    max_val = max(max_val, 1e-6)
    grayscale: list[list[int]] = []
    for row in ssm:
        grayscale.append([int(min(max(val / max_val, 0.0), 1.0) * 255) for val in row])

    _write_png(grayscale, output_path)


def save_novelty_plot(novelty: Sequence[float], peaks: Sequence[int], output_path: Path) -> None:
    """Save a compact novelty curve visualization with peak markers."""

    width = max(len(novelty), 1)
    height = 100
    canvas: list[list[list[int]]] = [
        [[255, 255, 255] for _ in range(width)] for _ in range(height)
    ]

    if novelty:
        max_val = max(max(novelty), 1e-6)
        normalized = [min(max(value / max_val, 0.0), 1.0) for value in novelty]
        for x, value in enumerate(normalized):
            bar_height = int(value * (height - 1))
            for y in range(height - bar_height - 1, height):
                canvas[y][x] = [200, 200, 200]

        for peak in peaks:
            if 0 <= peak < width:
                peak_height = int(normalized[peak] * (height - 1))
                y = height - peak_height - 1
                if 0 <= y < height:
                    canvas[y][peak] = [255, 0, 0]

    _write_png(canvas, output_path)
