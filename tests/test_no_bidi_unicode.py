from __future__ import annotations

import re
from pathlib import Path


BIDI_PATTERN = re.compile(r"[\u202A-\u202E\u2066-\u2069\u200E\u200F\u061C]")
TARGET_GLOBS: list[tuple[str, str]] = [
    ("src", "**/*.py"),
    ("tests", "**/*.py"),
    ("configs", "**/*.yaml"),
]


def _iter_target_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    paths: list[Path] = []
    for base, pattern in TARGET_GLOBS:
        base_path = root / base
        if base_path.exists():
            paths.extend(base_path.rglob(pattern))
    readme = root / "README.md"
    if readme.exists():
        paths.append(readme)
    return paths


def test_no_bidi_unicode_characters_present() -> None:
    offenders: list[str] = []
    for path in _iter_target_files():
        text = path.read_text(encoding="utf-8")
        for match in BIDI_PATTERN.finditer(text):
            codepoint = f"U+{ord(match.group()):04X}"
            offenders.append(f"{path}: {codepoint} at index {match.start()}")

    assert not offenders, "Found bidi control characters:\n" + "\n".join(offenders)
