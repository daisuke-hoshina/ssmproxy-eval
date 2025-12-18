"""Minimal YAML writer stub for offline environments."""

from __future__ import annotations

import json
from typing import Any


def safe_dump(data: Any, stream) -> None:
    """Serialize data to a stream using JSON formatting as a YAML stand-in."""

    json.dump(data, stream, indent=2, default=str)
