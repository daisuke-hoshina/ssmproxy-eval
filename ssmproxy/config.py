"""Configuration loading utilities for ssmproxy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# Repository-level default configuration file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Optional path to a configuration file. When omitted, the
            repository default at ``configs/default.yaml`` is used.

    Returns:
        Parsed configuration dictionary (empty when the file is empty).
    """

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def get_run_defaults(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Extract evaluation defaults from the loaded configuration."""

    cfg = config if config is not None else load_config()
    evaluation_cfg = cfg.get("evaluation", {}) or {}
    lag_cfg = evaluation_cfg.get("lag", {}) or {}
    features_cfg = cfg.get("features", {}) or {}

    return {
        "output_root": evaluation_cfg.get("output_dir", "outputs"),
        "novelty_L": evaluation_cfg.get("novelty_L", 8),
        "lag_top_k": lag_cfg.get("top_k", 2),
        "lag_min_lag": lag_cfg.get("min_lag", 4),
        "exclude_drums": features_cfg.get("exclude_drums", True),
    }
