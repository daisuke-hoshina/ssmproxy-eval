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


def _resolve_ssm_weights(ssm_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Resolve SSM weights preventing double counting."""
    # Prefer 'weight_onh_bin' but fallback to 'weight_onh'
    w_onh_bin = ssm_cfg.get("weight_onh_bin")
    w_onh = ssm_cfg.get("weight_onh")
    
    # Logic: if onh_bin is set, use it. Else if onh is set, use it as onh_bin. Default 0.1
    final_onh_bin = 0.1
    if w_onh_bin is not None:
        final_onh_bin = float(w_onh_bin)
    elif w_onh is not None:
        final_onh_bin = float(w_onh)
        
    weights = {
        "pch": float(ssm_cfg.get("weight_pch", 0.3)),
        "onh_bin": final_onh_bin,
        "onh_count": float(ssm_cfg.get("weight_onh_count", 0.2)),
        "chroma_roll": float(ssm_cfg.get("weight_chroma_roll", 0.4)),
        "density": float(ssm_cfg.get("weight_density", 0.0)),
    }
    return weights


def get_run_defaults(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Extract evaluation defaults from the loaded configuration."""

    cfg = config if config is not None else load_config()
    evaluation_cfg = cfg.get("evaluation", {}) or {}
    lag_cfg = evaluation_cfg.get("lag", {}) or {}
    ssm_cfg = evaluation_cfg.get("ssm", {}) or {}
    novelty_peaks_cfg = evaluation_cfg.get("novelty_peaks", {}) or {}
    features_cfg = cfg.get("features", {}) or {}

    return {
        "output_root": evaluation_cfg.get("output_dir", "outputs"),
        "novelty_L": evaluation_cfg.get("novelty_L", 8),
        "novelty_multi_Ls": evaluation_cfg.get("novelty_multi_Ls", None),
        "lag_top_k": lag_cfg.get("top_k", 2),
        "lag_min_lag": lag_cfg.get("min_lag", 4),
        "lag_max_lag": lag_cfg.get("max_lag", None),
        "lag_min_support": lag_cfg.get("min_support", None),
        "lag_hierarchy_auto_mode": lag_cfg.get("hierarchy_auto_mode", "slope_prom"),
        "lag_hierarchy_auto_max_levels": lag_cfg.get("hierarchy_auto_max_levels", 5),
        "lag_hierarchy_auto_harmonics": lag_cfg.get("hierarchy_auto_harmonics", [1, 2, 4, 8]),
        "lag_hierarchy_auto_weights": lag_cfg.get("hierarchy_auto_weights", [1.0, 0.8, 0.6, 0.5]),
        "lag_hierarchy_auto_prom_window": lag_cfg.get("hierarchy_auto_prom_window", "adaptive"),
        "lag_best_lag_mode": lag_cfg.get("best_lag_mode", "mean"),
        "lag_best_lag_lcb_z": float(lag_cfg.get("best_lag_lcb_z", 1.0)),
        "lag_min_support_ratio": lag_cfg.get("min_support_ratio", None),
        "lag_l0_tau": float(lag_cfg.get("l0_tau", 32.0)),
        "lag_l0_min_hits": int(lag_cfg.get("l0_min_hits", 2)),
        "exclude_drums": features_cfg.get("exclude_drums", True),
        "quantize_mode": features_cfg.get("quantize_mode", "beat_grid"),
        "analysis_beats_per_bar": features_cfg.get("analysis_beats_per_bar", 4),
        "steps_per_beat": features_cfg.get("steps_per_beat", 4),
        "feature_mode": features_cfg.get("feature_mode", "enhanced"),
        "ssm_weight_pch": ssm_cfg.get("weight_pch", 0.5), # Keep for legacy
        "ssm_weight_onh": ssm_cfg.get("weight_onh", 0.5), # Keep for legacy
        "ssm_weights": _resolve_ssm_weights(ssm_cfg),
        "ssm_map_to_unit_interval": ssm_cfg.get("map_to_unit_interval", True),
        "novelty_peak_prominence": novelty_peaks_cfg.get("prominence", 0.10),
        "novelty_peak_min_distance": novelty_peaks_cfg.get("min_distance", None),
        "novelty_peaks_mode": novelty_peaks_cfg.get("mode", "integrated"),
        "novelty_consensus_min_scales": novelty_peaks_cfg.get("consensus_min_scales", 2),
        "novelty_consensus_tolerance": novelty_peaks_cfg.get("consensus_tolerance", 2),
        "novelty_consensus_keep_lmax": novelty_peaks_cfg.get("consensus_keep_if_in_lmax", True),
        "novelty_consensus_fallback": novelty_peaks_cfg.get("consensus_fallback_to_integrated", True),
        "require_4_4": evaluation_cfg.get("require_4_4", False),
        "max_bars": evaluation_cfg.get("max_bars", None),
    }
