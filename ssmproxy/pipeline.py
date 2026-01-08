"""End-to-end evaluation pipeline for MIDI directories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml
import statistics
import numpy as np

from .config import get_run_defaults
from .bar_features import compute_bar_features, compute_bar_features_from_path
from .midi_io import load_midi
from .lag import (
    compute_lag_energy, 
    compute_lag_prominence,
    estimate_base_period_comb,
    compute_hierarchy_index_auto_slope,
    compute_hierarchy_index_auto_adjdiff
)
from .metrics import (
    PieceMetrics,
    build_piece_metrics,
    canonical_metrics_path,
    legacy_metrics_path,
    write_metrics_csv,
)
from .novelty import NoveltyResult, compute_novelty, compute_novelty_multiscale
from .plots import save_novelty_plot, save_ssm_plot
from .plots import save_novelty_plot, save_ssm_plot
from .ssm import compute_ssm, compute_ssm_multi
import json
from typing import Dict, Any


RUN_DEFAULTS = get_run_defaults()


def resolve_auto_hierarchy_metrics(
    prominence: Sequence[float | None],
    lag_base_period: int | None,
    fallback_mode: bool,
    fallback_index: float | None,
    auto_mode: str,
    max_levels: int,
) -> tuple[float | None, int | None, int | None, int, str]: # index, mult, lag_used, valid, reason
    """Resolve auto hierarchy metrics, preserving None for missing values."""
    
    # If no base period (failed estim), we can't do auto logic unless fallback mode handles it?
    # Actually if fallback_mode is True, lag_base_period might be None or best_lag.
    
    if fallback_mode:
        # Fallback case: use fixed index (e.g. E8 - E4)
        if fallback_index is not None:
             return fallback_index, None, None, 1, "fallback_ok"
        else:
             return None, None, None, 0, "fallback_no_fixed_index"

    if lag_base_period is None:
        return None, None, None, 0, "no_base_period"

    # Calculate robust index
    if auto_mode == "adjdiff_prom":
        idx_val = compute_hierarchy_index_auto_adjdiff(
            prominence, 
            lag_base_period, 
            max_levels=max_levels
        )
    else:
        idx_val = compute_hierarchy_index_auto_slope(
            prominence, 
            lag_base_period, 
            max_levels=max_levels
        )
    
    # idx_val can be None. We return it as is (do NOT default to 0.0).
    auto_idx = idx_val
    valid = 1 if auto_idx is not None else 0
    reason = "ok" if auto_idx is not None else "insufficient_points"
    
    # Populate meta-info
    mult = None
    lag_used = None
    
    max_m = 0
    
    def _safe_get(arr, i):
        if 0 <= i < len(arr):
            return arr[i]
        return None

    for level in range(max_levels):
        m = 2**level
        target = lag_base_period * m
        if _safe_get(prominence, target) is not None:
            max_m = m
    
    if max_m > 0:
        mult = max_m
        lag_used = lag_base_period * max_m
        
    return auto_idx, mult, lag_used, valid, reason



@dataclass
class RunConfig:
    """Configuration for an evaluation run."""

    input_dir: Path
    output_root: Path = Path(RUN_DEFAULTS["output_root"])
    run_id: str | None = None
    novelty_L: int = RUN_DEFAULTS["novelty_L"]
    novelty_multi_Ls: list[int] | None = field(
        default_factory=lambda: list(RUN_DEFAULTS["novelty_multi_Ls"]) if RUN_DEFAULTS["novelty_multi_Ls"] else None
    )
    lag_top_k: int = RUN_DEFAULTS["lag_top_k"]
    lag_min_lag: int = RUN_DEFAULTS["lag_min_lag"]
    exclude_drums: bool = RUN_DEFAULTS["exclude_drums"]
    ssm_weight_pch: float = RUN_DEFAULTS["ssm_weight_pch"]
    ssm_weight_onh: float = RUN_DEFAULTS["ssm_weight_onh"]
    ssm_map_to_unit_interval: bool = RUN_DEFAULTS["ssm_map_to_unit_interval"]
    novelty_peak_prominence: float = RUN_DEFAULTS["novelty_peak_prominence"]
    novelty_peak_min_distance: int | None = RUN_DEFAULTS["novelty_peak_min_distance"]
    novelty_peaks_mode: str = RUN_DEFAULTS["novelty_peaks_mode"]
    novelty_consensus_min_scales: int = RUN_DEFAULTS["novelty_consensus_min_scales"]
    novelty_consensus_tolerance: int = RUN_DEFAULTS["novelty_consensus_tolerance"]
    novelty_consensus_keep_lmax: bool = RUN_DEFAULTS["novelty_consensus_keep_lmax"]
    novelty_consensus_fallback: bool = RUN_DEFAULTS["novelty_consensus_fallback"]
    require_4_4: bool = RUN_DEFAULTS["require_4_4"]
    max_bars: int | None = RUN_DEFAULTS["max_bars"]
    lag_max_lag: int | None = RUN_DEFAULTS["lag_max_lag"]
    lag_min_support: int | None = RUN_DEFAULTS["lag_min_support"]
    lag_hierarchy_auto_mode: str = RUN_DEFAULTS["lag_hierarchy_auto_mode"]
    lag_hierarchy_auto_max_levels: int = RUN_DEFAULTS["lag_hierarchy_auto_max_levels"]
    lag_hierarchy_auto_harmonics: list[int] = field(default_factory=lambda: list(RUN_DEFAULTS["lag_hierarchy_auto_harmonics"]))
    lag_hierarchy_auto_weights: list[float] = field(default_factory=lambda: list(RUN_DEFAULTS["lag_hierarchy_auto_weights"]))
    lag_hierarchy_auto_prom_window: int | str = RUN_DEFAULTS["lag_hierarchy_auto_prom_window"]
    lag_best_lag_mode: str = RUN_DEFAULTS["lag_best_lag_mode"]
    lag_best_lag_lcb_z: float = RUN_DEFAULTS["lag_best_lag_lcb_z"]
    lag_min_support_ratio: float | None = RUN_DEFAULTS["lag_min_support_ratio"]
    lag_l0_tau: float = RUN_DEFAULTS["lag_l0_tau"]
    lag_l0_min_hits: int = RUN_DEFAULTS["lag_l0_min_hits"]
    
    quantize_mode: str = RUN_DEFAULTS["quantize_mode"]
    analysis_beats_per_bar: int = RUN_DEFAULTS["analysis_beats_per_bar"]
    steps_per_beat: int = RUN_DEFAULTS["steps_per_beat"]
    feature_mode: str = RUN_DEFAULTS["feature_mode"]
    ssm_weights: Dict[str, float] = field(default_factory=lambda: dict(RUN_DEFAULTS["ssm_weights"]))
    fail_fast: bool = False # Task A: Flag to re-raise errors

    def resolved_run_id(self) -> str:
        return self.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")


def _iter_midi_files(input_dir: Path) -> Iterable[Path]:
    exts = {".mid", ".midi"}
    midi_paths = (
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in exts
    )

    for path in sorted(midi_paths, key=lambda p: p.relative_to(input_dir).as_posix()):
        yield path


def _safe_compute_novelty(
    ssm: Sequence[Sequence[float]], L: int, prominence: float, min_distance: int | None
) -> NoveltyResult | None:
    if not ssm or any(len(row) != len(ssm) for row in ssm):
        return None

    size = len(ssm)
    tuned_L = max(1, min(L, max(1, size // 2)))
    tuned_L = max(1, min(L, max(1, size // 2)))
    return compute_novelty(ssm, L=tuned_L, prominence=prominence, min_distance=min_distance)


def _is_all_4_4(midi) -> bool:
    tsc = getattr(midi, "time_signature_changes", None) or []
    if not tsc:
        return True
    return all(ts.numerator == 4 and ts.denominator == 4 for ts in tsc)


def run_evaluation(config: RunConfig) -> Path:
    """Run the evaluation pipeline and return the output directory."""
    
    # Task D: Validation
    # Task D: Validation
    if config.feature_mode == "enhanced" and config.quantize_mode == "legacy_fixed_tempo":
        raise ValueError("Feature mode 'enhanced' requires quantize_mode='beat_grid'. Please set quantize_mode='beat_grid' or feature_mode='basic'.")

    resolved_config: dict = {
        key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(config).items()
    }
    run_id = config.resolved_run_id()
    output_dir = config.output_root / run_id
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist resolved configuration snapshot for reproducibility.
    config_snapshot_path = output_dir / "config.yaml"
    with config_snapshot_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(resolved_config | {"run_id": run_id}, fp)

    # Task C: Strong Validation
    if config.analysis_beats_per_bar < 1:
        raise ValueError(f"analysis_beats_per_bar must be >= 1, got {config.analysis_beats_per_bar}")
    if config.steps_per_beat < 1:
        raise ValueError(f"steps_per_beat must be >= 1, got {config.steps_per_beat}")
    if sum(config.ssm_weights.values()) <= 0:
        raise ValueError("Total SSM weights must be positive")

    # Task B: Observability Counters
    counters = {
        "total_files": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0
    }

    metrics: List[PieceMetrics] = []
    lag_energies_rows: List[dict] = []
    errors: List[dict] = []

    for midi_path in _iter_midi_files(config.input_dir):
        counters["total_files"] += 1
        
        # Log progress
        if counters["total_files"] % 50 == 0:
             print(f"Propcessing: {counters['total_files']} files... "
                   f"(Success: {counters['processed']}, Skipped: {counters['skipped']}, Failed: {counters['failed']})")

        # Use current iteration tracking for robust logging
        current_piece_id = midi_path.stem 
        # Fallback if stem fails for some reason (unlikely with Path)
        if not current_piece_id:
             current_piece_id = "unknown"

        try:
            piece_id, midi = load_midi(midi_path)
            # Update current_piece_id with the one from load_midi which is authoritative
            current_piece_id = piece_id
    
            if config.require_4_4:
                if not _is_all_4_4(midi):
                    # Skip non-4/4 pieces
                    counters["skipped"] += 1
                    continue
        
            _, features = compute_bar_features(
                midi,
                piece_id,
                exclude_drums=config.exclude_drums,
                max_bars=config.max_bars,
                quantize_mode=config.quantize_mode,
                analysis_beats_per_bar=config.analysis_beats_per_bar,
                steps_per_beat=config.steps_per_beat,
                feature_mode=config.feature_mode,
            )
            
            # Task 3: Skip empty pieces
            if not features.get("pch"):
                # If pch is empty list, num_bars=0.
                counters["skipped"] += 1
                continue
            
            pch = features["pch"]
            onh = features["onh"]
            
            if config.feature_mode == "enhanced":
                ssm = compute_ssm_multi(
                    features,
                    weights=config.ssm_weights,
                    map_to_unit_interval=config.ssm_map_to_unit_interval,
                    strict=True,
                )
            else:
                ssm = compute_ssm(
                    pch,
                    onh,
                    weight_pch=config.ssm_weight_pch,
                    weight_onh=config.ssm_weight_onh,
                    map_to_unit_interval=config.ssm_map_to_unit_interval,
                )
    
            if config.novelty_multi_Ls:
                # Multi-scale novelty
                novelty = compute_novelty_multiscale(
                    ssm,
                    Ls=config.novelty_multi_Ls,
                    prominence=config.novelty_peak_prominence,
                    min_distance=config.novelty_peak_min_distance,
                    mode=config.novelty_peaks_mode,
                    consensus_min_scales=config.novelty_consensus_min_scales,
                    consensus_tolerance=config.novelty_consensus_tolerance,
                    keep_lmax=config.novelty_consensus_keep_lmax,
                    fallback=config.novelty_consensus_fallback,
                )
            else:
                 # Single-scale novelty (legacy)
                 novelty = _safe_compute_novelty(
                    ssm,
                    L=config.novelty_L,
                    prominence=config.novelty_peak_prominence,
                    min_distance=config.novelty_peak_min_distance,
                )
            # Determine effective min_support
            B = len(ssm)
            min_support_effective = config.lag_min_support
            
            if config.lag_min_support_ratio is not None:
                from math import ceil
                ratio_support = ceil(config.lag_min_support_ratio * B)
                min_support_effective = max(min_support_effective or 0, ratio_support)
                
            lag_energy, best_lag, lag_energies = compute_lag_energy(
                ssm,
                min_lag=config.lag_min_lag,
                top_k=config.lag_top_k,
                return_full=True,
                max_lag=config.lag_max_lag,
                min_support=min_support_effective,
                best_lag_mode=config.lag_best_lag_mode,
                best_lag_lcb_z=config.lag_best_lag_lcb_z,
            )
            
            # Automated Hierarchy Index & Advanced Lag Metrics
            # Compute Prominence
            prominence = compute_lag_prominence(
                lag_energies,
                window=config.lag_hierarchy_auto_prom_window
            )
    
            lag_base_period, lag_hierarchy_auto_fallback = estimate_base_period_comb(
                lag_energies, 
                prominence, 
                min_lag=config.lag_min_lag,
                harmonics=config.lag_hierarchy_auto_harmonics,
                weights=config.lag_hierarchy_auto_weights,
                tau=config.lag_l0_tau,
                min_hits=config.lag_l0_min_hits,
            )
    
            # 1. Calculate Fixed Metrics (needed for fallback)
            lag_e4: float | None = None
            lag_e8: float | None = None
            lag_hierarchy_index: float | None = None
            lag_mult4_std: float | None = None
            
            lag_hierarchy_index_auto: float | None = None
            lag_hierarchy_mult: int | None = None
            lag_hierarchy_lag_used: int | None = None
            lag_hierarchy_index_auto_valid: int = 0
            lag_hierarchy_index_auto_reason: str = "no_lag_energies"
            lag_e_l0: float | None = None
            lag_e_2l0: float | None = None
    
            if lag_energies: 
                def get_energy(lag: int) -> float | None:
                    if 0 <= lag < len(lag_energies):
                        val = lag_energies[lag]
                        return float(val) if val is not None else None
                    return None
    
                lag_e4 = get_energy(4)
                lag_e8 = get_energy(8)
                
                if lag_e4 is not None and lag_e8 is not None:
                    lag_hierarchy_index = lag_e8 - lag_e4
                
                mult4_vals = []
                for lag in range(4, len(lag_energies), 4):
                     val = get_energy(lag)
                     if val is not None:
                         mult4_vals.append(val)
                
                if len(mult4_vals) > 1:
                    lag_mult4_std = float(statistics.stdev(mult4_vals))
                elif mult4_vals:
                    lag_mult4_std = 0.0
    
            # 2. Calculate Automated Metrics with Generalization and Fallback
            lag_hierarchy_mult: int | None = None
            lag_hierarchy_lag_used: int | None = None
            lag_e_l0: float | None = None
            lag_e_2l0: float | None = None
            lag_hierarchy_index_auto: float | None = None
    
            if lag_base_period is not None:
                  def _safe_e(energies, idx):
                      if not energies or idx < 0 or idx >= len(energies):
                          return None
                      v = energies[idx]
                      return float(v) if v is not None else None
    
                  val_l0 = _safe_e(lag_energies, lag_base_period)
                  lag_e_l0 = val_l0
                  
                  val_2l0 = _safe_e(lag_energies, lag_base_period * 2)
                  lag_e_2l0 = val_2l0
                  
                  lag_e_2l0 = val_2l0
                  
                  # Calculate Hierarchy Index (Refactored)
                  (
                      lag_hierarchy_index_auto, 
                      lag_hierarchy_mult, 
                      lag_hierarchy_lag_used,
                      lag_hierarchy_index_auto_valid,
                      lag_hierarchy_index_auto_reason
                  ) = resolve_auto_hierarchy_metrics(
                      prominence,
                      lag_base_period,
                      lag_hierarchy_auto_fallback,
                      lag_hierarchy_index,
                      config.lag_hierarchy_auto_mode,
                      config.lag_hierarchy_auto_max_levels
                  ) 

            try:
                relative_path = midi_path.relative_to(config.input_dir)
            except ValueError:
                relative_path = midi_path
            group = relative_path.parts[0] if len(relative_path.parts) > 1 else ""

            lag_energies_rows.append(
                {
                    "piece_id": piece_id,
                    "group": group,
                    "midi_path": relative_path.as_posix(),
                    "num_bars": len(pch),
                    "lag_min_lag": config.lag_min_lag,
                    "lag_top_k": config.lag_top_k,
                    "lag_max_lag": config.lag_max_lag,
                    "lag_min_support": min_support_effective,
                    "lag_best_lag_mode": config.lag_best_lag_mode,
                    "lag_energies": lag_energies,
                }
            )

            metrics.append(
                build_piece_metrics(
                    piece_id=piece_id,
                    num_bars=len(pch),
                    novelty=novelty,
                    lag_energy=lag_energy,
                    best_lag=best_lag,
                    lag_min_lag=config.lag_min_lag,
                    midi_path=relative_path.as_posix(),
                    group=group,
                    bars=len(pch),
                    
                    lag_base_period=lag_base_period,
                    lag_hierarchy_index_auto=lag_hierarchy_index_auto,
                    lag_e_l0=lag_e_l0,
                    lag_e_2l0=lag_e_2l0,
                    lag_hierarchy_mult=lag_hierarchy_mult,
                    lag_hierarchy_lag_used=lag_hierarchy_lag_used,
                    lag_hierarchy_auto_fallback=lag_hierarchy_auto_fallback,
                    
                    # Advanced Lag Metrics
                    lag_e4=lag_e4,
                    lag_e8=lag_e8,
                    lag_hierarchy_index=lag_hierarchy_index,
                    lag_mult4_std=lag_mult4_std,
                )
            )
            counters["processed"] += 1
    
            if ssm:
                save_ssm_plot(ssm, figures_dir / "ssm" / f"{piece_id}.png")
            if novelty:
                save_novelty_plot(novelty.novelty, novelty.peaks, figures_dir / "novelty" / f"{piece_id}.png")
                
        except Exception as e:
            if config.fail_fast:
                raise # bare raise per request
            
            # Log failure
            counters["failed"] += 1
            import traceback
            
            error_rec = {
                "piece_id": current_piece_id,
                "midi_path": str(midi_path),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            # We need a place to store errors. Let's add 'errors' list parallel to metrics.
            # But here we are inside loop.
            # We can write to file properly later if we store it.
            # Define errors list outside loop.
            errors.append(error_rec) # Need to init errors list before loop
            continue


    lag_energies_path = output_dir / "metrics" / "lag_energies.jsonl"
    lag_energies_path.parent.mkdir(parents=True, exist_ok=True)
    with lag_energies_path.open("w", encoding="utf-8") as fp:
        for row in lag_energies_rows:
            fp.write(json.dumps(row) + "\n")

    errors_path = output_dir / "metrics" / "errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as fp:
        for err in errors:
            fp.write(json.dumps(err) + "\n")
            
    # Task B: Write Run Summary
    summary_path = output_dir / "metrics" / "run_summary.json"
    run_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "counts": counters,
        "output_dir": str(output_dir)
    }
    
    def _json_default(obj):
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(run_summary, fp, indent=2, default=_json_default)

    metrics_path = canonical_metrics_path(output_dir)
    write_metrics_csv(metrics_path, metrics, extra_paths=[legacy_metrics_path(output_dir)])
    return output_dir
