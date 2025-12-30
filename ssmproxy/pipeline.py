"""End-to-end evaluation pipeline for MIDI directories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml
import statistics
import numpy as np

from .config import get_run_defaults
from .bar_features import compute_bar_features, compute_bar_features_from_path
from .midi_io import load_midi
from .lag import compute_lag_energy, estimate_base_period
from .metrics import (
    PieceMetrics,
    build_piece_metrics,
    canonical_metrics_path,
    legacy_metrics_path,
    write_metrics_csv,
)
from .novelty import NoveltyResult, compute_novelty
from .plots import save_novelty_plot, save_ssm_plot
from .ssm import compute_ssm
import json

RUN_DEFAULTS = get_run_defaults()


@dataclass
class RunConfig:
    """Configuration for an evaluation run."""

    input_dir: Path
    output_root: Path = Path(RUN_DEFAULTS["output_root"])
    run_id: str | None = None
    novelty_L: int = RUN_DEFAULTS["novelty_L"]
    lag_top_k: int = RUN_DEFAULTS["lag_top_k"]
    lag_min_lag: int = RUN_DEFAULTS["lag_min_lag"]
    exclude_drums: bool = RUN_DEFAULTS["exclude_drums"]
    ssm_weight_pch: float = RUN_DEFAULTS["ssm_weight_pch"]
    ssm_weight_onh: float = RUN_DEFAULTS["ssm_weight_onh"]
    ssm_map_to_unit_interval: bool = RUN_DEFAULTS["ssm_map_to_unit_interval"]
    novelty_peak_prominence: float = RUN_DEFAULTS["novelty_peak_prominence"]
    novelty_peak_min_distance: int | None = RUN_DEFAULTS["novelty_peak_min_distance"]
    require_4_4: bool = RUN_DEFAULTS["require_4_4"]
    max_bars: int | None = RUN_DEFAULTS["max_bars"]
    lag_max_lag: int | None = RUN_DEFAULTS["lag_max_lag"]
    lag_min_support: int | None = RUN_DEFAULTS["lag_min_support"]

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

    metrics: List[PieceMetrics] = []
    lag_energies_rows: List[dict] = []

    for midi_path in _iter_midi_files(config.input_dir):
        piece_id, midi = load_midi(midi_path)

        if config.require_4_4:
            if not _is_all_4_4(midi):
                # Skip non-4/4 pieces
                continue

        _, pch, onh = compute_bar_features(
            midi, piece_id, exclude_drums=config.exclude_drums, max_bars=config.max_bars
        )
        # piece_id, pch, onh = compute_bar_features_from_path(midi_path, exclude_drums=config.exclude_drums)
        ssm = compute_ssm(
            pch,
            onh,
            weight_pch=config.ssm_weight_pch,
            weight_onh=config.ssm_weight_onh,
            map_to_unit_interval=config.ssm_map_to_unit_interval,
        )

        novelty = _safe_compute_novelty(
            ssm,
            L=config.novelty_L,
            prominence=config.novelty_peak_prominence,
            min_distance=config.novelty_peak_min_distance,
        )
        lag_energy, best_lag, lag_energies = compute_lag_energy(
            ssm,
            min_lag=config.lag_min_lag,
            top_k=config.lag_top_k,
            return_full=True,
            max_lag=config.lag_max_lag,
            min_support=config.lag_min_support,
        )
        
        # Automated Hierarchy Index & Advanced Lag Metrics
        lag_base_period = estimate_base_period(lag_energies, min_lag=config.lag_min_lag)

        # 1. Calculate Fixed Metrics (needed for fallback)
        lag_e4: float | None = None
        lag_e8: float | None = None
        lag_hierarchy_index: float | None = None
        lag_mult4_std: float | None = None

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
        lag_hierarchy_auto_fallback: bool = False
        lag_e_l0: float | None = None
        lag_e_2l0: float | None = None
        lag_hierarchy_index_auto: float | None = None

        if lag_base_period is not None:
              def _safe_e(energies, idx):
                  if not energies or idx < 0 or idx >= len(energies):
                      return None
                  v = energies[idx]
                  return float(v) if v is not None else None

              max_valid_lag = 0
              for i, v in enumerate(lag_energies):
                  if v is not None:
                      max_valid_lag = i
              
              if lag_base_period > 0:
                  m = max_valid_lag // lag_base_period
                  if m >= 2:
                      k = m * lag_base_period
                      val_k = _safe_e(lag_energies, k)
                      val_l0 = _safe_e(lag_energies, lag_base_period)
                      if val_k is not None and val_l0 is not None:
                          lag_hierarchy_index_auto = val_k - val_l0
                          lag_hierarchy_mult = m
                          lag_hierarchy_lag_used = k
                          lag_e_l0 = val_l0
                          lag_e_2l0 = val_k 
                  else:
                      # Fallback
                      if lag_hierarchy_index is not None:
                          lag_hierarchy_index_auto = lag_hierarchy_index
                          lag_hierarchy_auto_fallback = True 


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
                "lag_min_support": config.lag_min_support,
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
                lag_e4=lag_e4,
                lag_e8=lag_e8,
                lag_hierarchy_index=lag_hierarchy_index,
                lag_mult4_std=lag_mult4_std,
            )
        )

        if ssm:
            save_ssm_plot(ssm, figures_dir / "ssm" / f"{piece_id}.png")
        if novelty:
            save_novelty_plot(novelty.novelty, novelty.peaks, figures_dir / "novelty" / f"{piece_id}.png")

    lag_energies_path = output_dir / "metrics" / "lag_energies.jsonl"
    lag_energies_path.parent.mkdir(parents=True, exist_ok=True)
    with lag_energies_path.open("w", encoding="utf-8") as fp:
        for row in lag_energies_rows:
            fp.write(json.dumps(row) + "\n")

    metrics_path = canonical_metrics_path(output_dir)
    write_metrics_csv(metrics_path, metrics, extra_paths=[legacy_metrics_path(output_dir)])
    return output_dir
