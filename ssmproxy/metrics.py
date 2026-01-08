"""Aggregate evaluation metrics for MIDI pieces."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import statistics
from typing import Iterable, Sequence

from .novelty import NoveltyResult


@dataclass
class PieceMetrics:
    """Container for per-piece summary statistics."""

    piece_id: str
    num_bars: int
    num_novelty_peaks: int
    novelty_peak_rate: float
    novelty_prom_mean: float
    novelty_prom_median: float
    novelty_interval_mean: float
    novelty_interval_cv: float
    lag_energy: float
    lag_best: int | None
    lag_min_lag: int
    
    # Advanced Lag Metrics
    lag_e4: float | None = None
    lag_e8: float | None = None
    lag_hierarchy_index: float | None = None
    lag_mult4_std: float | None = None
    
    # Auto Hierarchy Metrics
    lag_base_period: int | None = None
    lag_hierarchy_index_auto: float | None = None
    lag_e_l0: float | None = None
    lag_e_2l0: float | None = None
    lag_hierarchy_mult: int | None = None
    lag_hierarchy_lag_used: int | None = None
    lag_hierarchy_mult: int | None = None
    lag_hierarchy_lag_used: int | None = None
    lag_hierarchy_auto_fallback: bool = False
    lag_hierarchy_index_auto_valid: int = 0
    lag_hierarchy_index_auto_reason: str = ""
    
    # Advanced Novelty Metrics
    novelty_std: float | None = None
    novelty_tv: float | None = None
    novelty_topk_mean: float | None = None

    midi_path: str = ""
    group: str = ""
    bars: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


METRICS_COLUMNS: list[str] = [
    "piece_id",
    "num_bars",
    "num_novelty_peaks",
    "novelty_peak_rate",
    "novelty_prom_mean",
    "novelty_prom_median",
    "novelty_interval_mean",
    "novelty_interval_cv",
    "lag_energy",
    "lag_best",
    "lag_min_lag",
    "lag_e4",
    "lag_e8",
    "lag_hierarchy_index",
    "lag_mult4_std",
    "lag_base_period",
    "lag_hierarchy_index_auto",
    "lag_e_l0",
    "lag_e_2l0",
    "lag_hierarchy_mult",
    "lag_hierarchy_lag_used",
    "lag_hierarchy_auto_fallback",
    "lag_hierarchy_index_auto_valid",
    "lag_hierarchy_index_auto_reason",
    "novelty_std",
    "novelty_tv",
    "novelty_topk_mean",
    "midi_path",
    "group",
    "bars",
]


def build_piece_metrics(
    *,
    piece_id: str,
    num_bars: int,
    novelty: NoveltyResult | None,
    lag_energy: float,
    best_lag: int | None,
    lag_min_lag: int,
    midi_path: str = "",
    group: str = "",
    bars: int | None = None,
    # New args
    lag_e4: float | None = None,
    lag_e8: float | None = None,
    lag_hierarchy_index: float | None = None,
    lag_mult4_std: float | None = None,
    lag_base_period: int | None = None,
    lag_hierarchy_index_auto: float | None = None,
    lag_e_l0: float | None = None,
    lag_e_2l0: float | None = None,
    lag_hierarchy_mult: int | None = None,
    lag_hierarchy_lag_used: int | None = None,
    lag_hierarchy_auto_fallback: bool = False,
    lag_hierarchy_index_auto_valid: int = 0,
    lag_hierarchy_index_auto_reason: str = "",
) -> PieceMetrics:
    """Construct a :class:`PieceMetrics` instance with sensible defaults."""

    resolved_bars = num_bars if bars is None else bars
    novelty_stats = novelty.stats if novelty else {}
    
    # Compute advanced novelty metrics
    novelty_std = 0.0
    novelty_tv = 0.0
    novelty_topk_mean = 0.0
    
    if novelty and novelty.novelty:
        vals = novelty.novelty
        
        # Determining valid range from stats (fallback to full if missing)
        valid_start = int(novelty_stats.get("valid_start", 0))
        valid_end = int(novelty_stats.get("valid_end", len(vals)))
        
        # Safety clamp
        valid_start = max(0, min(valid_start, len(vals)))
        valid_end = max(valid_start, min(valid_end, len(vals)))
        
        vals_valid = vals[valid_start:valid_end]
        
        if len(vals_valid) > 1:
            novelty_std = float(statistics.stdev(vals_valid))
            # TV: mean(|n[i]-n[i-1]|)
            diffs = [abs(vals_valid[i] - vals_valid[i - 1]) for i in range(1, len(vals_valid))]
            novelty_tv = float(statistics.fmean(diffs)) if diffs else 0.0
            
        vals_sorted = sorted(vals_valid, reverse=True)
        top_k = vals_sorted[:5]
        novelty_topk_mean = float(statistics.fmean(top_k)) if top_k else 0.0
    
    return PieceMetrics(
        piece_id=piece_id,
        num_bars=num_bars,
        num_novelty_peaks=len(novelty.peaks) if novelty else 0,
        novelty_peak_rate=float(novelty_stats.get("peak_rate", 0.0)),
        novelty_prom_mean=float(novelty_stats.get("prom_mean", 0.0)),
        novelty_prom_median=float(novelty_stats.get("prom_median", 0.0)),
        novelty_interval_mean=float(novelty_stats.get("interval_mean", 0.0)),
        novelty_interval_cv=float(novelty_stats.get("interval_cv", 0.0)),
        lag_energy=float(lag_energy),
        lag_best=best_lag,
        lag_min_lag=lag_min_lag,
        
        lag_e4=lag_e4,
        lag_e8=lag_e8,
        lag_hierarchy_index=lag_hierarchy_index,
        lag_mult4_std=lag_mult4_std,
        
        lag_base_period=lag_base_period,
        lag_hierarchy_index_auto=lag_hierarchy_index_auto,
        lag_e_l0=lag_e_l0,
        lag_e_2l0=lag_e_2l0,
        lag_hierarchy_mult=lag_hierarchy_mult,
        lag_hierarchy_lag_used=lag_hierarchy_lag_used,
        lag_hierarchy_auto_fallback=lag_hierarchy_auto_fallback,
        lag_hierarchy_index_auto_valid=lag_hierarchy_index_auto_valid,
        lag_hierarchy_index_auto_reason=lag_hierarchy_index_auto_reason,
        
        novelty_std=novelty_std if novelty else 0.0,
        novelty_tv=novelty_tv if novelty else 0.0,
        novelty_topk_mean=novelty_topk_mean if novelty else 0.0,
        
        midi_path=midi_path,
        group=group,
        bars=resolved_bars,
    )


CANONICAL_METRICS_REL_PATH = Path("metrics") / "ssm_proxy.csv"
LEGACY_METRICS_FILENAME = Path("metrics.csv")


def canonical_metrics_path(run_dir: Path) -> Path:
    """Return the canonical metrics path under a run directory."""

    return run_dir / CANONICAL_METRICS_REL_PATH


def legacy_metrics_path(run_dir: Path) -> Path:
    """Return the legacy metrics path under a run directory."""

    return run_dir / LEGACY_METRICS_FILENAME


def _format_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _sorted_metrics(rows: Sequence[PieceMetrics]) -> list[PieceMetrics]:
    return sorted(rows, key=lambda row: (row.midi_path or row.piece_id))


def write_metrics_csv(output_path: Path, rows: Iterable[PieceMetrics], *, extra_paths: Iterable[Path] | None = None) -> None:
    """Write metrics to CSV with a stable column order.

    Args:
        output_path: Primary path to write.
        rows: Metrics rows to persist.
        extra_paths: Optional additional paths to write the same CSV to (for backward compatibility).
    """

    rows_list: list[PieceMetrics] = _sorted_metrics(list(rows))

    paths = [output_path]
    if extra_paths:
        paths.extend(extra_paths)

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=METRICS_COLUMNS)
            writer.writeheader()
            for row in rows_list:
                row_dict = row.as_dict()
                writer.writerow({col: _format_value(row_dict.get(col)) for col in METRICS_COLUMNS})
