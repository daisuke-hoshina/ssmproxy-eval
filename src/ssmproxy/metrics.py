"""Aggregate evaluation metrics for MIDI pieces."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
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
) -> PieceMetrics:
    """Construct a :class:`PieceMetrics` instance with sensible defaults."""

    resolved_bars = num_bars if bars is None else bars
    novelty_stats = novelty.stats if novelty else {}
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
