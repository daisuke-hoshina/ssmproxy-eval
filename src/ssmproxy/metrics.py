"""Aggregate evaluation metrics for MIDI pieces."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List

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

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_piece_metrics(
    *,
    piece_id: str,
    num_bars: int,
    novelty: NoveltyResult | None,
    lag_energy: float,
    best_lag: int | None,
) -> PieceMetrics:
    """Construct a :class:`PieceMetrics` instance with sensible defaults."""

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
    )


def write_metrics_csv(output_path: Path, rows: Iterable[PieceMetrics]) -> None:
    """Write metrics to CSV with a stable column order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list: List[PieceMetrics] = list(rows)
    columns = list(PieceMetrics.__annotations__.keys())

    lines = [",".join(columns)]
    for row in rows_list:
        values = []
        row_dict = row.as_dict()
        for column in columns:
            value = row_dict[column]
            values.append("" if value is None else str(value))
        lines.append(",".join(values))

    output_path.write_text("\n".join(lines) + "\n")
