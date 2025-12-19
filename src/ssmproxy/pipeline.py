"""End-to-end evaluation pipeline for MIDI directories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml

from .config import get_run_defaults
from .bar_features import compute_bar_features_from_path
from .lag import compute_lag_energy
from .metrics import PieceMetrics, build_piece_metrics, canonical_metrics_path, legacy_metrics_path, write_metrics_csv
from .novelty import NoveltyResult, compute_novelty
from .plots import save_novelty_plot, save_ssm_plot
from .ssm import compute_ssm

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

    def resolved_run_id(self) -> str:
        return self.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")


def _iter_midi_files(input_dir: Path) -> Iterable[Path]:
    exts = {".mid", ".midi"}
    midi_paths = (
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in exts
    )

    for path in sorted(midi_paths, key=lambda p: p.relative_to(input_dir).as_posix()):
        yield path


def _safe_compute_novelty(ssm: Sequence[Sequence[float]], L: int) -> NoveltyResult | None:
    if not ssm or any(len(row) != len(ssm) for row in ssm):
        return None

    size = len(ssm)
    tuned_L = max(1, min(L, max(1, size // 2)))
    return compute_novelty(ssm, L=tuned_L)


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

    for midi_path in _iter_midi_files(config.input_dir):
        piece_id, pch, onh = compute_bar_features_from_path(midi_path, exclude_drums=config.exclude_drums)
        ssm = compute_ssm(pch, onh, map_to_unit_interval=True)

        novelty = _safe_compute_novelty(ssm, L=config.novelty_L)
        lag_energy, best_lag, _ = compute_lag_energy(
            ssm, min_lag=config.lag_min_lag, top_k=config.lag_top_k, return_full=True
        )

        try:
            relative_path = midi_path.relative_to(config.input_dir)
        except ValueError:
            relative_path = midi_path
        group = relative_path.parts[0] if len(relative_path.parts) > 1 else ""

        metrics.append(
            build_piece_metrics(
                piece_id=piece_id,
                num_bars=len(pch),
                novelty=novelty,
                lag_energy=lag_energy,
                best_lag=best_lag,
                lag_min_lag=config.lag_min_lag,
                midi_path=str(relative_path.as_posix()),
                group=group,
                bars=len(pch),
            )
        )

        if ssm:
            save_ssm_plot(ssm, figures_dir / "ssm" / f"{piece_id}.png")
        if novelty:
            save_novelty_plot(novelty.novelty, novelty.peaks, figures_dir / "novelty" / f"{piece_id}.png")

    metrics_path = canonical_metrics_path(output_dir)
    write_metrics_csv(metrics_path, metrics, extra_paths=[legacy_metrics_path(output_dir)])
    return output_dir
