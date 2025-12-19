"""Reporting utilities for aggregating evaluation outputs without external dependencies."""

from __future__ import annotations

import csv
import importlib.util
import logging
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from .metrics import canonical_metrics_path, legacy_metrics_path
from .plots import _write_png

LOGGER = logging.getLogger(__name__)

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    import matplotlib  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    matplotlib.use("Agg")
else:
    plt = None


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        rows = [row for row in reader]
    return rows, reader.fieldnames or []


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_metrics(metrics_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load the metrics CSV into a list of dictionaries and return rows + columns."""

    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_csv}")
    return _read_csv(metrics_csv)


def resolve_metrics_csv(eval_out: Path, metrics_csv: Path | None = None) -> Path:
    """Resolve the metrics CSV path, preferring the canonical location.

    Args:
        eval_out: Root output directory for the evaluation run.
        metrics_csv: Optional override path provided via CLI.

    Returns:
        Path to the metrics CSV to load. If the canonical file is missing, the legacy path is used
        when it exists; otherwise the canonical path is returned to surface a clear error message.
    """

    if metrics_csv:
        return metrics_csv

    canonical_path = canonical_metrics_path(eval_out)
    if canonical_path.is_file():
        return canonical_path

    legacy_path = legacy_metrics_path(eval_out)
    if legacy_path.is_file():
        LOGGER.warning("Canonical metrics not found at %s; falling back to legacy path %s", canonical_path, legacy_path)
        return legacy_path

    return canonical_path


def join_manifest(metrics: list[dict[str, str]], manifest_path: Path | None, *, columns: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    """Join manifest metadata on piece_id when provided."""

    if not manifest_path:
        return metrics, columns
    if not manifest_path.is_file():
        LOGGER.warning("Manifest file not found at %s; skipping join.", manifest_path)
        return metrics, columns

    manifest_rows, manifest_fields = _read_csv(manifest_path)
    if "piece_id" not in manifest_fields:
        LOGGER.warning("Manifest at %s missing piece_id column; skipping join.", manifest_path)
        return metrics, columns

    manifest_lookup = {row["piece_id"]: row for row in manifest_rows}
    joined: list[dict[str, str]] = []
    new_columns = columns + [field for field in manifest_fields if field not in columns]
    for row in metrics:
        merged = dict(row)
        extra = manifest_lookup.get(row.get("piece_id", ""), {})
        for key, value in extra.items():
            if key not in merged:
                merged[key] = value
        joined.append(merged)
    return joined, new_columns


def _is_numeric(value: str | object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _numeric_columns(rows: list[dict[str, str]], exclude: set[str]) -> list[str]:
    if not rows:
        return []
    columns: list[str] = []
    for key in rows[0].keys():
        if key in exclude:
            continue
        if all((value == "" or _is_numeric(value)) for value in (row.get(key, "") for row in rows)):
            columns.append(key)
    return columns


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_vals[int(pos)]
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * (pos - lower)


def compute_group_stats(rows: list[dict[str, str]], group_col: str) -> tuple[list[dict[str, object]], list[str]]:
    """Aggregate metrics per group with summary statistics."""

    if not rows or group_col not in rows[0]:
        raise ValueError(f"Group column '{group_col}' not found in metrics.")

    numeric_cols = _numeric_columns(rows, exclude={group_col})
    if not numeric_cols:
        raise ValueError("No numeric metric columns found to aggregate.")

    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        group_value = row.get(group_col, "")
        group_bucket = grouped.setdefault(group_value, {col: [] for col in numeric_cols})
        for col in numeric_cols:
            value = row.get(col, "")
            if value == "" or not _is_numeric(value):
                continue
            group_bucket[col].append(float(value))

    result_rows: list[dict[str, object]] = []
    stats_columns: list[str] = [group_col]
    stat_order = ["mean", "std", "median", "q25", "q75", "count"]
    for col in numeric_cols:
        stats_columns.extend([f"{col}_{stat}" for stat in stat_order])

    for group_value, metrics in grouped.items():
        entry: dict[str, object] = {group_col: group_value}
        for col, values in metrics.items():
            count = len(values)
            mean = statistics.fmean(values) if values else 0.0
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            median = statistics.median(values) if values else 0.0
            entry[f"{col}_mean"] = mean
            entry[f"{col}_std"] = std
            entry[f"{col}_median"] = median
            entry[f"{col}_q25"] = _quantile(values, 0.25)
            entry[f"{col}_q75"] = _quantile(values, 0.75)
            entry[f"{col}_count"] = count
        result_rows.append(entry)

    return result_rows, stats_columns


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _placeholder_image(width: int, height: int, fill: int = 220) -> list[list[int]]:
    return [[fill for _ in range(width)] for _ in range(height)]


def _save_placeholder(output_path: Path, label: str) -> None:
    image = _placeholder_image(240, 120)
    _write_png(image, output_path)
    LOGGER.warning("Matplotlib unavailable; wrote placeholder figure for %s", label)


def _collect_values(rows: list[dict[str, str]], group_col: str, metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        group_value = row.get(group_col, "")
        grouped.setdefault(group_value, [])
        value = row.get(metric, "")
        if value != "" and _is_numeric(value):
            grouped[group_value].append(float(value))
    return grouped


def plot_boxplots(rows: list[dict[str, str]], group_col: str, metric_names: list[str], output_dir: Path) -> list[Path]:
    figure_paths: list[Path] = []
    _ensure_dir(output_dir)
    if not metric_names:
        LOGGER.warning("No metrics available for boxplots; skipping.")
        return figure_paths

    for metric in metric_names:
        output_path = output_dir / f"boxplot_{metric}.png"
        grouped = _collect_values(rows, group_col, metric)
        if plt is None:
            _save_placeholder(output_path, f"boxplot {metric}")
            figure_paths.append(output_path)
            continue

        labels = list(grouped.keys())
        data = [grouped[label] for label in labels]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, labels=labels)
        ax.set_title(f"{metric} by {group_col}")
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        figure_paths.append(output_path)
    return figure_paths


def plot_bars(rows: list[dict[str, str]], group_col: str, metric_names: list[str], output_dir: Path) -> list[Path]:
    figure_paths: list[Path] = []
    _ensure_dir(output_dir)
    if not metric_names:
        LOGGER.warning("No metrics available for bar plots; skipping.")
        return figure_paths

    for metric in metric_names:
        output_path = output_dir / f"bar_{metric}.png"
        grouped = _collect_values(rows, group_col, metric)
        if plt is None:
            _save_placeholder(output_path, f"bar {metric}")
            figure_paths.append(output_path)
            continue

        labels = list(grouped.keys())
        means = [statistics.fmean(vals) if vals else 0.0 for vals in grouped.values()]
        stds = [statistics.stdev(vals) if len(vals) > 1 else 0.0 for vals in grouped.values()]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} mean by {group_col}")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        figure_paths.append(output_path)
    return figure_paths


def plot_scatter(
    rows: list[dict[str, str]],
    group_col: str,
    novelty_metric_candidates: list[str],
    lag_metric_candidates: list[str],
    output_dir: Path,
) -> Path | None:
    _ensure_dir(output_dir)
    novelty_metric = next((m for m in novelty_metric_candidates if m in rows[0]), None) if rows else None
    lag_metric = next((m for m in lag_metric_candidates if m in rows[0]), None) if rows else None

    if not novelty_metric or not lag_metric:
        LOGGER.warning("Skipping scatter plot; required metrics missing.")
        return None

    output_path = output_dir / "scatter_novelty_vs_lag.png"
    grouped_novelty = _collect_values(rows, group_col, novelty_metric)
    grouped_lag = _collect_values(rows, group_col, lag_metric)
    if plt is None:
        _save_placeholder(output_path, "scatter novelty vs lag")
        return output_path

    fig, ax = plt.subplots(figsize=(6, 4))
    for group_value in grouped_novelty.keys():
        xs = grouped_novelty.get(group_value, [])
        ys = grouped_lag.get(group_value, [])
        if not xs or not ys:
            continue
        ax.scatter(xs, ys, label=str(group_value))

    ax.set_xlabel(novelty_metric)
    ax.set_ylabel(lag_metric)
    ax.set_title(f"{novelty_metric} vs {lag_metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_report(
    *,
    metrics_csv: Path,
    out_dir: Path,
    group_col: str = "group",
    manifest: Path | None = None,
) -> dict[str, list[Path] | Path]:
    """Generate summary tables and figures for a metrics CSV."""

    metrics_rows, columns = load_metrics(metrics_csv)
    metrics_joined, joined_columns = join_manifest(metrics_rows, manifest, columns=columns)
    _ensure_dir(out_dir)
    metrics_joined_path = out_dir / "metrics_joined.csv"
    _write_csv(metrics_joined_path, joined_columns, metrics_joined)

    group_stats_rows, group_stat_columns = compute_group_stats(metrics_joined, group_col)
    group_stats_path = out_dir / "metrics_group_stats.csv"
    _write_csv(group_stats_path, group_stat_columns, group_stats_rows)

    figures_dir = out_dir / "figures"
    key_metrics = [metric for metric in ("novelty_peak_rate", "lag_energy") if metric in joined_columns]
    boxplots = plot_boxplots(metrics_joined, group_col, key_metrics, figures_dir)
    barplots = plot_bars(metrics_joined, group_col, key_metrics, figures_dir)
    scatter = plot_scatter(
        metrics_joined,
        group_col,
        ["novelty_peak_rate", "novelty_prom_mean"],
        ["lag_energy", "lag_best"],
        figures_dir,
    )

    artifacts: dict[str, list[Path] | Path] = {
        "metrics_joined": metrics_joined_path,
        "group_stats": group_stats_path,
        "boxplots": boxplots,
        "barplots": barplots,
    }
    if scatter:
        artifacts["scatter"] = scatter
    return artifacts
