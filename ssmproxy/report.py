"""Reporting utilities for aggregating evaluation outputs without external dependencies."""

from __future__ import annotations

import csv
import importlib.util
import logging
import math
import statistics
import random
from pathlib import Path
from typing import Iterable, Sequence

from .metrics import canonical_metrics_path, legacy_metrics_path
from .plots import _write_png

import json
import numpy as np

LOGGER = logging.getLogger(__name__)

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    import matplotlib  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    matplotlib.use("Agg")
else:
    plt = None


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = [row for row in reader]
    return rows, reader.fieldnames or []


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
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

    for group_value, metrics in sorted(grouped.items(), key=lambda item: item[0]):
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

        labels = sorted(grouped.keys())
        data = [grouped[label] for label in labels]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, labels=labels)
        
        # Custom overlays
        # Check if we should use jitter points for this metric
        jitter_metrics = {
            "novelty_peak_rate", 
            "lag_hierarchy_index", 
            "novelty_tv", 
            "novelty_std"
        }
        
        if metric in jitter_metrics:
            # Add jitter strip plot
            # Seed for reproducibility as requested
            rng = random.Random(0)
            for i, label in enumerate(labels):
                y_values = grouped[label]
                # Center x at i+1
                x_values = [i + 1 + rng.uniform(-0.12, 0.12) for _ in y_values]
                ax.scatter(x_values, y_values, color='blue', alpha=0.5, s=10, zorder=3)
        else:
            # Default mean overlay (e.g. for lag_energy)
            # Calculate means for overlay
            means = [statistics.fmean(grouped[label]) if grouped[label] else 0.0 for label in labels]
            # Overlay means: x-positions are 1, 2, ..., len(labels)
            x_positions = range(1, len(labels) + 1)
            ax.scatter(x_positions, means, color='red', marker='o', s=30, label='Mean', zorder=3)
        
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

        labels = sorted(grouped.keys())
        means = [statistics.fmean(grouped[label]) if grouped[label] else 0.0 for label in labels]
        stds = [statistics.stdev(grouped[label]) if len(grouped[label]) > 1 else 0.0 for label in labels]

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
    filename: str = "scatter_novelty_vs_lag.png",
) -> Path | None:
    _ensure_dir(output_dir)
    novelty_metric = next((m for m in novelty_metric_candidates if m in rows[0]), None) if rows else None
    lag_metric = next((m for m in lag_metric_candidates if m in rows[0]), None) if rows else None

    if not novelty_metric or not lag_metric:
        LOGGER.warning("Skipping scatter plot; required metrics missing.")
        return None

    output_path = output_dir / filename
    
    # Collect paired values to ensure alignment
    # grouped: dict[group, list[tuple[x, y]]]
    grouped_points: dict[str, list[tuple[float, float]]] = {}
    
    for row in rows:
        group_value = row.get(group_col, "")
        x_str = row.get(novelty_metric, "")
        y_str = row.get(lag_metric, "")
        
        if x_str != "" and _is_numeric(x_str) and y_str != "" and _is_numeric(y_str):
             grouped_points.setdefault(group_value, []).append((float(x_str), float(y_str)))

    if plt is None:
        _save_placeholder(output_path, f"scatter {filename}")
        return output_path

    if not grouped_points:
         return None

    fig, ax = plt.subplots(figsize=(6, 4))
    for group_value in sorted(grouped_points.keys()):
        points = grouped_points[group_value]
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, label=str(group_value))

    ax.set_xlabel(novelty_metric)
    ax.set_ylabel(lag_metric)
    ax.set_title(f"{novelty_metric} vs {lag_metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_lag_spectrum_mean(
    lag_energies_path: Path, output_dir: Path
) -> Path | None:
    _ensure_dir(output_dir)
    output_path = output_dir / "lag_spectrum_mean_by_group.png"

    if not lag_energies_path.is_file():
        LOGGER.warning("Lag energies file missing: %s", lag_energies_path)
        return None

    if plt is None:
        _save_placeholder(output_path, "lag spectrum mean")
        return output_path

    # Load and aggregate data
    grouped_energies: dict[str, list[list[float]]] = {}
    with lag_energies_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            group = row.get("group", "")
            energies = row.get("lag_energies", [])
            # Skip empty or null values
            energies = [e for e in energies if e is not None]
            if energies:
                grouped_energies.setdefault(group, []).append(energies)

    if not grouped_energies:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Determine max lag for x-axis alignment
    # Assuming all runs have same lag settings, but to be safe we find max length
    max_len = 0
    for group_runs in grouped_energies.values():
        for run in group_runs:
            max_len = max(max_len, len(run))
    
    for group, runs in sorted(grouped_energies.items()):
        # Pad shorter runs with NaNs if necessary (though usually they should match)
        # or just truncate/align. Here we assume alignment by index (lag index).
        # We'll compute mean per index.
        
        # Structure: runs is list of lists.
        # Check max length for this group
        g_max_len = max(len(r) for r in runs)
        
        # Accumulate sums and counts
        sums = np.zeros(g_max_len)
        counts = np.zeros(g_max_len)
        sq_sums = np.zeros(g_max_len)
        
        valid_runs = 0
        for run in runs:
            arr = np.array(run)
            length = len(arr)
            sums[:length] += arr
            sq_sums[:length] += arr ** 2
            counts[:length] += 1
            valid_runs += 1
            
        # Avoid division by zero
        with np.errstate(invalid='ignore'):
            means = sums / counts
            stds = np.sqrt((sq_sums / counts) - (means ** 2))
            sems = stds / np.sqrt(counts)
            
        # x-axis: indices (representing lag 0, 1, 2...)
        # Note: lag_energies usually starts from lag=0 or lag=min_lag depending on impl.
        # The pipeline returns valid energies. We plot by index.
        x = np.arange(g_max_len)
        
        # Filter out indices with no data
        mask = counts > 0
        ax.errorbar(x[mask], means[mask], yerr=sems[mask], label=str(group), capsize=3, alpha=0.8)

    ax.set_xlabel("Lag Index (offset from min_lag)")
    ax.set_ylabel("Lag Energy (Mean ± SEM)")
    ax.set_title("Group Mean Lag Spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_best_lag_distribution(
    rows: list[dict[str, str]], group_col: str, output_dir: Path
) -> Path | None:
    _ensure_dir(output_dir)
    output_path = output_dir / "best_lag_distribution_by_group.png"
    
    if plt is None:
        _save_placeholder(output_path, "best lag distribution")
        return output_path

    grouped = _collect_values(rows, group_col, "lag_best")
    if not grouped:
        return None

    labels = sorted(grouped.keys())
    data = [grouped[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    # Violin plot for detailed distribution or boxplot
    # Using boxplot as requested/consistent with others
    ax.boxplot(data, labels=labels)
    ax.set_title(f"Best Lag Distribution by {group_col}")
    ax.set_ylabel("Best Lag Index")
    ax.set_xlabel(group_col)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def predict_group_rule(row: dict[str, object]) -> str:
    """Predict group based on simple rules derived for the toy dataset."""

    def get_float(key: str) -> float | None:
        val = row.get(key)
        if val is None or val == "":
             return None
        try:
            f = float(str(val))
            if math.isfinite(f):
                return f
        except (ValueError, TypeError):
            pass
        return None

    def get_int(key: str) -> int | None:
        val = row.get(key)
        if val is None or val == "":
             return None
        try:
            return int(float(str(val)))
        except (ValueError, TypeError):
            return None

    lag_energy = get_float("lag_energy")
    lag_best = get_int("lag_best")
    
    # 1. Random check
    if lag_energy is not None and lag_energy < 1.2:
        return "random"
        
    if lag_best is None:
        return "unknown"

    # 2. Branch by best_lag
    if lag_best == 24:
        return "AABA"
    
    lag_hierarchy_index_auto = get_float("lag_hierarchy_index_auto")
    lag_hierarchy_index = get_float("lag_hierarchy_index")
    
    # Use auto if available, else fixed
    idx = lag_hierarchy_index_auto if lag_hierarchy_index_auto is not None else lag_hierarchy_index
    
    if lag_best == 4:
        # repeat vs hierarchical
        if idx is not None and idx > 0.08:
            return "hierarchical"
        else:
            return "repeat"
            
    if lag_best == 8:
        return "hierarchical"
        
    if lag_best == 16:
        # ABAB vs partial_copy
        if idx is not None and idx < 0:
            return "ABAB"
        
        novelty_peak_rate = get_float("novelty_peak_rate")
        if novelty_peak_rate is not None and novelty_peak_rate >= 0.07:
            return "ABAB"
            
        return "partial_copy"

    return "unknown"


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    output_dir: Path,
    filename: str = "confusion_matrix_rule",
) -> tuple[Path, Path, Path] | None:
    """Generate confusion matrix artifacts (CSV, PNG, Summary)."""
    
    _ensure_dir(output_dir)
    
    # Compute confusion matrix
    # Rows: True, Cols: Pred
    matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
    
    all_labels = set(labels)
    for t in y_true: all_labels.add(t)
    for p in y_pred: all_labels.add(p)
    
    sorted_labels = sorted(list(all_labels))
    preferred_order = ["AABA", "ABAB", "hierarchical", "partial_copy", "random", "repeat"]
    final_labels = [l for l in preferred_order if l in all_labels]
    for l in sorted_labels:
        if l not in final_labels:
            final_labels.append(l)
            
    # Re-init matrix with full labels
    matrix = {t: {p: 0 for p in final_labels} for t in final_labels}
    
    correct_count = 0
    total_count = 0
    
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
             matrix[t][p] += 1
        
        if t == p:
            correct_count += 1
        total_count += 1
        
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # Save CSV
    csv_path = output_dir.parent / f"{filename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true_group \\ pred_group"] + final_labels)
        for t in final_labels:
            row = [t] + [matrix[t][p] for p in final_labels]
            writer.writerow(row)
            
    # Save Summary
    summary_path = output_dir.parent / f"{filename}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write(f"accuracy={accuracy:.4f}\n")
        fp.write(f"count={total_count}\n")
        fp.write(f"correct={correct_count}\n")

    # Plot Heatmap
    png_path = output_dir / f"{filename}.png"
    if plt is None:
        _save_placeholder(png_path, "confusion matrix")
        return csv_path, summary_path, png_path

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare array for imshow
    # imshow expects (nrows, ncols)
    arr = np.array([[matrix[t][p] for p in final_labels] for t in final_labels])
    
    im = ax.imshow(arr, cmap="Blues")
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(final_labels)))
    ax.set_yticks(np.arange(len(final_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(final_labels, rotation=45, ha="right")
    ax.set_yticklabels(final_labels)

    # Loop over data dimensions and create text annotations.
    for i in range(len(final_labels)):
        for j in range(len(final_labels)):
            text = ax.text(j, i, arr[i, j],
                           ha="center", va="center", color="black")

    ax.set_title(f"Confusion Matrix (Rule-Based)\nAccuracy: {accuracy:.2%}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    
    return csv_path, summary_path, png_path


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
    
    # Exclude lag_energy and novelty_peak_rate from bar plots
    bar_metrics = [m for m in key_metrics if m not in ("lag_energy", "novelty_peak_rate")]
    barplots = plot_bars(metrics_joined, group_col, bar_metrics, figures_dir)
    scatter = plot_scatter(
        metrics_joined,
        group_col,
        ["novelty_peak_rate", "novelty_prom_mean"],
        ["lag_energy", "lag_best"],
        figures_dir,
    )
    
    # Advanced metrics plots
    # Boxplot + Jitter for lag_hierarchy_index and novelty_tv (and optionally novelty_std)
    adv_metrics = ["lag_hierarchy_index", "lag_hierarchy_index_auto", "novelty_tv", "novelty_std"]
    # Filter only available metrics
    adv_metrics = [m for m in adv_metrics if m in joined_columns]
    plot_boxplots(metrics_joined, group_col, adv_metrics, figures_dir)
    
    # Scatter plot: x=lag_hierarchy_index, y=novelty_tv
    plot_scatter(
        metrics_joined,
        group_col,
        ["lag_hierarchy_index"],
        ["novelty_tv"],
        figures_dir,
        filename="lag_hierarchy_index_vs_novelty_tv.png",
    )
    
    # Scatter plot: x=lag_hierarchy_index_auto, y=novelty_tv
    plot_scatter(
        metrics_joined,
        group_col,
        ["lag_hierarchy_index_auto"],
        ["novelty_tv"],
        figures_dir,
        filename="lag_hierarchy_index_auto_vs_novelty_tv.png",
    )
    
    # New plots
    lag_spectrum = plot_lag_spectrum_mean(metrics_csv.parent / "lag_energies.jsonl", figures_dir)
    best_lag_dist = plot_best_lag_distribution(metrics_joined, group_col, figures_dir)
    
    # Rule-based classification
    # Only run if we detect toy classes or if explicitly requested
    toy_classes = {"AABA", "ABAB", "hierarchical", "partial_copy", "random", "repeat"}
    actual_groups = set(row.get(group_col, "") for row in metrics_joined)
    
    cm_files: tuple[Path, Path, Path] | None = None
    
    # Calculate overlap or just run if prediction feasible
    if not actual_groups.isdisjoint(toy_classes):
        y_true = []
        y_pred = []
        for row in metrics_joined:
            pred = predict_group_rule(row)
            row["pred_group_rule"] = pred
            
            true_g = row.get(group_col, "")
            if true_g:
                y_true.append(true_g)
                y_pred.append(pred)
        
        # Rewrite metrics_joined with prediction
        joined_columns.append("pred_group_rule")
        _write_csv(metrics_joined_path, joined_columns, metrics_joined)
        
        cm_files = plot_confusion_matrix(y_true, y_pred, list(toy_classes), figures_dir)

    artifacts: dict[str, list[Path] | Path] = {
        "metrics_joined": metrics_joined_path,
        "group_stats": group_stats_path,
        "boxplots": boxplots,
        "barplots": barplots,
    }
    if scatter:
        artifacts["scatter"] = scatter
    if cm_files:
        artifacts["confusion_matrix"] = list(cm_files)
        
    return artifacts
