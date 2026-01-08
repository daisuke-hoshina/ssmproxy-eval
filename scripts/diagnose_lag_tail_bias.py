
"""Diagnostic script for visualizing lag tail bias.

Refactored to be standalone and robust.
"""

import csv
import json
import logging
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import typer
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("diagnose_lag_tail_bias")

app = typer.Typer(help="Diagnose lag tail bias in evaluation results.")


# --- Standalone Helpers ---

def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV file returning list of dicts and fieldnames."""
    if not path.is_file():
        return [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames or []

def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    """Write list of dicts to CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _is_numeric(s: Any) -> bool:
    if s is None:
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
        
def _resolve_metrics_csv(eval_out: Path) -> Path:
    """Resolve the best metrics CSV path."""
    # Priority 1: summary/metrics_joined.csv
    p1 = eval_out / "summary" / "metrics_joined.csv"
    if p1.is_file(): return p1
    # Priority 2: metrics/ssm_proxy.csv
    p2 = eval_out / "metrics" / "ssm_proxy.csv"
    if p2.is_file(): return p2
    # Priority 3: metrics.csv
    p3 = eval_out / "metrics.csv"
    return p3

def _safe_float(x: Any) -> Optional[float]:
    if _is_numeric(x):
        return float(x)
    return None

# --- Core Logic ---

def load_data(eval_out: Path, group_col: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load metrics and join with lag energies."""
    
    # 1. Load Metrics
    metrics_path = _resolve_metrics_csv(eval_out)
    LOGGER.info(f"Loading metrics from {metrics_path}")
    raw_rows, _ = _read_csv(metrics_path)
    if not raw_rows:
        LOGGER.error("No metrics found.")
        return [], {}

    # Map piece_id -> group
    piece_map = {} # piece_id -> {metrics_row}
    pid_to_group = {}
    
    for r in raw_rows:
        pid = r.get("piece_id")
        if not pid: continue
        
        # Normalize fields
        # best_lag
        bl = r.get("lag_best") or r.get("best_lag")
        # l0
        l0 = r.get("lag_base_period") or r.get("base_period")
        # num_bars
        nb = r.get("num_bars") or r.get("bars")
        
        # group
        grp = r.get(group_col, "unknown")
        
        # Store clean values
        piece_map[pid] = {
            "piece_id": pid,
            "group": grp,
            "best_lag": _safe_float(bl),
            "l0": _safe_float(l0),
            "num_bars": _safe_float(nb),
        }
        pid_to_group[pid] = grp
        
    LOGGER.info(f"Loaded metrics for {len(piece_map)} pieces.")

    # 2. Load Lag Energies
    jsonl_path = eval_out / "metrics" / "lag_energies.jsonl"
    lag_data = [] # List of {piece_id, energies, min_lag}
    
    if jsonl_path.is_file():
        LOGGER.info(f"Loading lag energies from {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    pid = row.get("piece_id")
                    if pid in piece_map:
                        # Join group from metrics!
                        row["group"] = pid_to_group[pid]
                        # Join other meta for filtering if needed
                        # Ensure numeric energies
                        ens = row.get("lag_energies", [])
                        row["lag_energies"] = [float(e) if e is not None else 0.0 for e in ens]
                        lag_data.append(row)
                except json.JSONDecodeError:
                    pass
    else:
        LOGGER.warning("lag_energies.jsonl not found.")
        
    # Return merged simple list for plotting, and piece_map for table
    return list(piece_map.values()), lag_data


def plot_group_mean_lag_spectrum(lag_data: List[Dict[str, Any]], output_path: Path):
    """Plot mean lag spectrum per group with correct grouping."""
    if not lag_data:
        LOGGER.warning("No lag data for spectrum plot.")
        return
        
    grouped = {}
    for row in lag_data:
        g = row.get("group", "unknown")
        ens = row.get("lag_energies", [])
        if not ens: continue
        grouped.setdefault(g, []).append(ens)
        
    if not grouped:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We use offset index 0..max_len
    # Assuming min_lag is reasonably consistent or we just visualize profile shape
    # The requirement says "lag index offset from min_lag" is OK.
    
    for g, curves in sorted(grouped.items()):
        # Pad to max length
        max_len = max(len(c) for c in curves)
        arr = np.zeros((len(curves), max_len))
        arr[:] = np.nan
        
        for i, c in enumerate(curves):
            arr[i, :len(c)] = c
            
        # Calc mean / std ignoring nan
        mean = np.nanmean(arr, axis=0)
        # sem = std / sqrt(n)
        count = np.sum(~np.isnan(arr), axis=0)
        std = np.nanstd(arr, axis=0)
        sem = std / np.sqrt(count)
        
        xs = np.arange(max_len)
        ax.plot(xs, mean, label=f"{g} (n={len(curves)})")
        ax.fill_between(xs, mean-sem, mean+sem, alpha=0.2)
        
    ax.set_xlabel("Lag Offset (steps from min_lag)")
    ax.set_ylabel("Mean Energy")
    ax.set_title("Group Mean Lag Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info(f"Saved {output_path}")


def plot_best_lag_vs_support(rows: List[Dict[str, Any]], output_path: Path):
    """Plot Best Lag vs Support."""
    grouped = {}
    for r in rows:
        if r["best_lag"] is not None and r["num_bars"] is not None:
            g = r["group"]
            s = r["num_bars"] - r["best_lag"]
            grouped.setdefault(g, []).append((r["best_lag"], s))
            
    if not grouped:
        LOGGER.warning("No valid data for Best Lag vs Support.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    for g, points in sorted(grouped.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, label=g, alpha=0.6, s=20)
        
    ax.set_xlabel("Best Lag (L)")
    ax.set_ylabel("Support (bars - L)")
    ax.set_title("Best Lag vs Support")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info(f"Saved {output_path}")


def plot_l0_vs_best_lag(rows: List[Dict[str, Any]], output_path: Path):
    """Plot L0 vs Best Lag with diagonal."""
    grouped = {}
    skipped = 0
    for r in rows:
        if r["best_lag"] is not None and r["l0"] is not None:
            g = r["group"]
            grouped.setdefault(g, []).append((r["best_lag"], r["l0"]))
        else:
            skipped += 1
            
    if skipped > 0:
        LOGGER.info(f"Skipped {skipped} rows missing L0 or Best Lag for scatter plot.")

    if not grouped:
        LOGGER.warning("No valid data for L0 vs Best Lag.")
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Find max range for diagonal
    all_vals = []
    
    for g, points in sorted(grouped.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, label=g, alpha=0.6, s=20)
        all_vals.extend(xs)
        all_vals.extend(ys)
        
    # Diagonal
    if all_vals:
        mn, mx = min(all_vals), max(all_vals)
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label="y=x")
        
    ax.set_xlabel("Best Lag")
    ax.set_ylabel("L0 (Estimated Base Period)")
    ax.set_title("L0 vs Best Lag")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info(f"Saved {output_path}")


def plot_best_lag_histogram(rows: List[Dict[str, Any]], output_path: Path):
    """Stacked histogram of best lag."""
    grouped = {}
    for r in rows:
        if r["best_lag"] is not None:
            g = r["group"]
            grouped.setdefault(g, []).append(r["best_lag"])
            
    if not grouped:
        return
        
    labels = sorted(grouped.keys())
    data = [grouped[l] for l in labels]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data, bins=30, label=labels, alpha=0.7, stacked=True)
    
    ax.set_xlabel("Best Lag Index")
    ax.set_ylabel("Count")
    ax.set_title("Best Lag Distribution (Stacked)")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info(f"Saved {output_path}")

def save_table(rows: List[Dict[str, Any]], group_col: str, output_path: Path):
    """Save summary table CSV."""
    out_rows = []
    for r in rows:
        bl = r["best_lag"]
        l0 = r["l0"]
        nb = r["num_bars"]
        
        support = (nb - bl) if (nb is not None and bl is not None) else None
        l0_diff = (l0 - bl) if (l0 is not None and bl is not None) else None
        l0_missing = (l0 is None)
        
        out_rows.append({
            "piece_id": r["piece_id"],
            "group_value": r["group"],
            "num_bars": nb,
            "best_lag": bl,
            "support": support,
            "l0": l0,
            "l0_minus_best": l0_diff,
            "l0_is_missing": l0_missing
        })
        
    _write_csv(output_path, out_rows, [
        "piece_id", "group_value", "num_bars", "best_lag", "support", "l0", "l0_minus_best", "l0_is_missing"
    ])
    LOGGER.info(f"Saved {output_path}")

@app.command()
def main(
    eval_out: Path = typer.Option(..., "--eval-out", "-e", exists=True, file_okay=False, resolve_path=True, help="Evaluation output directory."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o", resolve_path=True, help="Output directory for plots."),
    group_col: str = typer.Option("group", "--group-col", "-g", help="Column to group by.")
):
    """Diagnose lag tail bias using evaluation outputs."""
    
    if out_dir is None:
        out_dir = eval_out / "diagnostics" / "lag_tail_bias"
    
    _ensure_dir(out_dir)
    
    LOGGER.info(f"Diagnosing {eval_out}, output to {out_dir}")
    
    # 1. Load Data
    metrics_rows, lag_data = load_data(eval_out, group_col)
    if not metrics_rows:
        raise typer.Exit(code=1)
        
    # 2. Plots
    plot_group_mean_lag_spectrum(lag_data, out_dir / "lag_spectrum_mean_by_group.png")
    plot_best_lag_vs_support(metrics_rows, out_dir / "scatter_best_lag_vs_support.png")
    plot_best_lag_histogram(metrics_rows, out_dir / "hist_best_lag.png")
    plot_l0_vs_best_lag(metrics_rows, out_dir / "scatter_l0_vs_best_lag.png")
    
    # 3. Table
    save_table(metrics_rows, group_col, out_dir / "lag_tail_bias_piece_table.csv")
    
    LOGGER.info("Done.")

if __name__ == "__main__":
    app()
