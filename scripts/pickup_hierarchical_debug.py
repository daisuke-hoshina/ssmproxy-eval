
"""Tool to create composite debug plots for hierarchical pieces."""

import csv
import json
import logging
import math
from pathlib import Path
from typing import Optional, List, Dict

import typer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Reuse report utils
try:
    from ssmproxy.report import _read_csv, _ensure_dir, _is_numeric, resolve_metrics_csv, report_plot_style
except ImportError:
    # Minimal fallback
    def _read_csv(path: Path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader), reader.fieldnames
            
    def _ensure_dir(path: Path):
        path.mkdir(parents=True, exist_ok=True)
        
    def _is_numeric(s):
        try:
             float(s)
             return True
        except (ValueError, TypeError):
             return False

    def resolve_metrics_csv(out_dir, explicit=None):
        if explicit: return explicit
        p1 = out_dir / "metrics" / "ssm_proxy.csv"
        if p1.exists(): return p1
        return out_dir / "metrics.csv"
        
    import contextlib
    @contextlib.contextmanager
    def report_plot_style():
        yield

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("pickup_hierarchical_debug")

app = typer.Typer(help="Create composite debug plots for hierarchical pieces.")

def load_lag_energies(eval_out: Path, piece_ids: set[str]) -> Dict[str, dict]:
    """Load specific lag energies from jsonl."""
    # Assuming jsonl is not huge, or we scan it.
    # Usually metrics/lag_energies.jsonl
    path = eval_out / "metrics" / "lag_energies.jsonl"
    data = {}
    if not path.exists():
        LOGGER.warning(f"Lag energies not found at {path}")
        return data
        
    with open(path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                pid = row.get("piece_id")
                if pid in piece_ids:
                    data[pid] = row
            except json.JSONDecodeError:
                pass
    return data

@app.command()
def main(
    eval_out: Path = typer.Option(..., "--eval-out", "-e", exists=True, file_okay=False, resolve_path=True, help="Evaluation output directory."),
    group: str = typer.Option("hierarchical", "--group", "-g", help="Group to filter by."),
    n_small: int = typer.Option(3, "--n-small", help="Number of pieces with smallest best_lag."),
    n_large: int = typer.Option(3, "--n-large", help="Number of pieces with largest best_lag."),
    harmonics: str = typer.Option("1,2,4,8", "--harmonics", help="Comma-separated list of L0 multiples to visualize."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", "-o", resolve_path=True, help="Output directory."),
):
    """Deep dive into hierarchical group pieces with extreme best_lag."""
    
    # Parse harmonics
    try:
        harmonic_mults = [int(h.strip()) for h in harmonics.split(",") if h.strip()]
    except ValueError:
        LOGGER.warning(f"Invalid harmonics format: {harmonics}. Using default [1,2,4,8].")
        harmonic_mults = [1, 2, 4, 8]
        
    # 1. Load Metrics
    joined_csv = eval_out / "summary" / "metrics_joined.csv"
    if not joined_csv.exists():
        metrics_csv = resolve_metrics_csv(eval_out)
        if not metrics_csv.exists():
             LOGGER.error("No metrics found.")
             raise typer.Exit(1)
        rows, _ = _read_csv(metrics_csv)
    else:
        rows, _ = _read_csv(joined_csv)
        
    # 2. Filter & Sort
    group_rows = [r for r in rows if r.get("group") == group]
    valid_rows = []
    for r in group_rows:
        bl = r.get("lag_best") or r.get("best_lag")
        if bl and _is_numeric(bl):
            r["_lag_val"] = float(bl)
            valid_rows.append(r)
            
    valid_rows.sort(key=lambda x: x["_lag_val"]) # Ascending
    
    selected_rows = []
    seen = set()
    
    # Smallest (first N)
    for r in valid_rows[:n_small]:
        if r["piece_id"] not in seen:
            selected_rows.append((r, "small"))
            seen.add(r["piece_id"])
            
    # Largest (last N, reversed)
    if n_large > 0:
        for r in reversed(valid_rows[-n_large:]):
            if r["piece_id"] not in seen:
                selected_rows.append((r, "large"))
                seen.add(r["piece_id"])
                
    if not selected_rows:
        LOGGER.warning("No pieces selected.")
        return

    LOGGER.info(f"Selected {len(selected_rows)} pieces.")
    
    target_dir = outdir or eval_out / "diagnostics" / f"{group}_debug"
    _ensure_dir(target_dir)
    
    # Load lag energies for selected
    pids = seen
    lag_data_map = load_lag_energies(eval_out, pids)
    
    # 3. Create Plots
    figures_ssm = eval_out / "figures" / "ssm"
    figures_novelty = eval_out / "figures" / "novelty"
    
    csv_out_rows = []
    
    for row, reason in selected_rows:
        pid = row["piece_id"]
        lag_val = row["_lag_val"]
        l0 = row.get("lag_base_period")
        num_bars = row.get("num_bars") or row.get("bars")
        
        # Prepare CSV output
        csv_out_rows.append({
            "piece_id": pid,
            "best_lag": lag_val,
            "lag_base_period": l0,
            "num_bars": num_bars,
            "reason": reason,
            "lag_hierarchy_index_auto": row.get("lag_hierarchy_index_auto"),
            "lag_energy": row.get("lag_energy")
        })
        
        # Read Images
        ssm_path = figures_ssm / f"{pid}.png"
        nov_path = figures_novelty / f"{pid}.png"
        
        lag_info = lag_data_map.get(pid)
        
        # Create Layout
        with report_plot_style():
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2)
            
            # SSM
            ax_ssm = fig.add_subplot(gs[:, 0])
            if ssm_path.exists():
                img = mpimg.imread(str(ssm_path))
                ax_ssm.imshow(img)
                ax_ssm.axis('off')
                ax_ssm.set_title(f"SSM: {pid}")
            else:
                 ax_ssm.text(0.5, 0.5, "No SSM Image", ha='center')
                 ax_ssm.axis('off')
                 
            # Lag Spectrum
            ax_lag = fig.add_subplot(gs[0, 1])
            if lag_info:
                energies = lag_info.get("lag_energies", [])
                min_lag = lag_info.get("min_lag", 1)
                xs = list(range(min_lag, min_lag + len(energies)))
                ax_lag.plot(xs, energies, label="Lag Energy")
                
                # Markers
                ax_lag.axvline(x=lag_val, color='r', linestyle='--', label=f"Best Lag={lag_val:.1f}")
                
                if l0 and _is_numeric(l0):
                    l0_val = float(l0)
                    # L0 dashed line
                    ax_lag.axvline(x=l0_val, color='g', linestyle=':', label=f"L0={l0_val:.1f}")
                    
                    # Harmonic Guides (lighter)
                    # We check range of xs to only plot visible lines
                    x_min, x_max = min(xs), max(xs)
                    for m in harmonic_mults:
                        h_val = m * l0_val
                        # Avoid duplicates with L0 line if m=1
                        if m == 1: continue 
                        if x_min <= h_val <= x_max:
                            ax_lag.axvline(x=h_val, color='g', linestyle=':', alpha=0.5, linewidth=0.8)
                            # Optional: text label?
                            # ax_lag.text(h_val, max(energies), f"{m}L0", fontsize=6, color='g')

                ax_lag.legend(loc='upper right', fontsize='small')
                ax_lag.set_title("Lag Spectrum (with L0 harmonics)")
                ax_lag.set_xlabel("Lag (bars)")
                ax_lag.set_ylabel("Energy")
            else:
                ax_lag.text(0.5, 0.5, "No Lag Data", ha='center')
                ax_lag.axis('off')
                
            # Novelty
            ax_nov = fig.add_subplot(gs[1, 1])
            if nov_path.exists():
                img = mpimg.imread(str(nov_path))
                ax_nov.imshow(img)
                ax_nov.axis('off')
                ax_nov.set_title("Novelty Curve")
            else:
                ax_nov.text(0.5, 0.5, "No Novelty Image", ha='center')
                ax_nov.axis('off')
                
            fig.suptitle(f"Debug: {pid} ({reason.upper()})\nBestLag={lag_val}, L0={l0}, Bars={num_bars}")
            
            # Robust Layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
            
            out_path = target_dir / f"{pid}_debug.png"
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            LOGGER.info(f"Saved {out_path}")
            
    # Write CSV
    if csv_out_rows:
        csv_path = target_dir / "selected_pieces.csv"
        # Determine strict fieldnames from first row or predefined keys
        fieldnames = ["piece_id", "reason", "best_lag", "lag_base_period", "num_bars", "lag_hierarchy_index_auto", "lag_energy"]
        # Ensure all rows have all keys
        for r in csv_out_rows:
            for k in fieldnames:
                if k not in r: r[k] = None
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_out_rows)
            
    LOGGER.info("Done.")

if __name__ == "__main__":
    app()
