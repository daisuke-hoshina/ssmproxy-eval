
"""Tool to pickup and visualize novelty plots for specific pieces."""

import shutil
import logging
import math
import random
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Reuse report utils
try:
    from ssmproxy.report import _read_csv, _ensure_dir, _is_numeric, resolve_metrics_csv
except ImportError:
    # Fallback if specific env setup issues
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

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("pickup_random_novelty")

app = typer.Typer(help="Pick up novelty plots for inspection.")

def create_contact_sheet(image_paths: list[tuple[Path, str]], output_path: Path):
    """Create a grid of images with titles."""
    n = len(image_paths)
    if n == 0:
        return

    cols = 4 # Fixed columns or adaptive
    rows = math.ceil(n / cols)
    
    fig_width = 4 * cols
    fig_height = 3 * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Robust axes flattening
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    elif not isinstance(axes, (list, tuple, type(np.array([])))): # Single axis
         axes = [axes]
    else:
        # Should be list/array
        axes = list(axes)
        
    # Ensure axes is a flat list
    # (If using numpy.ndarray, flatten() returns an iterator/array, fine)
    
    # Fallback check if simple list
    import numpy as np
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    
    # Final safety for n=1 case where subplots returns just ax
    if n == 1 and not isinstance(axes, (list, np.ndarray)):
         axes = [axes]

    for i, (path, title) in enumerate(image_paths):
        ax = axes[i]
        try:
            img = mpimg.imread(str(path))
            ax.imshow(img)
            # title with smaller font
            ax.set_title(title, fontsize=6)
            ax.axis('off')
        except Exception as e:
            LOGGER.warning(f"Failed to read image {path}: {e}")
            ax.text(0.5, 0.5, f"Load Failed\n{path.name}", ha='center', fontsize=6)
            ax.axis('off')
            
    # Turn off remaining axes
    for i in range(n, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info(f"Saved contact sheet to {output_path}")

@app.command()
def main(
    eval_out: Path = typer.Option(..., "--eval-out", "-e", exists=True, file_okay=False, resolve_path=True, help="Evaluation output directory."),
    group: str = typer.Option("random", "--group", "-g", help="Group to filter by."),
    top_n: int = typer.Option(6, "--top-n", help="Number of pieces with highest best_lag to pick."),
    also_bottom_n: int = typer.Option(3, "--also-bottom-n", help="Number of pieces with lowest best_lag to pick."),
    sample_n: int = typer.Option(0, "--sample-n", help="Number of random samples to pick from the remaining."),
    seed: int = typer.Option(0, "--seed", help="Random seed for sampling."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", "-o", resolve_path=True, help="Output directory."),
):
    """Select random novelty plots and create a contact sheet."""
    
    # 1. Load Metrics
    joined_csv = eval_out / "summary" / "metrics_joined.csv"
    if not joined_csv.exists():
        LOGGER.warning(f"{joined_csv} not found, trying base metrics.")
        metrics_csv = resolve_metrics_csv(eval_out)
        if not metrics_csv.exists():
             LOGGER.error("No metrics found.")
             raise typer.Exit(1)
        rows, _ = _read_csv(metrics_csv)
    else:
        rows, _ = _read_csv(joined_csv)
        
    # 2. Filter Group
    group_rows = [r for r in rows if r.get("group") == group]
    LOGGER.info(f"Found {len(group_rows)} pieces for group '{group}'.")
    
    if not group_rows:
        return

    # 3. Sort by best_lag
    # Filter valid best_lag
    valid_rows = []
    for r in group_rows:
        bl = r.get("lag_best") or r.get("best_lag")
        if bl and _is_numeric(bl):
            r["_lag_val"] = float(bl)
            valid_rows.append(r)
            
    # Sort Descending for Top N
    valid_rows.sort(key=lambda x: x["_lag_val"], reverse=True) 
    
    # 4. Selection Logic
    selection = [] # List[Tuple[row, reason]]
    seen_ids = set()
    
    # Top N (Highest Lags)
    for r in valid_rows[:top_n]:
        pid = r["piece_id"]
        if pid not in seen_ids:
            selection.append((r, "top"))
            seen_ids.add(pid)
            
    # Bottom N (Lowest Lags)
    # We take from the end of sorted list
    if also_bottom_n > 0 and len(valid_rows) > top_n:
        for r in valid_rows[-also_bottom_n:]:
            pid = r["piece_id"]
            if pid not in seen_ids:
                selection.append((r, "bottom"))
                seen_ids.add(pid)
                
    # Random Sample from Middle
    if sample_n > 0:
        candidates = [r for r in valid_rows if r["piece_id"] not in seen_ids]
        if candidates:
            rng = random.Random(seed)
            picked = rng.sample(candidates, k=min(len(candidates), sample_n))
            for r in picked:
                selection.append((r, "sample"))
                seen_ids.add(r["piece_id"])
    
    # Sort selection by reason or keep order? Top -> Bottom -> Sample?
    # Let's keep them in order of addition for now, or sort by reason for contact sheet grouping.
    # User might prefer Top then Sample then Bottom (descending Lag).
    # But current logic is Top(High), Bottom(Low), Sample(Middle).
    # Re-sorting selection by lag value (descending) usually makes more visual sense.
    selection.sort(key=lambda x: x[0]["_lag_val"], reverse=True)
    
    LOGGER.info(f"Selected {len(selection)} pieces.")
    
    if not selection:
        return
        
    # 5. Output Preparation
    target_dir = outdir or eval_out / "diagnostics" / f"{group}_pickup"
    _ensure_dir(target_dir)
    
    figures_dir = eval_out / "figures" / "novelty"
    
    copied_images = []
    csv_out_rows = []
    
    for row, reason in selection:
        pid = row["piece_id"]
        
        # Prepare CSV Data
        bl = row["_lag_val"]
        l0 = row.get("lag_base_period") or row.get("base_period")
        nb = row.get("num_bars") or row.get("bars")
        
        # Support = nb - bl (if robust)
        try:
             nb_val = float(nb)
             support = nb_val - bl
        except (ValueError, TypeError):
             support = None
             
        csv_out_rows.append({
            "piece_id": pid,
            "reason": reason,
            "best_lag": bl,
            "lag_base_period": l0,
            "num_bars": nb,
            "support": support,
            "lag_energy": row.get("lag_energy"),
            "lag_hierarchy_index_auto": row.get("lag_hierarchy_index_auto")
        })

        # Copy Image
        src = figures_dir / f"{pid}.png"
        dst = target_dir / f"{pid}.png"
        
        image_ok = False
        if src.exists():
            shutil.copy2(src, dst)
            image_ok = True
        else:
            LOGGER.warning(f"Image not found: {src}")
            # Still add to contact sheet list but logic inside create_contact_sheet handles missing?
            # Or we skip image but keep CSV?
            # Requirement says "CSV exists even if image missing".
            # For contact sheet, we probably want a placeholder or skip.
            # Let's skip contact sheet entry if image missing, or let create_contact_sheet handle fail.
            # create_contact_sheet tries to read. So let's pass a dummy path or skip.
            pass
            
        if image_ok:
            # Title formatting
            # "{pid}\nL={best_lag:.1f}, Bars={num_bars}, Sup={support} (L0={l0}) [{reason}]"
            l0_str = f"{float(l0):.1f}" if _is_numeric(l0) else "?"
            sup_str = f"{support:.1f}" if support is not None else "?"
            nb_str = f"{nb}"
            
            title = f"{pid}\nL={bl:.1f}, Bars={nb_str}, Sup={sup_str} (L0={l0_str}) [{reason}]"
            copied_images.append((dst, title))
        
    # 6. Save CSV
    if csv_out_rows:
        headers = ["piece_id", "reason", "best_lag", "lag_base_period", "num_bars", "support", "lag_energy", "lag_hierarchy_index_auto"]
        csv_path = target_dir / "selected_pieces.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_out_rows)
        LOGGER.info(f"Saved CSV to {csv_path}")

    # 7. Contact Sheet
    if copied_images:
        contact_path = target_dir / "contact_sheet.png"
        create_contact_sheet(copied_images, contact_path)
        
    LOGGER.info("Done.")

if __name__ == "__main__":
    app()
