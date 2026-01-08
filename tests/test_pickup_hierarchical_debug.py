
import csv
import json
import logging
from pathlib import Path
from typer.testing import CliRunner
import matplotlib.pyplot as plt


# Import path changed: ssmproxy.tools -> scripts
import sys
sys.path.append(".")
from scripts.pickup_hierarchical_debug import app

runner = CliRunner()

def create_dummy_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(path)
    plt.close(fig)

def test_pickup_hierarchical_debug(tmp_path):
    """Test debug pickup tool end-to-end."""
    eval_out = tmp_path / "eval_out"
    metrics_dir = eval_out / "metrics"
    figures_ssm = eval_out / "figures" / "ssm"
    figures_nov = eval_out / "figures" / "novelty"
    
    metrics_dir.mkdir(parents=True)
    figures_ssm.mkdir(parents=True)
    figures_nov.mkdir(parents=True)
    
    # 1. Create metrics
    csv_path = metrics_dir / "ssm_proxy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "piece_id", "lag_best", "lag_base_period", "num_bars", "lag_hierarchy_index_auto", "lag_energy"])
        
        # Hierarchical group
        # Smallest 1: h1 (best_lag=4)
        writer.writerow(["hierarchical", "h1", "4.0", "4", "32", "0.5", "0.9"])
        # Largest 1: h2 (best_lag=16)
        writer.writerow(["hierarchical", "h2", "16.0", "4", "32", "0.6", "0.8"])
        # Middle: h3 (skipped if n=1)
        writer.writerow(["hierarchical", "h3", "8.0", "4", "32", "0.4", "0.7"])
        
        # Other group
        writer.writerow(["other", "o1", "30.0", "4", "32", "0.1", "0.1"])
        
    # 2. Lag Energies
    jsonl_path = metrics_dir / "lag_energies.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps({"piece_id": "h1", "lag_energies": [0.1, 0.2, 0.8, 0.9], "min_lag": 1}) + "\n")
        f.write(json.dumps({"piece_id": "h2", "lag_energies": [0.1, 0.1, 0.1, 0.1, 0.9], "min_lag": 1}) + "\n")
        
    # 3. Create Images
    for pid in ["h1", "h2"]:
        create_dummy_png(figures_ssm / f"{pid}.png")
        create_dummy_png(figures_nov / f"{pid}.png")
        
    # 4. Run Tool
    out_dir = tmp_path / "debug"
    result = runner.invoke(app, [
        "--eval-out", str(eval_out),
        "--group", "hierarchical",
        "--n-small", "1",
        "--n-large", "1",
        "--harmonics", "2,4",
        "--outdir", str(out_dir)
    ])
    
    if result.exit_code != 0:
        print(result.stdout)
        
    assert result.exit_code == 0
    
    # 5. Verify Output
    assert (out_dir / "h1_debug.png").exists()
    assert (out_dir / "h2_debug.png").exists()
    # h3 skipped
    assert not (out_dir / "h3_debug.png").exists()
    
    assert (out_dir / "selected_pieces.csv").exists()
    
    # Verify CSV content
    with open(out_dir / "selected_pieces.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 2
    row1 = next(r for r in rows if r["piece_id"] == "h1")
    assert row1["reason"] == "small"
    assert row1["lag_hierarchy_index_auto"] == "0.5"
    assert row1["lag_energy"] == "0.9"
