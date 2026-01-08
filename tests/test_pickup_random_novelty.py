
import csv
import logging
from pathlib import Path
from typer.testing import CliRunner
import matplotlib.pyplot as plt


# Import path changed: ssmproxy.tools -> scripts
import sys
sys.path.append(".")
from scripts.pickup_random_novelty import app

runner = CliRunner()

def create_dummy_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(path)
    plt.close(fig)

def test_pickup_random_novelty(tmp_path):
    """Test pickup tool end-to-end with sampling."""
    eval_out = tmp_path / "eval_out"
    metrics_dir = eval_out / "metrics"
    figures_dir = eval_out / "figures" / "novelty"
    
    metrics_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    
    # 1. Create metrics
    csv_path = metrics_dir / "ssm_proxy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "piece_id", "lag_best", "lag_base_period", "num_bars"])
        
        # Random group - 6 pieces
        # High
        writer.writerow(["random", "r_top1", "20.0", "4", "32"]) 
        writer.writerow(["random", "r_top2", "19.0", "4", "32"])
        # Low
        writer.writerow(["random", "r_bot1", "2.0", "4", "32"])
        # Middle
        writer.writerow(["random", "r_mid1", "10.0", "4", "32"])
        writer.writerow(["random", "r_mid2", "11.0", "4", "32"])
        writer.writerow(["random", "r_mid3", "12.0", "4", "32"])
        
        # Other group
        writer.writerow(["other", "o1", "30.0", "4", "32"])
        
    # 2. Create images for all
    for pid in ["r_top1", "r_top2", "r_bot1", "r_mid1", "r_mid2", "r_mid3", "o1"]:
        create_dummy_png(figures_dir / f"{pid}.png")
        
    # 3. Run tool
    out_dir = tmp_path / "pickup"
    result = runner.invoke(app, [
        "--eval-out", str(eval_out),
        "--group", "random",
        "--top-n", "1",           # Should pick r_top1
        "--also-bottom-n", "1",   # Should pick r_bot1
        "--sample-n", "1",        # Should pick 1 from mid1, mid2, mid3, top2
        "--seed", "42",
        "--outdir", str(out_dir)
    ])
    
    assert result.exit_code == 0
    
    # 4. Verify output
    # Images
    assert (out_dir / "r_top1.png").exists()
    assert (out_dir / "r_bot1.png").exists()
    
    # CSV
    csv_out = out_dir / "selected_pieces.csv"
    assert csv_out.exists()
    
    with open(csv_out, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 3 # top1, bot1, sample1
    
    reasons = [r["reason"] for r in rows]
    assert "top" in reasons
    assert "bottom" in reasons
    assert "sample" in reasons
    
    assert (out_dir / "contact_sheet.png").exists()
    
def test_pickup_missing_images(tmp_path):
    """Test behavior when images are missing but csv is generated."""
    eval_out = tmp_path / "eval_out"
    metrics_dir = eval_out / "metrics"
    metrics_dir.mkdir(parents=True)
    
    csv_path = metrics_dir / "ssm_proxy.csv"
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "piece_id", "lag_best"])
        writer.writerow(["random", "r1", "10.0"])
        
    out_dir = tmp_path / "pickup"
    # No figures created
    
    result = runner.invoke(app, [
        "--eval-out", str(eval_out),
        "--outdir", str(out_dir)
    ])
    
    # Should warn but succeed
    assert result.exit_code == 0
    # CSV should exist
    assert (out_dir / "selected_pieces.csv").exists()
    # Contact sheet not created if no images
    assert not (out_dir / "contact_sheet.png").exists()
