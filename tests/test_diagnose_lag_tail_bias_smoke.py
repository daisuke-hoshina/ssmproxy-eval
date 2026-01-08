
import csv
import json
import logging
import shutil
from pathlib import Path
from typer.testing import CliRunner


# Import path changed: ssmproxy.scripts -> scripts
# We need to ensure 'scripts' is importable or use sys.path hack for test if not a package.
# Assuming scripts/ is root level and not a package, we might need to add to path.
import sys
sys.path.append(".") 
from scripts.diagnose_lag_tail_bias import app

runner = CliRunner()

def test_diagnose_script_smoke(tmp_path):
    """Test standard execution with dummy data."""
    # Setup dummy environment
    eval_out = tmp_path / "eval_out"
    metrics_dir = eval_out / "metrics"
    summary_dir = eval_out / "summary"
    metrics_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    
    # 1. Create dummy summary/metrics_joined.csv
    # Use metrics_joined this time to test priority
    joined_csv = summary_dir / "metrics_joined.csv"
    with open(joined_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "piece_id", "lag_best", "num_bars", "lag_energy", "lag_base_period"])
        writer.writerow(["g1", "p1", "4", "32", "0.8", "4"])
        writer.writerow(["g1", "p2", "8", "32", "0.9", "4"])
        writer.writerow(["g2", "p3", "16", "32", "0.5", "16"])
        
    # 2. Create dummy metrics/lag_energies.jsonl
    jsonl_path = metrics_dir / "lag_energies.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps({"piece_id": "p1", "lag_energies": [0.1, 0.2, 0.8, 0.1, 0.9]}) + "\n")
        f.write(json.dumps({"piece_id": "p2", "lag_energies": [0.1, 0.1, 0.1, 0.9]}) + "\n")
        # p3 missing lag energies
        
    # Run script
    # Note: Default out_dir changed to eval_out/diagnostics/lag_tail_bias
    result = runner.invoke(app, ["--eval-out", str(eval_out)])
    
    print(result.stdout)
    assert result.exit_code == 0
    
    out_dir = eval_out / "diagnostics" / "lag_tail_bias"
    assert (out_dir / "lag_spectrum_mean_by_group.png").exists()
    assert (out_dir / "scatter_best_lag_vs_support.png").exists()
    assert (out_dir / "hist_best_lag.png").exists()
    assert (out_dir / "scatter_l0_vs_best_lag.png").exists()
    assert (out_dir / "lag_tail_bias_piece_table.csv").exists()

def test_diagnose_script_fallback(tmp_path):
    """Test fallback when summary csv is missing."""
    eval_out = tmp_path / "eval_out"
    metrics_dir = eval_out / "metrics"
    metrics_dir.mkdir(parents=True)
    
    # Create canonical metrics only
    metrics_csv = metrics_dir / "ssm_proxy.csv"
    with open(metrics_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "piece_id", "lag_best", "num_bars"])
        writer.writerow(["g1", "p1", "4", "32"])
        
    result = runner.invoke(app, ["--eval-out", str(eval_out)])
    assert result.exit_code == 0
    out_dir = eval_out / "diagnostics" / "lag_tail_bias"
    assert (out_dir / "hist_best_lag.png").exists()

