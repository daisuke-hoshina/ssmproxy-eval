from pathlib import Path
from typing import Optional

import typer

from .config import DEFAULT_CONFIG_PATH, get_run_defaults, load_config
from .pipeline import RunConfig, run_evaluation
from .report import generate_report, resolve_metrics_csv
from .toy_generator import generate_corpus

app = typer.Typer(help="Command line interface for the ssmproxy toolkit.")


@app.command()
def eval(
    dir: Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o", resolve_path=True),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    novelty_L: Optional[int] = typer.Option(
        None, "--novelty-l", "--novelty-L", help="Half-size of the novelty kernel."
    ),
    lag_top_k: Optional[int] = typer.Option(None, help="Number of lag energies to aggregate."),
    lag_min_lag: Optional[int] = typer.Option(None, "--lag-min-lag", help="Minimum lag to consider."),
    exclude_drums: Optional[bool] = typer.Option(
        None,
        "--exclude-drums/--include-drums",
        help="Exclude percussion instruments when extracting features.",
        show_default=False,
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", resolve_path=True, help="Optional path to a configuration file."
    ),
) -> None:
    """Run evaluations for the given directory."""

    config_data = load_config(config_path or DEFAULT_CONFIG_PATH)
    defaults = get_run_defaults(config_data)

    config = RunConfig(
        input_dir=dir,
        output_root=output or Path(defaults["output_root"]),
        run_id=run_id,
        novelty_L=novelty_L or defaults["novelty_L"],
        lag_top_k=lag_top_k or defaults["lag_top_k"],
        lag_min_lag=lag_min_lag or defaults["lag_min_lag"],
        exclude_drums=defaults["exclude_drums"] if exclude_drums is None else exclude_drums,
    )
    result_dir = run_evaluation(config)
    typer.echo(f"Saved evaluation artifacts to {result_dir}")


toy_app = typer.Typer(help="Commands for toy examples and utilities.")


@toy_app.command("generate")
def toy_generate(
    out_dir: Path = typer.Option(..., "--out-dir", "-o", resolve_path=True),
    variants: int = typer.Option(1, "--variants", "-n", help="Number of variants per pattern."),
    seed: int = typer.Option(0, "--seed", help="Seed for reproducible generation."),
    flat: bool = typer.Option(False, "--flat", help="Output files to a flat directory (legacy behavior)."),
) -> None:
    """Generate toy MIDI corpora demonstrating structural patterns."""

    pieces = generate_corpus(out_dir, variants=variants, seed=seed, flat=flat)
    typer.echo(f"Wrote {len(pieces)} pieces to {out_dir}")
    typer.echo(f"Manifest saved to {out_dir / 'manifest.csv'}")


pc_app = typer.Typer(help="Commands related to proxy compute operations.")


@pc_app.command("run")
def pc_run(config: Optional[Path] = typer.Option(None, "--config", "-c", resolve_path=True)) -> None:
    """Run proxy compute tasks using the provided configuration."""
    details = f" using config {config}" if config else " with default settings"
    typer.echo(f"Starting proxy compute pipeline{details}.")


report_app = typer.Typer(help="Commands for generating summary reports.")


@report_app.command("run")
def report_run(
    eval_out: Path = typer.Option(..., "--eval-out", "-e", resolve_path=True, help="Path to an evaluation output dir."),
    metrics_csv: Optional[Path] = typer.Option(
        None,
        "--metrics-csv",
        "-m",
        resolve_path=True,
        help="Path to metrics CSV (defaults to <eval_out>/metrics/ssm_proxy.csv, falling back to metrics.csv).",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", "-o", resolve_path=True, help="Directory to write summary outputs."
    ),
    group_col: str = typer.Option("group", "--group-col", help="Column to use for grouping."),
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        resolve_path=True,
        help="Optional manifest.csv containing extra metadata to join on piece_id.",
    ),
) -> None:
    """Generate grouped summaries and figures from the metrics CSV."""

    metrics_path = resolve_metrics_csv(eval_out, metrics_csv)
    summary_dir = out_dir or eval_out / "summary"
    artifacts = generate_report(metrics_csv=metrics_path, out_dir=summary_dir, group_col=group_col, manifest=manifest)
    typer.echo(f"Wrote summary artifacts to {summary_dir}")
    typer.echo(f"Joined metrics: {artifacts['metrics_joined']}")
    typer.echo(f"Group stats: {artifacts['group_stats']}")


app.add_typer(toy_app, name="toy")
app.add_typer(pc_app, name="pc")
app.add_typer(report_app, name="report")


if __name__ == "__main__":
    app()
