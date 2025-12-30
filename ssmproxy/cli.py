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
    lag_max_lag: Optional[int] = typer.Option(None, "--lag-max-lag", help="Maximum lag to consider."),
    lag_min_support: Optional[int] = typer.Option(
        None, "--lag-min-support", help="Minimum support (diagonal length) for lag energy calculation."
    ),
    exclude_drums: Optional[bool] = typer.Option(
        None,
        "--exclude-drums/--include-drums",
        help="Exclude percussion instruments when extracting features.",
        show_default=False,
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", resolve_path=True, help="Optional path to a configuration file."
    ),
    require_4_4: Optional[bool] = typer.Option(
        None,
        "--require-4-4/--allow-non-4-4",
        help="Only process pieces with 4/4 time signature throughout.",
    ),
    max_bars: Optional[int] = typer.Option(
        None, "--max-bars", help="Limit analysis to the first N bars."
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
        lag_max_lag=lag_max_lag or defaults["lag_max_lag"],
        lag_min_support=lag_min_support or defaults["lag_min_support"],
        exclude_drums=defaults["exclude_drums"] if exclude_drums is None else exclude_drums,
        ssm_weight_pch=defaults["ssm_weight_pch"],
        ssm_weight_onh=defaults["ssm_weight_onh"],
        ssm_map_to_unit_interval=defaults["ssm_map_to_unit_interval"],
        novelty_peak_prominence=defaults["novelty_peak_prominence"],
        novelty_peak_min_distance=defaults["novelty_peak_min_distance"],
        require_4_4=defaults["require_4_4"] if require_4_4 is None else require_4_4,
        max_bars=max_bars if max_bars is not None else defaults["max_bars"],
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



dataset_app = typer.Typer(help="Commands for dataset management (scan, collect, sample).")


@dataset_app.command("scan")
def dataset_scan(
    input_dir: Path = typer.Option(..., "--input-dir", "-i", exists=True, file_okay=False, resolve_path=True),
    out_csv: Path = typer.Option(..., "--out-csv", "-o", resolve_path=True),
    min_bars: int = typer.Option(128, "--min-bars", help="Minimum number of bars to keep."),
    require_4_4: bool = typer.Option(
        False, "--require-4-4/--allow-non-4-4", help="Require piece to be entirely 4/4."
    ),
    unknown_ts_is_4_4: bool = typer.Option(
        True, "--unknown-ts-is-4-4/--unknown-ts-is-not-4-4", help="Treat unknown TS as 4/4."
    ),
    exclude_drums: bool = typer.Option(
        True, "--exclude-drums/--include-drums", help="Exclude drum tracks from bar estimation."
    ),
    max_files: Optional[int] = typer.Option(None, "--max-files", help="Limit number of files scanned (debug)."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for file order."),
    write_all: bool = typer.Option(
        False, "--write-all", help="Write all files to CSV with selection status."
    ),
) -> None:
    """Scan a directory of MIDI files and extract metadata to CSV."""
    from .dataset_utils import scan_dataset

    scan_dataset(
        input_dir=input_dir,
        out_csv=out_csv,
        min_bars=min_bars,
        require_4_4=require_4_4,
        unknown_ts_is_4_4=unknown_ts_is_4_4,
        exclude_drums=exclude_drums,
        max_files=max_files,
        seed=seed,
        write_all=write_all,
    )


@dataset_app.command("collect")
def dataset_collect(
    in_csv: Path = typer.Option(..., "--in-csv", "-i", exists=True, dir_okay=False, resolve_path=True),
    out_dir: Path = typer.Option(..., "--out-dir", "-o", resolve_path=True),
    mode: str = typer.Option("symlink", "--mode", help="Copy mode: 'symlink' or 'copy'."),
    flatten: bool = typer.Option(
        False, "--flatten/--preserve-tree", help="Flatten structure or preserve relative paths."
    ),
    name_from: str = typer.Option("rel", "--name-from", help="Naming strategy if flattened: 'rel' or 'hash'."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without copying."),
    only_selected: bool = typer.Option(True, "--only-selected/--all-rows", help="Process only selected rows."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit number of files collected."),
) -> None:
    """Collect MIDI files from a scan CSV to a target directory."""
    from .dataset_utils import collect_dataset

    collect_dataset(
        in_csv=in_csv,
        out_dir=out_dir,
        mode=mode,
        flatten=flatten,
        name_from=name_from,
        dry_run=dry_run,
        only_selected=only_selected,
        limit=limit,
    )


@dataset_app.command("sample")
def dataset_sample(
    in_csv: Path = typer.Option(..., "--in-csv", "-i", exists=True, dir_okay=False, resolve_path=True),
    out_csv: Path = typer.Option(..., "--out-csv", "-o", resolve_path=True),
    n: int = typer.Option(..., "--n", "-n", help="Number of samples."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    only_selected: bool = typer.Option(True, "--only-selected/--all-rows", help="Sample from only selected rows."),
    shuffle: bool = typer.Option(True, "--shuffle/--no-shuffle", help="Shuffle output order."),
) -> None:
    """Sample a random subset of rows from the CSV."""
    from .dataset_utils import sample_dataset

    sample_dataset(
        in_csv=in_csv,
        out_csv=out_csv,
        n=n,
        seed=seed,
        only_selected=only_selected,
        shuffle_output=shuffle,
    )


app.add_typer(toy_app, name="toy")
app.add_typer(pc_app, name="pc")
app.add_typer(report_app, name="report")
app.add_typer(dataset_app, name="dataset")


if __name__ == "__main__":
    app()
