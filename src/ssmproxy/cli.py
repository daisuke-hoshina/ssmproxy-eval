from pathlib import Path
from typing import Optional

import typer

from .pipeline import RunConfig, run_evaluation
from .toy_generator import generate_corpus

app = typer.Typer(help="Command line interface for the ssmproxy toolkit.")


@app.command()
def eval(
    dir: Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True),
    output: Path = typer.Option(Path("outputs"), "--output", "-o", resolve_path=True),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    novelty_L: int = typer.Option(8, help="Half-size of the novelty kernel."),
    lag_top_k: int = typer.Option(2, help="Number of lag energies to aggregate."),
) -> None:
    """Run evaluations for the given directory."""

    config = RunConfig(
        input_dir=dir,
        output_root=output,
        run_id=run_id,
        novelty_L=novelty_L,
        lag_top_k=lag_top_k,
    )
    result_dir = run_evaluation(config)
    typer.echo(f"Saved evaluation artifacts to {result_dir}")


toy_app = typer.Typer(help="Commands for toy examples and utilities.")


@toy_app.command("generate")
def toy_generate(
    out_dir: Path = typer.Option(..., "--out-dir", "-o", resolve_path=True),
    variants: int = typer.Option(1, "--variants", "-n", help="Number of variants per pattern."),
    seed: int = typer.Option(0, "--seed", help="Seed for reproducible generation."),
) -> None:
    """Generate toy MIDI corpora demonstrating structural patterns."""

    pieces = generate_corpus(out_dir, variants=variants, seed=seed)
    typer.echo(f"Wrote {len(pieces)} pieces to {out_dir}")
    typer.echo(f"Manifest saved to {out_dir / 'manifest.csv'}")


pc_app = typer.Typer(help="Commands related to proxy compute operations.")


@pc_app.command("run")
def pc_run(config: Optional[Path] = typer.Option(None, "--config", "-c", resolve_path=True)) -> None:
    """Run proxy compute tasks using the provided configuration."""
    details = f" using config {config}" if config else " with default settings"
    typer.echo(f"Starting proxy compute pipeline{details}.")


app.add_typer(toy_app, name="toy")
app.add_typer(pc_app, name="pc")


if __name__ == "__main__":
    app()
