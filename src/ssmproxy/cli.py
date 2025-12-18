from pathlib import Path
from typing import Optional

import typer

from .pipeline import RunConfig, run_evaluation

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
def toy_generate(output: Optional[Path] = typer.Option(None, "--output", "-o", resolve_path=True)) -> None:
    """Generate toy data or configuration placeholders."""
    target = output if output else Path.cwd() / "toy-output"
    typer.echo(f"Generating toy artifacts at: {target}")


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
