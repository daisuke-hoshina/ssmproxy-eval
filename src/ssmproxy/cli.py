from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Command line interface for the ssmproxy toolkit.")


def _print_placeholder(message: str) -> None:
    """Print a standard placeholder message."""
    typer.echo(message)


@app.command()
def eval(dir: Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True)) -> None:
    """Run evaluations for the given directory."""
    _print_placeholder(f"Running evaluation workflows in: {dir}")


toy_app = typer.Typer(help="Commands for toy examples and utilities.")


@toy_app.command("generate")
def toy_generate(output: Optional[Path] = typer.Option(None, "--output", "-o", resolve_path=True)) -> None:
    """Generate toy data or configuration placeholders."""
    target = output if output else Path.cwd() / "toy-output"
    _print_placeholder(f"Generating toy artifacts at: {target}")


pc_app = typer.Typer(help="Commands related to proxy compute operations.")


@pc_app.command("run")
def pc_run(config: Optional[Path] = typer.Option(None, "--config", "-c", resolve_path=True)) -> None:
    """Run proxy compute tasks using the provided configuration."""
    details = f" using config {config}" if config else " with default settings"
    _print_placeholder(f"Starting proxy compute pipeline{details}.")


app.add_typer(toy_app, name="toy")
app.add_typer(pc_app, name="pc")


if __name__ == "__main__":
    app()
