"""Command-line entry point."""

import typer

from costless import __version__

app = typer.Typer(
    name="costless",
    help="Evaluate an LLM application and gate merges on quality, cost and latency.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """costless command-line interface."""


@app.command()
def version() -> None:
    """Print the installed costless version."""
    typer.echo(__version__)
