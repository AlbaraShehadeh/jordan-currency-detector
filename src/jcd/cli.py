"""`jcd` command-line interface. Pipelines (data, train, evaluate, serve) are added per phase."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jcd import __version__
from jcd.catalog import Catalog

app = typer.Typer(
    name="jcd",
    help="Jordan Currency Detector: detect, count and value Jordanian coins and banknotes.",
    no_args_is_help=True,
    add_completion=False,
)
catalog_app = typer.Typer(help="Inspect the currency catalog.", no_args_is_help=True)
app.add_typer(catalog_app, name="catalog")

CatalogOption = Annotated[
    Path | None,
    typer.Option("--catalog", "-c", exists=True, dir_okay=False, help="Custom catalog YAML."),
]


def _load(path: Path | None) -> Catalog:
    return Catalog.from_yaml(path) if path else Catalog.default()


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@catalog_app.command("show")
def catalog_show(catalog: CatalogOption = None) -> None:
    """List denominations, their kind, value and accepted aliases."""
    cat = _load(catalog)
    typer.echo(f"{'id':<15}{'label':<13}{'kind':<6}{'value':>8}  aliases")
    for d in cat:
        aliases = ", ".join(d.aliases) or "-"
        typer.echo(f"{d.id:<15}{d.label:<13}{d.kind:<6}{d.value:>8}  {aliases}")
    typer.echo(f"\n{len(cat)} denominations ({cat.currency})")


@catalog_app.command("check")
def catalog_check(
    names: Annotated[list[str], typer.Argument(help="Model class names to resolve.")],
    catalog: CatalogOption = None,
) -> None:
    """Verify that model class names all resolve to denominations (exit 1 if not)."""
    cat = _load(catalog)
    ok = True
    for name in names:
        if name in cat:
            typer.echo(f"ok       {name!r} -> {cat.resolve(name).id}")
        else:
            ok = False
            typer.echo(f"UNKNOWN  {name!r}", err=True)
    raise typer.Exit(code=0 if ok else 1)
