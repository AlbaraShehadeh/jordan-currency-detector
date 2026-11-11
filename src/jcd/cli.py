"""`jcd` command-line interface. Pipelines (data, train, evaluate, serve) are added per phase."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from jcd import __version__
from jcd.catalog import Catalog

if TYPE_CHECKING:  # heavy modules are imported lazily inside commands
    from jcd.settings import DataConfig as DataConfigT
    from jcd.settings import HubConfig as HubConfigT

app = typer.Typer(
    name="jcd",
    help="Jordan Currency Detector: detect, count and value Jordanian coins and banknotes.",
    no_args_is_help=True,
    add_completion=False,
)
catalog_app = typer.Typer(help="Inspect the currency catalog.", no_args_is_help=True)
app.add_typer(catalog_app, name="catalog")
data_app = typer.Typer(help="Dataset pipeline: pull -> audit -> split.", no_args_is_help=True)
app.add_typer(data_app, name="data")

DEFAULT_DATA_CONFIG = Path("configs/data.yaml")
DEFAULT_REPORTS_DIR = Path("reports/data")

CatalogOption = Annotated[
    Path | None,
    typer.Option("--catalog", "-c", exists=True, dir_okay=False, help="Custom catalog YAML."),
]
DataConfigOption = Annotated[
    Path, typer.Option("--config", exists=True, dir_okay=False, help="Data pipeline config.")
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


# ---------------------------------------------------------------------------- data
@data_app.command("pull")
def data_pull(
    config: DataConfigOption = DEFAULT_DATA_CONFIG,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing download.")] = False,
) -> None:
    """Download the configured Roboflow dataset version into `raw_dir`."""
    from jcd.data.roboflow import RoboflowError, download_export
    from jcd.settings import DataConfig, Settings

    cfg = DataConfig.from_yaml(config)
    key = Settings().roboflow_api_key
    if key is None:
        typer.echo("ROBOFLOW_API_KEY is not set (add it to .env; see .env.example)", err=True)
        raise typer.Exit(2)
    src = cfg.source
    typer.echo(f"Downloading {src.workspace}/{src.project} v{src.version} ({src.format}) ...")
    try:
        dest = download_export(
            src.workspace,
            src.project,
            src.version,
            cfg.raw_dir,
            api_key=key.get_secret_value(),
            fmt=src.format,
            overwrite=overwrite,
        )
    except (RoboflowError, FileExistsError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Saved to {dest}")


def _summarize_audit(report_md: Path, leaked: int, total: int, groups: int, images: int) -> None:
    pct = f"{leaked / total:.1%}" if total else "n/a"
    typer.echo(f"images: {images}   source groups: {groups}")
    typer.echo(f"eval images leaked from train: {leaked}/{total} ({pct})")
    typer.echo(f"report: {report_md}")


@data_app.command("audit")
def data_audit(
    dataset: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="YOLO dataset dir.")
    ],
    out: Annotated[Path | None, typer.Option(help="Report dir [reports/data/<name>].")] = None,
    threshold: Annotated[int, typer.Option(help="pHash near-duplicate threshold (bits).")] = 6,
    catalog: CatalogOption = None,
) -> None:
    """Audit labels, class balance, near-duplicates and train->eval leakage."""
    from jcd.data.audit import audit_dataset, write_report
    from jcd.data.dataset import YoloDataset
    from jcd.data.grouping import group_dataset

    ds = YoloDataset.load(dataset)
    report = audit_dataset(ds, _load(catalog), grouping=group_dataset(ds, threshold=threshold))
    _, md = write_report(report, out or DEFAULT_REPORTS_DIR / dataset.name)
    lk = report.leakage
    _summarize_audit(md, lk.eval_images_leaked, lk.eval_images, lk.n_groups, len(ds.items))


@data_app.command("split")
def data_split(
    config: DataConfigOption = DEFAULT_DATA_CONFIG,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing processed dataset.")] = False,
    catalog: CatalogOption = None,
) -> None:
    """Group near-duplicates, split by group, write a canonical YOLO dataset, then re-audit it
    (exit 1 if any eval image still shares a source group with train)."""
    from jcd.data.audit import audit_dataset, write_report
    from jcd.data.dataset import YoloDataset, parse_label_file
    from jcd.data.grouping import group_dataset
    from jcd.data.split import grouped_split, materialize_split
    from jcd.settings import DataConfig

    cfg = DataConfig.from_yaml(config)
    cat = _load(catalog)
    ds = YoloDataset.load(cfg.raw_dir)
    mapping = cat.map_model_classes(ds.names)
    grouping = group_dataset(ds, threshold=cfg.phash_threshold)
    class_counts = [
        Counter(mapping[ds.names[b.cls]] for b in parse_label_file(it.label_path))
        for it in ds.items
    ]
    assignment = grouped_split(
        grouping.group_of, class_counts, cfg.split.ratios, seed=cfg.split.seed
    )
    try:
        data_yaml = materialize_split(
            ds, assignment, grouping.group_of, cfg.processed_dir, cat, overwrite=overwrite
        )
    except FileExistsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Wrote {data_yaml}  ({Counter(assignment)})")

    out = YoloDataset.load(cfg.processed_dir)
    report = audit_dataset(out, cat, grouping=group_dataset(out, threshold=cfg.phash_threshold))
    _, md = write_report(report, DEFAULT_REPORTS_DIR / cfg.processed_dir.name)
    lk = report.leakage
    _summarize_audit(md, lk.eval_images_leaked, lk.eval_images, lk.n_groups, len(out.items))
    if lk.eval_images_leaked:
        typer.echo("LEAKAGE remains after split", err=True)
        raise typer.Exit(1)


def _hub_config(config: Path) -> tuple[DataConfigT, HubConfigT]:
    from jcd.settings import DataConfig

    cfg = DataConfig.from_yaml(config)
    if cfg.hub is None:
        typer.echo(f"no `hub:` section in {config}", err=True)
        raise typer.Exit(2)
    return cfg, cfg.hub


@data_app.command("push")
def data_push(
    config: DataConfigOption = DEFAULT_DATA_CONFIG, catalog: CatalogOption = None
) -> None:
    """Upload the processed dataset (+ dataset card) to the Hugging Face Hub and tag it.
    Re-audits first and refuses to push if any leakage is found."""
    from jcd.data.audit import audit_dataset
    from jcd.data.dataset import YoloDataset
    from jcd.data.grouping import group_dataset
    from jcd.data.hub import push_dataset
    from jcd.settings import Settings

    cfg, hub = _hub_config(config)
    token = Settings().hf_token
    if token is None:
        typer.echo("HF_TOKEN is not set (add it to .env; see .env.example)", err=True)
        raise typer.Exit(2)
    ds = YoloDataset.load(cfg.processed_dir)
    report = audit_dataset(
        ds, _load(catalog), grouping=group_dataset(ds, threshold=cfg.phash_threshold)
    )
    if report.leakage.eval_images_leaked:
        typer.echo("refusing to push: processed dataset has train->eval leakage", err=True)
        raise typer.Exit(1)
    try:
        url = push_dataset(
            cfg.processed_dir,
            report,
            cfg.raw_dir,
            hub.repo_id,
            tag=hub.tag,
            private=hub.private,
            token=token.get_secret_value(),
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Pushed {hub.repo_id}@{hub.tag}: {url}")


@data_app.command("fetch")
def data_fetch(
    config: DataConfigOption = DEFAULT_DATA_CONFIG,
    dest: Annotated[Path | None, typer.Option(help="Target dir [processed_dir].")] = None,
) -> None:
    """Download the pinned dataset version from the Hugging Face Hub."""
    from jcd.data.hub import fetch_dataset
    from jcd.settings import Settings

    cfg, hub = _hub_config(config)
    token = Settings().hf_token
    path = fetch_dataset(
        hub.repo_id,
        hub.tag,
        dest or cfg.processed_dir,
        token=token.get_secret_value() if token else None,
    )
    typer.echo(f"Fetched {hub.repo_id}@{hub.tag} -> {path}")
