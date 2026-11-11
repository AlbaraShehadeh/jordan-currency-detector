from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from jcd import __version__
from jcd.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_catalog_show_default() -> None:
    result = runner.invoke(app, ["catalog", "show"])
    assert result.exit_code == 0
    assert "half-dinar" in result.stdout
    assert "9 denominations (JOD)" in result.stdout


def test_catalog_show_custom(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "currency: USD\ndenominations:\n  - {id: dime, label: Dime, value: '0.10', kind: coin}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["catalog", "show", "--catalog", str(path)])
    assert result.exit_code == 0
    assert "1 denominations (USD)" in result.stdout


def test_catalog_check_ok() -> None:
    result = runner.invoke(app, ["catalog", "check", "1⁄2-dinar", "5-Piastres"])
    assert result.exit_code == 0
    assert "half-dinar" in result.stdout


def test_catalog_check_unknown_exits_nonzero() -> None:
    result = runner.invoke(app, ["catalog", "check", "5-dinar", "euro"])
    assert result.exit_code == 1


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "catalog" in result.output
