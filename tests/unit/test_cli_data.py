from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from jcd.cli import app
from jcd.settings import DataConfig
from tests.data_factory import leaky_dataset

runner = CliRunner()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # no stray .env; relative report paths land in tmp
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    return tmp_path


def _config(workdir: Path, raw: Path) -> Path:
    cfg = workdir / "data.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "source": {"workspace": "ws", "project": "proj", "version": 5},
                "raw_dir": str(raw),
                "processed_dir": "processed/out",
                "split": {"ratios": {"train": 0.5, "valid": 0.25, "test": 0.25}, "seed": 0},
            }
        ),
        "utf-8",
    )
    return cfg


def test_audit_reports_leakage(workdir: Path) -> None:
    raw = leaky_dataset(workdir / "raw")
    result = runner.invoke(app, ["data", "audit", str(raw)])
    assert result.exit_code == 0, result.output
    assert "2/4 (50.0%)" in result.stdout
    assert (workdir / "reports/data/raw/audit.md").exists()


def test_split_produces_leak_free_dataset(workdir: Path) -> None:
    cfg = _config(workdir, leaky_dataset(workdir / "raw"))
    result = runner.invoke(app, ["data", "split", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "0/" in result.stdout
    assert (workdir / "processed/out/data.yaml").exists()
    again = runner.invoke(app, ["data", "split", "--config", str(cfg)])
    assert again.exit_code == 1
    assert "exists" in again.output


def test_pull_requires_api_key(workdir: Path) -> None:
    cfg = _config(workdir, workdir / "raw")
    result = runner.invoke(app, ["data", "pull", "--config", str(cfg)])
    assert result.exit_code == 2
    assert "ROBOFLOW_API_KEY" in result.output


def test_pull_reports_api_errors(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jcd.data import roboflow

    def boom(*args: object, **kwargs: object) -> Path:
        raise roboflow.RoboflowError("Roboflow rejected the API key")

    monkeypatch.setenv("ROBOFLOW_API_KEY", "k")
    monkeypatch.setattr(roboflow, "download_export", boom)
    result = runner.invoke(app, ["data", "pull", "--config", str(_config(workdir, workdir / "r"))])
    assert result.exit_code == 1
    assert "rejected" in result.output


def test_pull_success(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jcd.data import roboflow

    seen: dict[str, object] = {}

    def fake(workspace: str, project: str, version: int, dest: Path, **kw: object) -> Path:
        seen.update(workspace=workspace, version=version, api_key=kw["api_key"])
        return dest

    monkeypatch.setenv("ROBOFLOW_API_KEY", "secret-k")
    monkeypatch.setattr(roboflow, "download_export", fake)
    result = runner.invoke(app, ["data", "pull", "--config", str(_config(workdir, workdir / "r"))])
    assert result.exit_code == 0, result.output
    assert seen == {"workspace": "ws", "version": 5, "api_key": "secret-k"}
    assert "secret-k" not in result.output


def test_repo_data_config_is_valid() -> None:
    cfg = DataConfig.from_yaml(Path(__file__).parents[2] / "configs" / "data.yaml")
    assert cfg.source.project == "coin-vision"
    assert cfg.phash_threshold == 6


def test_config_rejects_unknown_split_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown split"):
        DataConfig.model_validate(
            {
                "source": {"workspace": "w", "project": "p", "version": 1},
                "raw_dir": "r",
                "processed_dir": "p",
                "split": {"ratios": {"training": 1.0}},
            }
        )
