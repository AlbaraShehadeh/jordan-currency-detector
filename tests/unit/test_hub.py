from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from jcd.catalog import Catalog
from jcd.cli import app
from jcd.data.audit import audit_dataset
from jcd.data.dataset import YoloDataset
from jcd.data.hub import CARD_NAME, dataset_card, fetch_dataset, push_dataset
from jcd.data.roboflow import SOURCE_FILE
from tests.data_factory import leaky_dataset

SOURCE = {"workspace": "jordancoinvision", "project": "coin-vision", "version": 5}


class FakeApi:
    def __init__(self, tags: list[str] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tags = tags or []

    def create_repo(self, repo_id: str, **kw: Any) -> str:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kw}))
        return f"https://huggingface.co/datasets/{repo_id}"

    def list_repo_refs(self, repo_id: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(tags=[SimpleNamespace(name=t) for t in self.tags])

    def upload_folder(self, **kw: Any) -> SimpleNamespace:
        self.calls.append(("upload_folder", kw))
        return SimpleNamespace(oid="abc123")

    def create_tag(self, repo_id: str, **kw: Any) -> None:
        self.calls.append(("create_tag", {"repo_id": repo_id, **kw}))


@pytest.fixture
def report_and_dirs(tmp_path: Path, catalog: Catalog) -> tuple[Any, Path, Path]:
    raw = leaky_dataset(tmp_path / "raw")
    (raw / SOURCE_FILE).write_text(json.dumps(SOURCE), "utf-8")
    return audit_dataset(YoloDataset.load(raw), catalog), raw, raw


def test_card_contents(report_and_dirs: tuple[Any, Path, Path]) -> None:
    report, _, _ = report_and_dirs
    card = dataset_card(report, SOURCE, "me/ds")
    assert card.startswith("---\nlicense: cc-by-4.0")
    assert "jordancoinvision/coin-vision` v5" in card
    assert "| `half-dinar` | 3 |" in card
    assert "**2 / 4**" in card  # reports whatever leakage the audit found
    assert 'snapshot_download("me/ds"' in card


def test_push_uploads_and_tags(report_and_dirs: tuple[Any, Path, Path]) -> None:
    report, ds_dir, raw = report_and_dirs
    api = FakeApi()
    url = push_dataset(ds_dir, report, raw, "me/ds", tag="v1.0", private=True, token="t", api=api)  # type: ignore[arg-type]
    assert url == "https://huggingface.co/datasets/me/ds"
    assert (ds_dir / CARD_NAME).exists()
    names = [c[0] for c in api.calls]
    assert names == ["create_repo", "upload_folder", "create_tag"]
    assert api.calls[0][1]["private"] is True
    assert api.calls[1][1]["folder_path"] == ds_dir
    assert api.calls[2][1] == {
        "repo_id": "me/ds",
        "tag": "v1.0",
        "revision": "abc123",
        "repo_type": "dataset",
    }


def test_push_refuses_existing_tag(report_and_dirs: tuple[Any, Path, Path]) -> None:
    report, ds_dir, raw = report_and_dirs
    with pytest.raises(ValueError, match="already exists"):
        push_dataset(
            ds_dir,
            report,
            raw,
            "me/ds",
            tag="v1.0",
            private=False,
            token="t",
            api=FakeApi(["v1.0"]),
        )  # type: ignore[arg-type]


def test_fetch_pins_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(repo_id: str, **kw: Any) -> str:
        seen.update(repo_id=repo_id, **kw)
        return str(kw["local_dir"])

    monkeypatch.setattr("jcd.data.hub.snapshot_download", fake)
    assert fetch_dataset("me/ds", "v1.0", tmp_path / "d") == tmp_path / "d"
    assert seen["revision"] == "v1.0"
    assert seen["repo_type"] == "dataset"


# ------------------------------------------------------------------ CLI
runner = CliRunner()


def _cfg(tmp_path: Path, processed: Path, hub: bool = True) -> Path:
    data: dict[str, Any] = {
        "source": {"workspace": "w", "project": "p", "version": 1},
        "raw_dir": str(processed),
        "processed_dir": str(processed),
    }
    if hub:
        data["hub"] = {"repo_id": "me/ds", "tag": "v1.0"}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), "utf-8")
    return path


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return tmp_path


def test_cli_push_requires_hub_section_and_token(clean_env: Path) -> None:
    ds = leaky_dataset(clean_env / "ds")
    no_hub = runner.invoke(app, ["data", "push", "--config", str(_cfg(clean_env, ds, hub=False))])
    assert no_hub.exit_code == 2
    assert "hub" in no_hub.output
    no_token = runner.invoke(app, ["data", "push", "--config", str(_cfg(clean_env, ds))])
    assert no_token.exit_code == 2
    assert "HF_TOKEN" in no_token.output


def test_cli_push_refuses_leaky_dataset(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "t")
    ds = leaky_dataset(clean_env / "ds")
    result = runner.invoke(app, ["data", "push", "--config", str(_cfg(clean_env, ds))])
    assert result.exit_code == 1
    assert "leakage" in result.output


def test_cli_push_and_fetch(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jcd.data import hub

    monkeypatch.setenv("HF_TOKEN", "t")
    ds = leaky_dataset(clean_env / "ds")
    for p in (ds / "valid").rglob("*"):  # drop eval splits -> no leakage
        p.unlink() if p.is_file() else None
    for p in (ds / "test").rglob("*"):
        p.unlink() if p.is_file() else None
    monkeypatch.setattr(hub, "push_dataset", lambda *a, **k: "https://hf/ds")
    pushed = runner.invoke(app, ["data", "push", "--config", str(_cfg(clean_env, ds))])
    assert pushed.exit_code == 0, pushed.output
    assert "me/ds@v1.0" in pushed.stdout

    def raise_exists(*a: Any, **k: Any) -> str:
        raise ValueError("tag 'v1.0' already exists")

    monkeypatch.setattr(hub, "push_dataset", raise_exists)
    again = runner.invoke(app, ["data", "push", "--config", str(_cfg(clean_env, ds))])
    assert again.exit_code == 1

    monkeypatch.setattr(hub, "fetch_dataset", lambda repo, tag, dest, token=None: dest)
    fetched = runner.invoke(app, ["data", "fetch", "--config", str(_cfg(clean_env, ds))])
    assert fetched.exit_code == 0, fetched.output
    assert "Fetched me/ds@v1.0" in fetched.stdout
