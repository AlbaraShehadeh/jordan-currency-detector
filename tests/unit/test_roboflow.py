from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from jcd.data.roboflow import SOURCE_FILE, RoboflowError, download_export

KEY = "test-key-not-real"
EXPORT_LINK = "https://storage.example.com/export.zip"


def _zip(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _client(api_status: int = 200, api_json: object = None, archive: bytes = b"") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.roboflow.com":
            assert request.url.path == "/ws/proj/5/yolov8"
            assert request.url.params["api_key"] == KEY
            body = api_json if api_json is not None else {"export": {"link": EXPORT_LINK}}
            return httpx.Response(api_status, json=body)
        assert str(request.url) == EXPORT_LINK
        return httpx.Response(200, content=archive)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _download(dest: Path, client: httpx.Client, **kw: bool) -> Path:
    return download_export("ws", "proj", 5, dest, api_key=KEY, client=client, **kw)


def test_download_extracts_and_records_source_without_key(tmp_path: Path) -> None:
    archive = _zip({"data.yaml": "names: [a]\n", "train/images/x.jpg": "img"})
    dest = _download(tmp_path / "raw", _client(archive=archive))
    assert (dest / "data.yaml").read_text() == "names: [a]\n"
    assert (dest / "train/images/x.jpg").exists()
    source = json.loads((dest / SOURCE_FILE).read_text())
    assert source["project"] == "proj"
    assert source["version"] == 5
    assert KEY not in (dest / SOURCE_FILE).read_text()


@pytest.mark.parametrize(
    ("status", "body", "match"),
    [
        (401, {}, "rejected the API key"),
        (403, {}, "rejected the API key"),
        (404, {}, "not found"),
        (200, {"export": {}}, "no export link"),
    ],
)
def test_api_errors(tmp_path: Path, status: int, body: object, match: str) -> None:
    with pytest.raises(RoboflowError, match=match):
        _download(tmp_path / "raw", _client(api_status=status, api_json=body))
    assert not (tmp_path / "raw").exists()


def test_server_error_raises(tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        _download(tmp_path / "raw", _client(api_status=500, api_json={}))


def test_zip_slip_rejected(tmp_path: Path) -> None:
    with pytest.raises(RoboflowError, match="unsafe path"):
        _download(tmp_path / "raw", _client(archive=_zip({"../evil.txt": "x"})))
    assert not (tmp_path / "evil.txt").exists()


def test_existing_dest(tmp_path: Path) -> None:
    dest = tmp_path / "raw"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    with pytest.raises(FileExistsError):
        _download(dest, _client(archive=_zip({"a.txt": "a"})))
    _download(dest, _client(archive=_zip({"a.txt": "a"})), overwrite=True)
    assert not (dest / "old.txt").exists()
    assert (dest / "a.txt").exists()
