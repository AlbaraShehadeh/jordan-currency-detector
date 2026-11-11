"""Download a dataset version export from Roboflow's REST API (no SDK dependency).

The API key is read from settings (env / .env) and is never logged or written to disk.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

API_URL = "https://api.roboflow.com"
SOURCE_FILE = "SOURCE.json"


class RoboflowError(RuntimeError):
    pass


def get_export_link(
    client: httpx.Client, workspace: str, project: str, version: int, fmt: str, api_key: str
) -> str:
    resp = client.get(
        f"{API_URL}/{workspace}/{project}/{version}/{fmt}", params={"api_key": api_key}
    )
    if resp.status_code in (401, 403):
        raise RoboflowError("Roboflow rejected the API key (check ROBOFLOW_API_KEY)")
    if resp.status_code == 404:
        raise RoboflowError(f"not found: {workspace}/{project}/{version} ({fmt})")
    resp.raise_for_status()
    link = resp.json().get("export", {}).get("link")
    if not link:
        raise RoboflowError(
            f"no export link for {workspace}/{project}/{version} in format {fmt!r} "
            "(generate the export once in the Roboflow UI)"
        )
    return str(link)


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract, refusing entries that would escape `dest` (zip-slip)."""
    root = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not target.is_relative_to(root):
            raise RoboflowError(f"unsafe path in archive: {member.filename!r}")
    zf.extractall(dest)


def download_export(
    workspace: str,
    project: str,
    version: int,
    dest: Path,
    *,
    api_key: str,
    fmt: str = "yolov8",
    overwrite: bool = False,
    client: httpx.Client | None = None,
) -> Path:
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"{dest} exists (use --overwrite)")
        shutil.rmtree(dest)
    own = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(30, read=300), follow_redirects=True)
    try:
        link = get_export_link(client, workspace, project, version, fmt, api_key)
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "export.zip"
            with client.stream("GET", link) as resp, zip_path.open("wb") as fh:
                resp.raise_for_status()
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
            staging = Path(tmp) / "out"
            with zipfile.ZipFile(zip_path) as zf:
                safe_extract(zf, staging)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), dest)
    finally:
        if own:
            client.close()
    (dest / SOURCE_FILE).write_text(
        json.dumps(
            {
                "provider": "roboflow",
                "workspace": workspace,
                "project": project,
                "version": version,
                "format": fmt,
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dest
