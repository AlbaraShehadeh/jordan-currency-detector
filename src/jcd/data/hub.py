"""Version processed datasets on the Hugging Face Hub (push with a tag, fetch a pinned tag)."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from jcd.data.audit import AuditReport
from jcd.data.roboflow import SOURCE_FILE

CARD_NAME = "README.md"


def dataset_card(report: AuditReport, source: dict[str, object], repo_id: str) -> str:
    """Hugging Face dataset card: provenance, split/class tables, leakage method, attribution."""
    ws, proj, ver = source.get("workspace"), source.get("project"), source.get("version")
    n_images = sum(s.images for s in report.splits.values())
    lk = report.leakage
    split_rows = "\n".join(
        f"| {name} | {s.images} | {s.boxes} |" for name, s in report.splits.items()
    )
    class_rows = "\n".join(
        f"| `{cid}` | {i} | "
        + " | ".join(str(report.splits[s].class_counts.get(cid, 0)) for s in report.splits)
        + " |"
        for i, cid in enumerate(report.class_order)
    )
    return f"""---
license: cc-by-4.0
task_categories:
- object-detection
tags:
- currency
- coins
- banknotes
- jordan
- yolo
size_categories:
- n<1K
pretty_name: Jordanian Currency Detection (coin-vision, leakage-free split)
---

# Jordanian Currency Detection: leakage-free split

{n_images} photos of Jordanian coins and banknotes with bounding boxes for **9 denominations**,
in **YOLO format**, re-split so that **no source photo appears in more than one split**.

Built by the [`jcd`](https://github.com/AlbaraShehadeh/jordan-currency-detector) data pipeline
(`jcd data pull → audit → split → push`).

## Splits

| split | images | boxes |
|---|---:|---:|
{split_rows}

Split: grouped by source photo, class-balanced, seeded. Leakage check on the output:
**{lk.eval_images_leaked} / {lk.eval_images}** eval images share a source group with train
({lk.n_groups} groups; {lk.method}).

## Classes

| id | index | {" | ".join(report.splits)} |
|---|---:|{"---:|" * len(report.splits)}
{class_rows}

Values (JOD): 5 and 10 piastres = 0.05 / 0.10; quarter/half/1 dinar = 0.25 / 0.50 / 1.00 (coins);
5 / 10 / 20 / 50 dinars (notes).

## Files

- `data.yaml`: Ultralytics dataset config (`names` in canonical order)
- `{{train,valid,test}}/images`, `{{train,valid,test}}/labels`: images + YOLO labels
  (`cls xc yc w h`, normalized)
- `manifest.csv`: per image: split, source group, sha256, original path

## Why this split exists

An earlier public version of this data (`coinsvision/jordan-coins-detection-nqdbs`) was an
*augmented export* that had been re-uploaded and randomly re-split: **88.5 % of its val/test images
were augmented copies of training photos**, inflating reported mAP. This dataset is built from the
original, un-augmented photos and split by source group.

## Source & license

Derived from Roboflow project [`{ws}/{proj}` v{ver}](https://universe.roboflow.com/{ws}/{proj})
("Real Data", no augmentation), licensed **CC BY 4.0**. Labels were clipped to the image bounds
(float-precision overshoot only); class names were canonicalized (e.g. `1-2-dinar` → `half-dinar`).

```python
from huggingface_hub import snapshot_download
path = snapshot_download("{repo_id}", repo_type="dataset", revision="<tag>")
```
"""


def push_dataset(
    dataset_dir: Path,
    report: AuditReport,
    raw_dir: Path,
    repo_id: str,
    *,
    tag: str,
    private: bool,
    token: str,
    api: HfApi | None = None,
) -> str:
    """Upload `dataset_dir` (+ generated card) to a dataset repo and tag the commit.
    Returns the repo URL. Refuses to move an existing tag."""
    api = api or HfApi(token=token)
    source_path = raw_dir / SOURCE_FILE
    source = json.loads(source_path.read_text("utf-8")) if source_path.exists() else {}
    (dataset_dir / CARD_NAME).write_text(dataset_card(report, source, repo_id), encoding="utf-8")

    url = api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    existing = {t.name for t in api.list_repo_refs(repo_id, repo_type="dataset").tags}
    if tag in existing:
        raise ValueError(f"tag {tag!r} already exists on {repo_id}; bump the version")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_dir,
        commit_message=f"Dataset {tag}",
        delete_patterns=["*"],  # the repo mirrors the folder exactly
    )
    api.create_tag(repo_id, tag=tag, revision=commit.oid, repo_type="dataset")
    return str(url)


def fetch_dataset(repo_id: str, tag: str, dest: Path, *, token: str | None = None) -> Path:
    """Download a pinned dataset version (cached by huggingface_hub)."""
    return Path(
        snapshot_download(repo_id, repo_type="dataset", revision=tag, local_dir=dest, token=token)
    )
