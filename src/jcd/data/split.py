"""Leakage-free, class-balanced split by source group, and materialization to a YOLO dataset.

The output dataset uses the catalog's canonical class ids in catalog order, so every model
trained from it emits canonical names (`half-dinar`, not `1⁄2-dinar`).
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from jcd.catalog import Catalog
from jcd.data.dataset import SPLITS, YoloBox, YoloDataset, format_label_rows, parse_label_file

MANIFEST_NAME = "manifest.csv"


def grouped_split(
    group_of: Sequence[int],
    class_counts: Sequence[Mapping[str, int]],
    ratios: Mapping[str, float],
    *,
    seed: int = 42,
) -> list[str]:
    """Assign each item a split such that all items of a group share it.

    Greedy: groups in descending size (seeded shuffle breaks ties) go to the split with the
    largest relative deficit in image count *and* in the classes the group contains.
    """
    if not ratios or any(r < 0 for r in ratios.values()) or sum(ratios.values()) <= 0:
        raise ValueError(f"invalid ratios {dict(ratios)}")
    total_r = sum(ratios.values())
    splits = [s for s, r in ratios.items() if r > 0]
    frac = {s: ratios[s] / total_r for s in splits}

    groups: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(group_of):
        groups[g].append(i)
    g_cls: dict[int, Counter[str]] = {}
    for g, idx in groups.items():
        c: Counter[str] = Counter()
        for i in idx:
            c.update(class_counts[i])
        g_cls[g] = c

    n = len(group_of)
    cls_total: Counter[str] = Counter()
    for c in g_cls.values():
        cls_total.update(c)

    order = list(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: len(groups[g]), reverse=True)

    cur_n: Counter[str] = Counter()
    cur_c: dict[str, Counter[str]] = {s: Counter() for s in splits}
    assign: dict[int, str] = {}
    for g in order:

        def need(s: str, g: int = g) -> float:
            img_need = (frac[s] * n - cur_n[s]) / max(frac[s] * n, 1e-9)
            cls = g_cls[g]
            if not cls:
                return img_need
            cls_need = sum(
                (frac[s] * cls_total[k] - cur_c[s][k]) / max(frac[s] * cls_total[k], 1e-9)
                for k in cls
            ) / len(cls)
            return img_need + cls_need

        best = max(splits, key=need)
        assign[g] = best
        cur_n[best] += len(groups[g])
        cur_c[best].update(g_cls[g])
    return [assign[g] for g in group_of]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _place(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # hardlink: no extra disk, same inode
    except OSError:
        shutil.copy2(src, dst)


def materialize_split(
    ds: YoloDataset,
    assignment: Sequence[str],
    group_of: Sequence[int],
    out_dir: Path,
    catalog: Catalog,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a YOLO dataset with canonical class ids plus a manifest (image, split, group,
    sha256, source path). Returns the path of the generated data.yaml."""
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (use overwrite=True)")
        shutil.rmtree(out_dir)
    mapping = catalog.map_model_classes(ds.names)
    remap = {i: catalog.ids.index(mapping[name]) for i, name in enumerate(ds.names)}

    rows = []
    for item, split, group in zip(ds.items, assignment, group_of, strict=True):
        img_dir, lbl_dir = out_dir / split / "images", out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        dst = img_dir / item.path.name
        _place(item.path, dst)
        boxes = [
            YoloBox(remap[b.cls], b.xc, b.yc, b.w, b.h) for b in parse_label_file(item.label_path)
        ]
        (lbl_dir / f"{item.path.stem}.txt").write_text(format_label_rows(boxes), encoding="utf-8")
        rows.append(
            {
                "image": dst.relative_to(out_dir).as_posix(),
                "split": split,
                "group": group,
                "sha256": _sha256(dst),
                "source": item.path.relative_to(ds.root).as_posix(),
            }
        )

    present = [s for s in SPLITS if any(r["split"] == s for r in rows)]
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                **{("val" if s == "valid" else s): f"{s}/images" for s in present},
                "names": dict(enumerate(catalog.ids)),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    with (out_dir / MANIFEST_NAME).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "split", "group", "sha256", "source"])
        writer.writeheader()
        writer.writerows(rows)
    return data_yaml
