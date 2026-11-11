"""Build tiny synthetic YOLO datasets on disk for data-pipeline tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

# Raw class names as in the original coin-vision export (U+2044 fraction slash).
RAW_NAMES = ["10-Piastres", "1⁄2-dinar", "5-dinar"]


def scene(seed: int, size: int = 96) -> Image.Image:
    """A smooth random 'photo': low-res noise upscaled, so distinct seeds hash far apart."""
    rng = np.random.default_rng(seed)
    small = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
    return Image.fromarray(small).resize((size, size), Image.Resampling.BICUBIC)


@dataclass
class Img:
    split: str
    stem: str
    seed: int
    mirror: bool = False
    labels: list[str] = field(default_factory=lambda: ["0 0.5 0.5 0.2 0.2"])
    write_label: bool = True


def build(root: Path, images: list[Img], names: list[str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names or RAW_NAMES}), "utf-8")
    for im in images:
        img_dir, lbl_dir = root / im.split / "images", root / im.split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        pic = scene(im.seed)
        if im.mirror:
            pic = ImageOps.mirror(pic)
        pic.save(img_dir / f"{im.stem}.jpg", quality=95)
        if im.write_label:
            (lbl_dir / f"{im.stem}.txt").write_text("\n".join(im.labels) + "\n", "utf-8")
    return root


def leaky_dataset(root: Path) -> Path:
    """6 source photos. Leaks: src_a copy in valid (same Roboflow source id), src_b mirrored
    re-upload under a new name in test (pHash only). src_c..src_f are clean."""
    return build(
        root,
        [
            Img("train", "src_a_jpg.rf.aaa111", 1),
            Img("valid", "src_a_jpg.rf.bbb222", 1, labels=["1 0.5 0.5 0.2 0.2"]),
            Img("train", "src_b_jpg.rf.ccc333", 2),
            Img("test", "renamed_b_jpg.rf.ddd444", 2, mirror=True),
            Img(
                "train", "src_c_jpg.rf.eee555", 3, labels=["2 0.5 0.5 0.3 0.3", "0 0.2 0.2 0.1 0.1"]
            ),
            Img("valid", "src_d_jpg.rf.fff666", 4, labels=["1 0.4 0.4 0.2 0.2"]),
            Img("test", "src_e_jpg.rf.abc777", 5, labels=["2 0.5 0.5 0.2 0.2"]),
            Img(
                "train", "src_f_jpg.rf.def888", 6, labels=["0 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.1"]
            ),
        ],
    )
