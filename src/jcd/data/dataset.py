"""Read YOLO-format datasets (Roboflow / Ultralytics layout).

root/
  data.yaml            # names: [...]  (list or {idx: name})
  train/images/*.jpg   train/labels/*.txt
  valid/images/...     (or val/)
  test/images/...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SPLITS: tuple[str, ...] = ("train", "valid", "test")
_SPLIT_DIR_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "valid": ("valid", "val"),
    "test": ("test",),
}
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")


@dataclass(frozen=True, slots=True)
class YoloBox:
    """One label row: class index + normalized center/size."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def clipped(self) -> YoloBox | None:
        """The box clipped to the image ([0, 1]); None if nothing remains."""
        lo_x, hi_x = self.xc - self.w / 2, self.xc + self.w / 2
        lo_y, hi_y = self.yc - self.h / 2, self.yc + self.h / 2
        if lo_x >= 0 and lo_y >= 0 and hi_x <= 1 and hi_y <= 1:
            return self  # untouched: avoid float round-trip drift
        x1, x2 = max(0.0, lo_x), min(1.0, hi_x)
        y1, y2 = max(0.0, lo_y), min(1.0, hi_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return YoloBox(self.cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


@dataclass(frozen=True, slots=True)
class ImageItem:
    path: Path
    split: str
    label_path: Path

    @property
    def source_id(self) -> str:
        """Original upload name. Roboflow renames exports to `<orig>_<ext>.rf.<hash>`, so all
        augmented copies of one source photo share this id."""
        return _ROBOFLOW_SUFFIX.sub("", self.path.stem)


class LabelParseError(ValueError):
    pass


def parse_label_file(path: Path) -> list[YoloBox]:
    """Parse a YOLO detection label file. A missing file means "no objects"."""
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise LabelParseError(f"{path.name}:{lineno}: expected 5 fields, got {len(parts)}")
        try:
            cls = int(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:])
        except ValueError as exc:
            raise LabelParseError(f"{path.name}:{lineno}: {exc}") from None
        boxes.append(YoloBox(cls, xc, yc, w, h))
    return boxes


def format_label_rows(boxes: list[YoloBox]) -> str:
    return "".join(f"{b.cls} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}\n" for b in boxes)


@dataclass
class YoloDataset:
    root: Path
    names: list[str]
    items: list[ImageItem]

    @classmethod
    def load(cls, root: str | Path) -> YoloDataset:
        root = Path(root)
        cfg_path = root / "data.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"no data.yaml in {root}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        raw_names = cfg.get("names")
        if isinstance(raw_names, dict):
            names = [str(raw_names[k]) for k in sorted(raw_names, key=int)]
        elif isinstance(raw_names, list):
            names = [str(n) for n in raw_names]
        else:
            raise ValueError(f"{cfg_path}: 'names' must be a list or mapping")

        items: list[ImageItem] = []
        for split in SPLITS:
            for dirname in _SPLIT_DIR_ALIASES[split]:
                img_dir = root / dirname / "images"
                if img_dir.is_dir():
                    lbl_dir = root / dirname / "labels"
                    items.extend(
                        ImageItem(p, split, lbl_dir / f"{p.stem}.txt")
                        for p in sorted(img_dir.iterdir())
                        if p.suffix.lower() in IMAGE_EXTS
                    )
                    break
        if not items:
            raise ValueError(f"no images found under {root}/{{train,valid,test}}/images")
        return cls(root=root, names=names, items=items)

    def by_split(self, split: str) -> list[ImageItem]:
        return [it for it in self.items if it.split == split]
