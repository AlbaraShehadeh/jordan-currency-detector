from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jcd.data.dataset import (
    ImageItem,
    LabelParseError,
    YoloBox,
    YoloDataset,
    format_label_rows,
    parse_label_file,
)
from tests.data_factory import RAW_NAMES, Img, build, leaky_dataset


def test_load_splits_and_names(tmp_path: Path) -> None:
    ds = YoloDataset.load(leaky_dataset(tmp_path / "ds"))
    assert ds.names == RAW_NAMES
    assert len(ds.items) == 8
    assert {it.split for it in ds.items} == {"train", "valid", "test"}
    assert len(ds.by_split("train")) == 4


def test_names_as_mapping_and_val_dir_alias(tmp_path: Path) -> None:
    root = build(tmp_path / "ds", [Img("train", "a", 1)])
    (root / "train").rename(root / "val")
    (root / "data.yaml").write_text(yaml.safe_dump({"names": {1: "b", 0: "a"}}), "utf-8")
    ds = YoloDataset.load(root)
    assert ds.names == ["a", "b"]
    assert [it.split for it in ds.items] == ["valid"]


@pytest.mark.parametrize(
    ("setup", "error"),
    [
        ("no_yaml", FileNotFoundError),
        ("bad_names", ValueError),
        ("no_images", ValueError),
    ],
)
def test_load_errors(tmp_path: Path, setup: str, error: type[Exception]) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    if setup != "no_yaml":
        names = "oops" if setup == "bad_names" else ["a"]
        (root / "data.yaml").write_text(yaml.safe_dump({"names": names}), "utf-8")
    with pytest.raises(error):
        YoloDataset.load(root)


def test_source_id_strips_roboflow_suffix() -> None:
    item = ImageItem(Path("x/IMG_1_jpg.rf.0a1b2c.jpg"), "train", Path("x.txt"))
    assert item.source_id == "IMG_1_jpg"
    assert ImageItem(Path("plain.jpg"), "train", Path("p.txt")).source_id == "plain"


def test_parse_and_format_labels_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "l.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n\n2 0.1 0.2 0.3 0.4\n", "utf-8")
    boxes = parse_label_file(p)
    assert boxes == [YoloBox(0, 0.5, 0.5, 0.2, 0.2), YoloBox(2, 0.1, 0.2, 0.3, 0.4)]
    p.write_text(format_label_rows(boxes), "utf-8")
    assert parse_label_file(p) == boxes
    assert parse_label_file(tmp_path / "missing.txt") == []


@pytest.mark.parametrize("line", ["0 0.5 0.5 0.2", "a 0.5 0.5 0.2 0.2", "0 x 0.5 0.2 0.2"])
def test_parse_rejects_malformed_lines(tmp_path: Path, line: str) -> None:
    p = tmp_path / "l.txt"
    p.write_text(line + "\n", "utf-8")
    with pytest.raises(LabelParseError, match=r"l\.txt:1"):
        parse_label_file(p)
