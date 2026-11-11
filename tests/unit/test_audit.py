from __future__ import annotations

import json
from pathlib import Path

import pytest

from jcd.catalog import Catalog, UnknownDenominationError
from jcd.data.audit import audit_dataset, leakage_stats, write_report
from jcd.data.dataset import YoloDataset
from tests.data_factory import Img, build, leaky_dataset


def test_detects_both_kinds_of_leakage(tmp_path: Path, catalog: Catalog) -> None:
    report = audit_dataset(YoloDataset.load(leaky_dataset(tmp_path / "ds")), catalog)
    lk = report.leakage
    assert lk.n_groups == 6
    assert lk.groups_spanning_splits == 2
    assert lk.eval_images == 4
    assert lk.eval_images_leaked == 2  # src_a (source id) + renamed_b (mirrored, pHash)
    assert lk.leaked_by_split == {"valid": 1, "test": 1}
    assert lk.leaked_fraction == 0.5


def test_class_counts_use_canonical_ids(tmp_path: Path, catalog: Catalog) -> None:
    report = audit_dataset(YoloDataset.load(leaky_dataset(tmp_path / "ds")), catalog)
    assert report.class_name_mapping["1⁄2-dinar"] == "half-dinar"
    assert report.splits["train"].class_counts == {"10-piastres": 4, "half-dinar": 1, "5-dinar": 1}
    assert report.splits["train"].boxes == 6
    assert report.instances_per_image == {1: 6, 2: 2}
    assert report.image_sizes == {"96x96": 8}
    assert report.n_label_issues == 0


def test_label_issues(tmp_path: Path, catalog: Catalog) -> None:
    root = build(
        tmp_path / "ds",
        [
            Img("train", "oob", 1, labels=["0 0.95 0.5 0.2 0.2"]),
            Img("train", "float_noise", 7, labels=["0 0.90002 0.5 0.2 0.2"]),  # 1.00002: fine
            Img("train", "badcls", 2, labels=["7 0.5 0.5 0.2 0.2"]),
            Img("train", "zero", 3, labels=["0 0.5 0.5 0.0 0.2"]),
            Img("train", "dup", 4, labels=["0 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.2 0.2"]),
            Img("train", "malformed", 5, labels=["0 0.5 0.5"]),
            Img("train", "nolabel", 6, write_label=False),
        ],
    )
    report = audit_dataset(YoloDataset.load(root), catalog)
    issues = {(i.image, i.issue) for i in report.label_issues}
    assert ("oob.jpg", "box extends outside the image") in issues
    assert ("badcls.jpg", "class index 7 out of range") in issues
    assert ("zero.jpg", "zero/negative box size") in issues
    assert ("dup.jpg", "duplicate box") in issues
    assert ("nolabel.jpg", "missing label file") in issues
    assert any(img == "malformed.jpg" and "expected 5 fields" in msg for img, msg in issues)
    assert report.n_label_issues == 6
    assert report.splits["train"].empty_images == 1  # nolabel


def test_unknown_class_names_fail_fast(tmp_path: Path, catalog: Catalog) -> None:
    root = build(tmp_path / "ds", [Img("train", "a", 1)], names=["euro"])
    with pytest.raises(UnknownDenominationError):
        audit_dataset(YoloDataset.load(root), catalog)


def test_leakage_stats_clean_split() -> None:
    lk = leakage_stats(["train", "train", "valid", "test"], [0, 0, 1, 2], "m")
    assert lk.eval_images_leaked == 0
    assert lk.groups_spanning_splits == 0
    assert lk.largest_group == 2
    assert leakage_stats(["train"], [0], "m").leaked_fraction == 0.0


def test_write_report(tmp_path: Path, catalog: Catalog) -> None:
    report = audit_dataset(YoloDataset.load(leaky_dataset(tmp_path / "ds")), catalog)
    js, md = write_report(report, tmp_path / "out")
    assert json.loads(js.read_text("utf-8"))["leakage"]["eval_images_leaked"] == 2
    text = md.read_text("utf-8")
    assert "**2 / 4 (50.0%)**" in text
    assert text.index("| 10-piastres |") < text.index("| half-dinar |") < text.index("| 5-dinar |")
