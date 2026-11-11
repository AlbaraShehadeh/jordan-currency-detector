from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest
import yaml

from jcd.catalog import Catalog
from jcd.data.audit import audit_dataset
from jcd.data.dataset import YoloDataset, parse_label_file
from jcd.data.grouping import group_dataset
from jcd.data.split import MANIFEST_NAME, grouped_split, materialize_split
from tests.data_factory import leaky_dataset

RATIOS = {"train": 0.7, "valid": 0.15, "test": 0.15}


def _synthetic(n_groups: int = 200) -> tuple[list[int], list[Counter[str]]]:
    group_of: list[int] = []
    counts: list[Counter[str]] = []
    for g in range(n_groups):
        for _ in range(1 + g % 3):  # groups of 1-3 images
            group_of.append(g)
            counts.append(Counter({f"c{g % 5}": 1 + g % 2}))
    return group_of, counts


def test_groups_never_cross_splits() -> None:
    group_of, counts = _synthetic()
    assign = grouped_split(group_of, counts, RATIOS, seed=0)
    by_group: dict[int, set[str]] = {}
    for g, s in zip(group_of, assign, strict=True):
        by_group.setdefault(g, set()).add(s)
    assert all(len(s) == 1 for s in by_group.values())


def test_ratios_and_class_balance_are_close() -> None:
    group_of, counts = _synthetic()
    assign = grouped_split(group_of, counts, RATIOS, seed=0)
    n = Counter(assign)
    for split, r in RATIOS.items():
        assert abs(n[split] / len(assign) - r) < 0.03
    for cls in {k for c in counts for k in c}:
        per = Counter()
        for c, s in zip(counts, assign, strict=True):
            per[s] += c[cls]
        total = sum(per.values())
        assert abs(per["train"] / total - 0.7) < 0.08


def test_deterministic_per_seed() -> None:
    group_of, counts = _synthetic()
    assert grouped_split(group_of, counts, RATIOS, seed=1) == grouped_split(
        group_of, counts, RATIOS, seed=1
    )


def test_zero_ratio_split_gets_nothing_and_empty_groups_ok() -> None:
    assign = grouped_split([0, 1, 2, 3], [Counter()] * 4, {"train": 0.5, "valid": 0.5, "test": 0})
    assert set(assign) <= {"train", "valid"}


@pytest.mark.parametrize("ratios", [{}, {"train": -1.0}, {"train": 0.0}])
def test_invalid_ratios(ratios: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="invalid ratios"):
        grouped_split([0], [Counter()], ratios)


def test_materialize_canonical_leak_free_dataset(tmp_path: Path, catalog: Catalog) -> None:
    ds = YoloDataset.load(leaky_dataset(tmp_path / "raw"))
    grouping = group_dataset(ds)
    assign = grouped_split(grouping.group_of, [Counter()] * len(ds.items), RATIOS, seed=0)
    data_yaml = materialize_split(ds, assign, grouping.group_of, tmp_path / "out", catalog)

    cfg = yaml.safe_load(data_yaml.read_text("utf-8"))
    assert cfg["names"] == dict(enumerate(catalog.ids))
    out = YoloDataset.load(tmp_path / "out")
    assert len(out.items) == len(ds.items)
    assert audit_dataset(out, catalog).leakage.eval_images_leaked == 0

    # raw class 1 ("1⁄2-dinar") must be remapped to the catalog index of half-dinar
    src_a_copy = next(it for it in out.items if it.path.stem == "src_a_jpg.rf.bbb222")
    assert parse_label_file(src_a_copy.label_path)[0].cls == catalog.ids.index("half-dinar")

    with (tmp_path / "out" / MANIFEST_NAME).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(ds.items)
    assert {r["split"] for r in rows} <= set(RATIOS)
    assert all(len(r["sha256"]) == 64 for r in rows)


def test_materialize_refuses_to_overwrite(tmp_path: Path, catalog: Catalog) -> None:
    ds = YoloDataset.load(leaky_dataset(tmp_path / "raw"))
    groups = list(range(len(ds.items)))
    assign = ["train"] * len(ds.items)
    materialize_split(ds, assign, groups, tmp_path / "out", catalog)
    with pytest.raises(FileExistsError):
        materialize_split(ds, assign, groups, tmp_path / "out", catalog)
    materialize_split(ds, assign, groups, tmp_path / "out", catalog, overwrite=True)
    assert not (tmp_path / "out" / "valid").exists()
