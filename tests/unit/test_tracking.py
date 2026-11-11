from __future__ import annotations

from decimal import Decimal

import pytest

from jcd.catalog import Catalog
from jcd.counting import TrackRegistry
from tests.conftest import MakeDet


def test_each_track_counted_once(catalog: Catalog, det: MakeDet) -> None:
    reg = TrackRegistry(catalog, min_hits=1)
    for _ in range(10):
        reg.update([det("5-dinar", track_id=1), det("1-dinar", track_id=2)])
    result = reg.result()
    assert dict(result.counts) == {"1-dinar": 1, "5-dinar": 1}
    assert result.total == Decimal("6.00")
    assert reg.frames_seen == 10


def test_label_is_majority_vote_not_first_frame(catalog: Catalog, det: MakeDet) -> None:
    """Regression (v1 bug): v1 locked a track's class to whatever its first frame said."""
    reg = TrackRegistry(catalog, min_hits=1)
    reg.update([det("10-dinar", 0.6, track_id=7)])  # misclassified on entry
    for _ in range(5):
        reg.update([det("20-dinar", 0.9, track_id=7)])
    assert reg.label_of(7) == "20-dinar"
    assert reg.result().total == Decimal("20.00")


def test_votes_are_confidence_weighted(catalog: Catalog, det: MakeDet) -> None:
    reg = TrackRegistry(catalog, min_hits=1)
    reg.update([det("5-dinar", 0.3, track_id=1)])
    reg.update([det("5-dinar", 0.3, track_id=1)])
    reg.update([det("50-dinar", 0.95, track_id=1)])  # 0.95 > 0.6
    assert reg.label_of(1) == "50-dinar"


def test_vote_ties_break_by_catalog_order(catalog: Catalog, det: MakeDet) -> None:
    reg = TrackRegistry(catalog, min_hits=1)
    reg.update([det("50-dinar", 0.5, track_id=1)])
    reg.update([det("1-dinar", 0.5, track_id=1)])
    assert reg.label_of(1) == "1-dinar"


def test_min_hits_suppresses_flicker(catalog: Catalog, det: MakeDet) -> None:
    reg = TrackRegistry(catalog, min_hits=3)
    reg.update([det("1-dinar", track_id=1), det("50-dinar", track_id=99)])  # 99: one-frame ghost
    reg.update([det("1-dinar", track_id=1)])
    assert reg.result().n_items == 0  # nothing confirmed yet
    reg.update([det("1-dinar", track_id=1)])
    assert reg.confirmed() == {1: "1-dinar"}
    assert not reg.is_confirmed(99)
    assert reg.result().total == Decimal("1.00")


def test_duplicate_id_in_one_frame_is_one_hit(catalog: Catalog, det: MakeDet) -> None:
    reg = TrackRegistry(catalog, min_hits=2)
    reg.update([det("1-dinar", track_id=1), det("1-dinar", track_id=1)])
    assert not reg.is_confirmed(1)


def test_untracked_and_low_confidence_are_ignored(catalog: Catalog, det: MakeDet) -> None:
    reg = TrackRegistry(catalog, min_hits=1, min_confidence=0.5)
    reg.update([det("5-dinar"), det("10-dinar", 0.2, track_id=3)])
    assert reg.result().n_items == 0
    assert reg.label_of(3) is None


def test_empty_frames_advance_frame_counter(catalog: Catalog) -> None:
    reg = TrackRegistry(catalog)
    reg.update([])
    reg.update([])
    assert reg.frames_seen == 2


def test_reset(catalog: Catalog, det: MakeDet) -> None:
    reg = TrackRegistry(catalog, min_hits=1)
    reg.update([det("1-dinar", track_id=1)])
    reg.reset()
    assert reg.frames_seen == 0
    assert reg.result().n_items == 0


def test_min_hits_must_be_positive(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="min_hits"):
        TrackRegistry(catalog, min_hits=0)
