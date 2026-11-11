from __future__ import annotations

from decimal import Decimal

import pytest

from jcd.catalog import Catalog, UnknownDenominationError
from jcd.counting import CountResult, count_detections, select_detections
from jcd.types import Detection
from tests.conftest import MakeDet


def test_counts_and_total(catalog: Catalog, det: MakeDet) -> None:
    dets = [det("5-dinar"), det("5-dinar"), det("1-2-dinar"), det("5-Piastres"), det("50-dinar")]
    result = count_detections(dets, catalog)
    assert dict(result.counts) == {"5-piastres": 1, "half-dinar": 1, "5-dinar": 2, "50-dinar": 1}
    assert result.n_items == 5
    assert result.total == Decimal("60.55")


def test_count_and_value_use_same_filtered_set(catalog: Catalog, det: MakeDet) -> None:
    """Regression (v1 bug): v1 counted *all* boxes but valued only those >= threshold."""
    dets = [det("20-dinar", 0.95), det("10-dinar", 0.40), det("1-dinar", 0.71)]
    result = count_detections(dets, catalog, min_confidence=0.7)
    assert result.n_items == 2
    assert result.total == Decimal("21.00")
    assert len(select_detections(dets, min_confidence=0.7)) == result.n_items


def test_threshold_is_inclusive(catalog: Catalog, det: MakeDet) -> None:
    assert count_detections([det("1-dinar", 0.7)], catalog, min_confidence=0.7).n_items == 1


def test_empty_image(catalog: Catalog) -> None:
    result = count_detections([], catalog)
    assert result.n_items == 0
    assert result.total == Decimal(0)
    assert result.as_dict() == {"currency": "JOD", "n_items": 0, "counts": {}, "total": "0"}


def test_counts_follow_catalog_order(catalog: Catalog, det: MakeDet) -> None:
    result = count_detections([det("50-dinar"), det("5-Piastres"), det("1-dinar")], catalog)
    assert list(result.counts) == ["5-piastres", "1-dinar", "50-dinar"]


def test_as_dict_is_json_safe_and_exact(catalog: Catalog, det: MakeDet) -> None:
    result = count_detections([det("10-Piastres")] * 3, catalog)
    assert result.as_dict() == {
        "currency": "JOD",
        "n_items": 3,
        "counts": {"10-piastres": 3},
        "total": "0.30",
    }


def test_unknown_label_is_an_error_not_a_silent_zero(catalog: Catalog, det: MakeDet) -> None:
    with pytest.raises(UnknownDenominationError):
        count_detections([det("euro")], catalog)


def test_counts_are_read_only(catalog: Catalog, det: MakeDet) -> None:
    result = count_detections([det("1-dinar")], catalog)
    with pytest.raises(TypeError):
        result.counts["1-dinar"] = 99  # type: ignore[index]


def test_from_counts_rejects_unknown_ids(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="not in catalog"):
        CountResult.from_counts({"euro": 1}, catalog)


@pytest.mark.parametrize("conf", [-0.1, 1.1, float("nan")])
def test_detection_rejects_bad_confidence(conf: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Detection(box=(0, 0, 1, 1), label="1-dinar", confidence=conf)


def test_detection_rejects_inverted_box() -> None:
    with pytest.raises(ValueError, match="box"):
        Detection(box=(10, 0, 1, 1), label="1-dinar", confidence=0.5)


def test_detection_center() -> None:
    assert Detection(box=(0, 0, 10, 20), label="1-dinar", confidence=0.5).center == (5.0, 10.0)
