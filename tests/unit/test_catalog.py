from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from jcd.catalog import Catalog, DenominationKind, UnknownDenominationError, normalize_label

# Class names exactly as emitted by the v1 model (Inference/model.pt == train7/best.pt).
V1_MODEL_CLASSES = [
    "1-2-dinar", "1-4-dinar", "1-dinar", "10-Piastres", "10-dinar",
    "20-dinar", "5-Piastres", "5-dinar", "50-dinar",
]  # fmt: skip
# Class names in the original Roboflow project jordancoinvision/coin-vision (U+2044 fraction slash).
COIN_VISION_CLASSES = [
    "10-Piastres", "50-dinar", "1⁄2-dinar", "20-dinar", "1⁄4-dinar",
    "10-dinar", "1-dinar", "5-dinar", "5-Piastres",
]  # fmt: skip


def test_default_catalog_has_all_jod_denominations(catalog: Catalog) -> None:
    assert catalog.currency == "JOD"
    assert catalog.ids == (
        "5-piastres", "10-piastres", "quarter-dinar", "half-dinar", "1-dinar",
        "5-dinar", "10-dinar", "20-dinar", "50-dinar",
    )  # fmt: skip
    assert sum(d.value for d in catalog) == Decimal("86.90")
    assert {d.kind for d in catalog} == {DenominationKind.COIN, DenominationKind.NOTE}


def test_values_are_exact_decimals(catalog: Catalog) -> None:
    assert catalog.value_of("5-piastres") == Decimal("0.05")
    assert catalog.value_of("10-piastres") * 3 == Decimal("0.30")  # would be 0.30000000000000004


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1⁄2-dinar", "1-2-dinar"),
        ("1/4 dinar", "1-4-dinar"),
        ("  5_Piastres ", "5-piastres"),
        ("10--dinar", "10-dinar"),
    ],
)
def test_normalize_label(raw: str, expected: str) -> None:
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("names", [V1_MODEL_CLASSES, COIN_VISION_CLASSES])
def test_known_dataset_class_names_all_resolve(catalog: Catalog, names: list[str]) -> None:
    mapping = catalog.map_model_classes(names)
    assert set(mapping.values()) == set(catalog.ids)


def test_aliases_resolve_to_same_denomination(catalog: Catalog) -> None:
    assert catalog.resolve("1-2-dinar").id == "half-dinar"
    assert catalog.resolve("1⁄2-dinar").id == "half-dinar"
    assert catalog.resolve("HALF-DINAR").id == "half-dinar"
    assert "1-4-dinar" in catalog
    assert 42 not in catalog


def test_unknown_label_raises(catalog: Catalog) -> None:
    with pytest.raises(UnknownDenominationError, match="100-dinar"):
        catalog.resolve("100-dinar")


def test_map_model_classes_reports_all_unknowns(catalog: Catalog) -> None:
    with pytest.raises(UnknownDenominationError) as exc:
        catalog.map_model_classes(["5-dinar", "euro", "dollar"])
    assert exc.value.labels == ("euro", "dollar")
    assert "euro, dollar" in str(exc.value)


def _spec(*denoms: dict[str, object]) -> dict[str, object]:
    return {"currency": "JOD", "denominations": list(denoms)}


def test_ambiguous_alias_rejected() -> None:
    a = {"id": "a", "label": "A", "value": "1", "kind": "coin", "aliases": ["x"]}
    b = {"id": "b", "label": "B", "value": "2", "kind": "coin", "aliases": ["X"]}
    with pytest.raises(ValueError, match="ambiguous"):
        Catalog.from_mapping(_spec(a, b))


def test_duplicate_id_rejected() -> None:
    a = {"id": "a", "label": "A", "value": "1", "kind": "coin"}
    with pytest.raises(ValueError, match="duplicate"):
        Catalog.from_mapping(_spec(a, a))


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "Bad Id", "label": "A", "value": "1", "kind": "coin"},
        {"id": "a", "label": "A", "value": "0", "kind": "coin"},
        {"id": "a", "label": "A", "value": "1", "kind": "token"},
        {"id": "a", "label": "A", "value": "1", "kind": "coin", "extra": True},
    ],
)
def test_invalid_spec_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Catalog.from_mapping(_spec(bad))


def test_float_values_become_exact_decimals() -> None:
    cat = Catalog.from_mapping(_spec({"id": "a", "label": "A", "value": 0.1, "kind": "coin"}))
    assert cat.value_of("a") == Decimal("0.1")


def test_from_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cat.yaml"
    path.write_text(
        "currency: USD\ndenominations:\n"
        "  - {id: quarter, label: Quarter, value: '0.25', kind: coin, aliases: [25c]}\n",
        encoding="utf-8",
    )
    cat = Catalog.from_yaml(path)
    assert cat.currency == "USD"
    assert len(cat) == 1
    assert cat.resolve("25C").id == "quarter"
