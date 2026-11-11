"""Currency catalog: the single source of truth for denominations and their values.

Model class names are mapped to denominations **by name** (with aliases), never by class index,
so reordering a model's classes can never silently change a computed total.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CATALOG_RESOURCE = "currency_jod.yaml"

_SEPARATORS = re.compile(r"[⁄∕/_\s]+")  # fraction slash, division slash, '/', '_', ws
_DASHES = re.compile(r"-{2,}")


class UnknownDenominationError(KeyError):
    """Raised when a label cannot be resolved to a denomination in the catalog."""

    def __init__(self, labels: Iterable[str], currency: str) -> None:
        self.labels = tuple(labels)
        super().__init__(f"unknown {currency} denomination label(s): {', '.join(self.labels)}")

    def __str__(self) -> str:  # KeyError.__str__ would repr() the message
        return str(self.args[0])


class DenominationKind(StrEnum):
    COIN = "coin"
    NOTE = "note"


def normalize_label(label: str) -> str:
    """Normalize a raw class label for alias matching.

    Case-folds and treats '/', '⁄' (fraction slash), '_' and whitespace as '-'.

    >>> normalize_label("1⁄2-dinar")
    '1-2-dinar'
    >>> normalize_label(" 5 Piastres ")
    '5-piastres'
    """
    s = _SEPARATORS.sub("-", label.strip().casefold())
    return _DASHES.sub("-", s).strip("-")


class Denomination(BaseModel):
    """One denomination (a coin or a banknote) and its face value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    label: str = Field(min_length=1)
    value: Decimal = Field(gt=0)
    kind: DenominationKind
    aliases: tuple[str, ...] = ()

    @field_validator("value", mode="before")
    @classmethod
    def _exact_decimal(cls, v: Any) -> Any:
        # Floats like 0.1 are inexact; go through str() so 0.1 -> Decimal("0.1").
        return Decimal(str(v)) if isinstance(v, float) else v


class CatalogSpec(BaseModel):
    """Validated on-disk catalog format."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    currency: str = Field(pattern=r"^[A-Z]{3}$")
    denominations: tuple[Denomination, ...] = Field(min_length=1)


class Catalog:
    """Lookup over a currency's denominations, in catalog order."""

    def __init__(self, spec: CatalogSpec) -> None:
        self.currency = spec.currency
        self._by_id: dict[str, Denomination] = {}
        self._index: dict[str, str] = {}
        for denom in spec.denominations:
            if denom.id in self._by_id:
                raise ValueError(f"duplicate denomination id {denom.id!r}")
            self._by_id[denom.id] = denom
            for key in (denom.id, *denom.aliases):
                norm = normalize_label(key)
                owner = self._index.setdefault(norm, denom.id)
                if owner != denom.id:
                    raise ValueError(f"label {key!r} is ambiguous: {owner!r} vs {denom.id!r}")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Catalog:
        return cls(CatalogSpec.model_validate(data))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Catalog:
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh))

    @classmethod
    def default(cls) -> Catalog:
        """The bundled Jordanian Dinar catalog."""
        return _load_default()

    # ------------------------------------------------------------------ queries
    def __iter__(self) -> Iterator[Denomination]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, label: object) -> bool:
        return isinstance(label, str) and normalize_label(label) in self._index

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def resolve(self, label: str) -> Denomination:
        """Map a raw label (id or alias, any case/separator style) to its denomination."""
        try:
            return self._by_id[self._index[normalize_label(label)]]
        except KeyError:
            raise UnknownDenominationError([label], self.currency) from None

    def value_of(self, label: str) -> Decimal:
        return self.resolve(label).value

    def map_model_classes(self, names: Iterable[str]) -> dict[str, str]:
        """Resolve every class name a model can emit; fail fast listing *all* unknown names.

        Call this when loading a model so a class/catalog mismatch is a startup error,
        not a wrong total at request time.
        """
        mapping: dict[str, str] = {}
        unknown: list[str] = []
        for name in names:
            if name in self:
                mapping[name] = self.resolve(name).id
            else:
                unknown.append(name)
        if unknown:
            raise UnknownDenominationError(unknown, self.currency)
        return mapping


@lru_cache(maxsize=1)
def _load_default() -> Catalog:
    text = resources.files("jcd.resources").joinpath(DEFAULT_CATALOG_RESOURCE).read_text("utf-8")
    return Catalog.from_mapping(yaml.safe_load(text))
