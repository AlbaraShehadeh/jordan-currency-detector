from __future__ import annotations

from collections.abc import Callable

import pytest

from jcd.catalog import Catalog
from jcd.types import Detection

MakeDet = Callable[..., Detection]


@pytest.fixture
def catalog() -> Catalog:
    return Catalog.default()


@pytest.fixture
def det() -> MakeDet:
    """Factory: det("5-dinar", 0.9, track_id=3)."""

    def _make(
        label: str,
        confidence: float = 0.9,
        *,
        track_id: int | None = None,
        box: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
    ) -> Detection:
        return Detection(box=box, label=label, confidence=confidence, track_id=track_id)

    return _make
