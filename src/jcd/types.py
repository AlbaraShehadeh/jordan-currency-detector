"""Backend-agnostic data types shared by detectors, counting, rendering and the API."""

from __future__ import annotations

import math
from dataclasses import dataclass

Box = tuple[float, float, float, float]
"""Axis-aligned box in absolute pixel coordinates: (x1, y1, x2, y2)."""


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected item.

    `label` is the raw class name emitted by the model; the catalog resolves it to a
    denomination. `track_id` is set only by a tracker (video), never for single images.
    """

    box: Box
    label: str
    confidence: float
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0) or math.isnan(self.confidence):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        x1, y1, x2, y2 = self.box
        if x2 < x1 or y2 < y1:
            raise ValueError(f"box must be (x1, y1, x2, y2) with x2>=x1 and y2>=y1, got {self.box}")

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
