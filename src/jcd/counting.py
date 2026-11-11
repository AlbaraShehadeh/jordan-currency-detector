"""Counting and valuation for single images and tracked video.

Fixes over the v1 notebooks:

* Images: the item count and the total value are computed from the **same** confidence-filtered
  set (v1 counted all boxes but valued only the confident ones).
* Video: a track's denomination is a confidence-weighted **vote over its whole lifetime**
  (v1 used whatever class the first frame said), and a track only counts once it has been seen
  in `min_hits` frames, which suppresses one-frame flicker / ID-switch ghosts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any

from jcd.catalog import Catalog
from jcd.types import Detection


@dataclass(frozen=True)
class CountResult:
    """Per-denomination counts (catalog order, non-zero only) and the exact total value."""

    currency: str
    counts: Mapping[str, int]
    total: Decimal

    @classmethod
    def from_counts(cls, counts: Mapping[str, int], catalog: Catalog) -> CountResult:
        ordered = {d.id: counts[d.id] for d in catalog if counts.get(d.id, 0) > 0}
        unknown = set(counts) - set(catalog.ids)
        if unknown:
            raise ValueError(f"counts reference ids not in catalog: {sorted(unknown)}")
        total = sum((catalog.resolve(i).value * n for i, n in ordered.items()), Decimal(0))
        return cls(currency=catalog.currency, counts=MappingProxyType(ordered), total=total)

    @property
    def n_items(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe representation; the total is a string to stay exact."""
        return {
            "currency": self.currency,
            "n_items": self.n_items,
            "counts": dict(self.counts),
            "total": str(self.total),
        }


def select_detections(
    detections: Iterable[Detection], *, min_confidence: float = 0.0
) -> list[Detection]:
    """Detections at or above the confidence threshold. Use this for rendering too, so what is
    drawn is exactly what is counted."""
    return [d for d in detections if d.confidence >= min_confidence]


def count_detections(
    detections: Iterable[Detection], catalog: Catalog, *, min_confidence: float = 0.0
) -> CountResult:
    """Count and value the items in a single image."""
    kept = select_detections(detections, min_confidence=min_confidence)
    return CountResult.from_counts(Counter(catalog.resolve(d.label).id for d in kept), catalog)


@dataclass
class _TrackState:
    first_frame: int
    last_frame: int
    hits: int = 0
    votes: dict[str, float] = field(default_factory=dict)


class TrackRegistry:
    """Accumulates tracker output across frames and yields de-duplicated counts.

    Call `update()` once per frame with that frame's tracked detections (frames with no
    detections should still call `update([])` so frame numbering stays correct).
    """

    def __init__(self, catalog: Catalog, *, min_hits: int = 3, min_confidence: float = 0.0) -> None:
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        self._catalog = catalog
        self._order = {denom_id: i for i, denom_id in enumerate(catalog.ids)}
        self.min_hits = min_hits
        self.min_confidence = min_confidence
        self._tracks: dict[int, _TrackState] = {}
        self._frame = -1

    @property
    def frames_seen(self) -> int:
        return self._frame + 1

    def update(self, detections: Iterable[Detection]) -> None:
        self._frame += 1
        hit_this_frame: set[int] = set()
        for det in detections:
            if det.track_id is None or det.confidence < self.min_confidence:
                continue
            denom_id = self._catalog.resolve(det.label).id
            state = self._tracks.get(det.track_id)
            if state is None:
                state = self._tracks[det.track_id] = _TrackState(self._frame, self._frame)
            state.votes[denom_id] = state.votes.get(denom_id, 0.0) + det.confidence
            state.last_frame = self._frame
            if det.track_id not in hit_this_frame:  # a duplicated id in one frame is one hit
                state.hits += 1
                hit_this_frame.add(det.track_id)

    def label_of(self, track_id: int) -> str | None:
        """Current majority-vote denomination id of a track (ties -> catalog order)."""
        state = self._tracks.get(track_id)
        if state is None or not state.votes:
            return None
        return max(state.votes.items(), key=lambda kv: (kv[1], -self._order[kv[0]]))[0]

    def is_confirmed(self, track_id: int) -> bool:
        state = self._tracks.get(track_id)
        return state is not None and state.hits >= self.min_hits

    def confirmed(self) -> dict[int, str]:
        """track_id -> denomination id, for tracks seen in at least `min_hits` frames."""
        out: dict[int, str] = {}
        for tid in self._tracks:
            label = self.label_of(tid)
            if label is not None and self.is_confirmed(tid):
                out[tid] = label
        return out

    def result(self) -> CountResult:
        return CountResult.from_counts(Counter(self.confirmed().values()), self._catalog)

    def reset(self) -> None:
        self._tracks.clear()
        self._frame = -1
