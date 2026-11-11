"""Assign every image to a *source group*: images that must never be split across train/eval.

Two complementary signals (calibrated on the v1 dataset, see docs/SYSTEM_DESIGN.md §4.3):

* **Roboflow source id**: all exported/augmented copies of one upload share it. Catches
  augmented copies that pHash cannot (rotation + crop + hue push them to a median of 18 bits).
* **Flip-invariant pHash <= threshold**: catches the same photo re-uploaded under another name
  (incl. mirrored). At <= 6 bits, sampled matches were all true duplicates; at 8-10 bits most
  matches were different scenes.
"""

from __future__ import annotations

from dataclasses import dataclass

from jcd.data.dataset import YoloDataset
from jcd.data.hashing import HashArray, group_items, hash_files, near_duplicate_pairs

DEFAULT_PHASH_THRESHOLD = 6


@dataclass(frozen=True)
class Grouping:
    group_of: list[int]  # per dataset item
    hashes: HashArray
    n_phash_pairs: int
    threshold: int

    @property
    def n_groups(self) -> int:
        return len(set(self.group_of))


def group_dataset(ds: YoloDataset, *, threshold: int = DEFAULT_PHASH_THRESHOLD) -> Grouping:
    hashes = hash_files(it.path for it in ds.items)
    pairs = near_duplicate_pairs(hashes, threshold)
    group_of = group_items(len(ds.items), pairs, keys=[it.source_id for it in ds.items])
    return Grouping(group_of=group_of, hashes=hashes, n_phash_pairs=len(pairs), threshold=threshold)
