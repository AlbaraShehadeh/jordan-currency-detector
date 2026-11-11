"""Perceptual hashing and near-duplicate grouping (numpy + Pillow only).

pHash: grayscale -> 32x32 -> 2D DCT -> 8x8 low-frequency block -> bits above the median.
Each image also gets hashes of its flipped variants, so a horizontally/vertically flipped
augmentation of a photo is still recognized as the same photo.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

HASH_BITS = 64
_IMG = 32
_LOW = 8

HashArray = npt.NDArray[np.uint64]


@lru_cache(maxsize=1)
def _dct_matrix(n: int = _IMG) -> npt.NDArray[np.float64]:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m: npt.NDArray[np.float64] = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0, :] = np.sqrt(1 / n)
    return m


def _bits_to_int(bits: npt.NDArray[np.bool_]) -> int:
    return int.from_bytes(np.packbits(bits.astype(np.uint8)).tobytes(), "big")


def phash(img: Image.Image) -> int:
    """64-bit perceptual hash of an image."""
    gray = np.asarray(
        img.convert("L").resize((_IMG, _IMG), Image.Resampling.LANCZOS), dtype=np.float64
    )
    d = _dct_matrix()
    low = (d @ gray @ d.T)[:_LOW, :_LOW]
    return _bits_to_int((low > np.median(low)).ravel())


def phash_variants(img: Image.Image) -> tuple[int, int, int, int]:
    """Hashes of (original, h-flip, v-flip, both flips)."""
    h = ImageOps.mirror(img)
    return (phash(img), phash(h), phash(ImageOps.flip(img)), phash(ImageOps.flip(h)))


def hash_files(paths: Iterable[Path]) -> HashArray:
    """(N, 4) uint64 array of flip-variant pHashes."""
    rows = []
    for p in paths:
        with Image.open(p) as im:
            rows.append(phash_variants(ImageOps.exif_transpose(im)))
    return np.array(rows, dtype=np.uint64).reshape(-1, 4)


def flip_invariant_distances(hashes: HashArray, i: int) -> npt.NDArray[np.int64]:
    """Min Hamming distance between image i and every image, allowing a flip of either one.

    pHash(flip(x)) is not a bit-permutation of pHash(x), so comparing i's original to j's
    variants alone is asymmetric; taking the min of both directions makes d(i, j) == d(j, i)
    and the grouping independent of image order.
    """
    i_vs_jvariants = np.bitwise_count(np.bitwise_xor(hashes, hashes[i, 0])).min(axis=1)
    ivariants_vs_j = np.bitwise_count(np.bitwise_xor(hashes[:, :1], hashes[i])).min(axis=1)
    dist: npt.NDArray[np.int64] = np.minimum(i_vs_jvariants, ivariants_vs_j).astype(np.int64)
    return dist


def near_duplicate_pairs(hashes: HashArray, threshold: int) -> list[tuple[int, int]]:
    """All pairs (i < j) whose flip-invariant pHash distance is <= threshold. O(N^2), vectorized
    per row; ~0.5 s for 2.5k images."""
    pairs: list[tuple[int, int]] = []
    for i in range(len(hashes) - 1):
        dist = flip_invariant_distances(hashes[i:], 0)[1:]
        pairs.extend((i, i + 1 + int(j)) for j in np.flatnonzero(dist <= threshold))
    return pairs


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> list[int]:
        """Dense group id per element (0..k-1), in first-seen order."""
        dense: dict[int, int] = {}
        return [dense.setdefault(self.find(i), len(dense)) for i in range(len(self.parent))]


def group_items(
    n: int, pairs: Iterable[tuple[int, int]], keys: Sequence[str] | None = None
) -> list[int]:
    """Connected components over near-duplicate pairs, plus items sharing the same key
    (e.g. Roboflow source id)."""
    uf = UnionFind(n)
    for a, b in pairs:
        uf.union(a, b)
    if keys is not None:
        first: dict[str, int] = {}
        for i, k in enumerate(keys):
            uf.union(i, first.setdefault(k, i))
    return uf.groups()
