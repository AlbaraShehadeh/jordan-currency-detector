from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import ImageOps

from jcd.data.hashing import (
    UnionFind,
    flip_invariant_distances,
    group_items,
    hash_files,
    near_duplicate_pairs,
    phash,
    phash_variants,
)
from tests.data_factory import scene


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def test_phash_is_deterministic_and_64_bit() -> None:
    h = phash(scene(1))
    assert h == phash(scene(1))
    assert 0 <= h < 2**64


def test_distinct_scenes_are_far_apart() -> None:
    hashes = [phash(scene(s)) for s in range(8)]
    dists = [_hamming(a, b) for i, a in enumerate(hashes) for b in hashes[i + 1 :]]
    assert min(dists) > 12


def test_jpeg_recompression_is_near_zero(tmp_path: Path) -> None:
    p = tmp_path / "x.jpg"
    scene(3).save(p, quality=60)
    assert hash_files([p])[0, 0] != 0
    assert _hamming(int(hash_files([p])[0, 0]), phash(scene(3))) <= 4


def test_variants_cover_flips() -> None:
    img = scene(4)
    orig, hflip, vflip, both = phash_variants(img)
    assert orig == phash(img)
    assert hflip == phash(ImageOps.mirror(img))
    assert vflip == phash(ImageOps.flip(img))
    assert both == phash(ImageOps.flip(ImageOps.mirror(img)))


def _arr(*imgs: object) -> np.ndarray:  # type: ignore[type-arg]
    return np.array([phash_variants(i) for i in imgs], dtype=np.uint64)  # type: ignore[arg-type]


def test_mirrored_copy_is_flip_invariant_duplicate() -> None:
    h = _arr(scene(5), ImageOps.mirror(scene(5)), scene(6))
    d = flip_invariant_distances(h, 0)
    assert d[0] == 0
    assert d[1] == 0
    assert d[2] > 12


def test_distance_is_symmetric() -> None:
    """Regression: comparing i's original to j's variants only was order-dependent."""
    imgs = [scene(s) for s in range(6)] + [ImageOps.mirror(scene(2)), ImageOps.flip(scene(3))]
    h = _arr(*imgs)
    full = np.stack([flip_invariant_distances(h, i) for i in range(len(h))])
    assert (full == full.T).all()


def test_near_duplicate_pairs() -> None:
    h = _arr(scene(7), scene(8), ImageOps.mirror(scene(7)))
    assert near_duplicate_pairs(h, threshold=6) == [(0, 2)]
    assert near_duplicate_pairs(h[:1], threshold=6) == []


def test_union_find_and_grouping_with_keys() -> None:
    uf = UnionFind(4)
    uf.union(3, 1)
    uf.union(1, 3)  # idempotent
    assert uf.groups() == [0, 1, 2, 1]
    # pairs join 0-1, keys join 2-4 ("x"), 3 alone
    assert group_items(5, [(0, 1)], keys=["a", "b", "x", "c", "x"]) == [0, 0, 1, 2, 1]
    assert group_items(3, []) == [0, 1, 2]
