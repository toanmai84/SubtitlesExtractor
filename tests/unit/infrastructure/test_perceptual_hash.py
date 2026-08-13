"""Test pHash + Hamming + pixel diff utility."""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.video.perceptual_hash import (
    compute_phash,
    hamming_distance,
    pixel_diff_ratio,
)


def _solid_image(color: tuple[int, int, int], size: int = 100) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, :] = color
    return img


class TestComputePhash:
    def test_identical_images_have_same_hash(self) -> None:
        img = _solid_image((128, 128, 128))
        assert compute_phash(img) == compute_phash(img.copy())

    def test_different_images_have_different_hash(self) -> None:
        a = _solid_image((10, 10, 10))
        # Tạo ảnh có pattern khác hẳn để DCT chắc chắn cho hash khác.
        b = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        # Trên ảnh ngẫu nhiên, hash gần như chắc chắn khác nhau.
        assert compute_phash(a) != compute_phash(b)


class TestHammingDistance:
    def test_zero_when_identical(self) -> None:
        assert hamming_distance(0xABCDEF, 0xABCDEF) == 0

    def test_count_matches_xor_popcount(self) -> None:
        assert hamming_distance(0b1010, 0b0101) == 4
        assert hamming_distance(0b1111, 0b1110) == 1


class TestPixelDiffRatio:
    def test_identical_returns_zero(self) -> None:
        img = _solid_image((100, 100, 100))
        assert pixel_diff_ratio(img, img.copy()) == 0.0

    def test_different_size_returns_one(self) -> None:
        a = _solid_image((50, 50, 50), size=100)
        b = _solid_image((50, 50, 50), size=80)
        assert pixel_diff_ratio(a, b) == 1.0

    def test_small_change_low_ratio(self) -> None:
        a = _solid_image((100, 100, 100))
        b = a.copy()
        b[0:5, 0:5, :] = 200  # đổi 25 px / 10000 px = 0.25%
        ratio = pixel_diff_ratio(a, b)
        assert 0.0 < ratio < 0.01
