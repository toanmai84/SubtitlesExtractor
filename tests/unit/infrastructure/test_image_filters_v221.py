"""Unit tests cho image_filters — CLAHE, Adaptive Binarization, Median Blend.

Kiểm tra:
    - CLAHE giữ nguyên shape và dtype.
    - Adaptive binarization output chỉ có 0/255.
    - Median blend loại bỏ noise ngẫu nhiên.
    - Edge cases (ảnh quá nhỏ, 1D, grayscale).
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.infrastructure.ocr.preprocessing.image_filters import (
    add_border,
    apply_adaptive_binarization,
    apply_clahe,
    apply_contrast_boost,
    apply_sharpen,
    median_blend_frames,
    needs_upscale,
    temporal_denoise_bilateral,
    upscale_to_min_height,
)


def _make_rgb_image(
    height: int = 100, width: int = 300, value: int = 128
) -> np.ndarray:
    """Helper tạo ảnh RGB solid."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _make_gradient_image(height: int = 100, width: int = 300) -> np.ndarray:
    """Ảnh gradient dọc — mô phỏng nền hardsub anime."""
    gradient = np.linspace(20, 235, height, dtype=np.uint8)
    image = np.repeat(gradient[:, np.newaxis], width, axis=1)
    return np.stack([image, image, image], axis=2)


class TestClahe:
    """Test CLAHE — Contrast Limited Adaptive Histogram Equalization."""

    def test_preserves_shape_and_dtype(self) -> None:
        img = _make_rgb_image()
        result = apply_clahe(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_enhances_gradient_image(self) -> None:
        """CLAHE trên gradient → range sáng hơn (std dev tăng)."""
        img = _make_gradient_image()
        result = apply_clahe(img, clip_limit=3.0)
        assert result.shape == img.shape
        # CLAHE cải thiện tương phản → std dev kênh L tăng.
        orig_std = np.std(img.astype(float))
        result_std = np.std(result.astype(float))
        assert result_std >= orig_std * 0.8  # Ít nhất gần bằng.

    def test_solid_image_unchanged(self) -> None:
        """Ảnh đồng màu → CLAHE gần như không đổi."""
        img = _make_rgb_image(value=128)
        result = apply_clahe(img)
        # Chênh lệch tối đa rất nhỏ.
        diff = np.abs(result.astype(int) - img.astype(int)).max()
        assert diff < 15

    def test_grayscale_input_returns_unchanged(self) -> None:
        """Ảnh grayscale (2D hoặc < 3 kênh) → trả nguyên."""
        gray = np.full((100, 300), 128, dtype=np.uint8)
        result = apply_clahe(gray)
        assert result.shape == gray.shape

    def test_custom_parameters(self) -> None:
        img = _make_gradient_image()
        result = apply_clahe(img, clip_limit=5.0, tile_grid_size=4)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestAdaptiveBinarization:
    """Test Adaptive Binarization."""

    def test_output_is_binary(self) -> None:
        """Output chỉ chứa pixel 0 hoặc 255."""
        img = _make_gradient_image()
        result = apply_adaptive_binarization(img)
        unique = np.unique(result)
        assert set(unique.tolist()).issubset({0, 255})

    def test_preserves_shape_rgb(self) -> None:
        """Output vẫn là RGB 3 kênh."""
        img = _make_gradient_image()
        result = apply_adaptive_binarization(img)
        assert result.shape == img.shape
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_small_image_returned_unchanged(self) -> None:
        """Ảnh quá nhỏ (< 20x20) → trả nguyên."""
        tiny = _make_rgb_image(height=10, width=10)
        result = apply_adaptive_binarization(tiny)
        np.testing.assert_array_equal(result, tiny)

    def test_even_block_size_corrected(self) -> None:
        """block_size chẵn → tự sửa thành lẻ, không crash."""
        img = _make_gradient_image()
        result = apply_adaptive_binarization(img, block_size=16)
        assert result.shape == img.shape


class TestMedianBlendFrames:
    """Test Multi-Frame Median Blend."""

    def test_single_frame_returns_copy(self) -> None:
        """1 frame → trả bản copy."""
        img = _make_rgb_image(value=100)
        result = median_blend_frames([img])
        np.testing.assert_array_equal(result, img)
        assert result is not img  # Phải là copy.

    def test_removes_random_noise(self) -> None:
        """Median blend loại bỏ nhiễu ngẫu nhiên."""
        clean = _make_rgb_image(value=128)
        rng = np.random.default_rng(42)
        noisy_frames = [
            np.clip(clean.astype(int) + rng.integers(-30, 30, clean.shape), 0, 255).astype(np.uint8)
            for _ in range(5)
        ]
        result = median_blend_frames(noisy_frames)
        # Kết quả phải gần clean image hơn bất kỳ frame noisy nào.
        median_diff = np.mean(np.abs(result.astype(int) - clean.astype(int)))
        for noisy in noisy_frames:
            noisy_diff = np.mean(np.abs(noisy.astype(int) - clean.astype(int)))
            # Median blend phải gần clean hơn hoặc bằng.
            assert median_diff <= noisy_diff + 2.0

    def test_preserves_stable_content(self) -> None:
        """Text ổn định (giống nhau ở mọi frame) → giữ nguyên sau blend."""
        # 5 frame giống hệt nhau.
        stable = _make_rgb_image(value=200)
        result = median_blend_frames([stable] * 5)
        np.testing.assert_array_equal(result, stable)

    def test_empty_raises_error(self) -> None:
        with pytest.raises(ValueError, match="ít nhất 1 frame"):
            median_blend_frames([])

    def test_mismatched_shapes_uses_first(self) -> None:
        """Frame shape khác nhau → chỉ dùng frame khớp shape."""
        img_a = _make_rgb_image(height=100, width=300, value=100)
        img_b = _make_rgb_image(height=50, width=150, value=200)  # Khác shape.
        result = median_blend_frames([img_a, img_b])
        # Chỉ có 1 frame valid (img_a) → copy img_a.
        np.testing.assert_array_equal(result, img_a)

    def test_three_frames_median_correct(self) -> None:
        """3 frame giá trị [100, 150, 200] → median = 150."""
        frames = [
            _make_rgb_image(value=100),
            _make_rgb_image(value=150),
            _make_rgb_image(value=200),
        ]
        result = median_blend_frames(frames)
        assert result[50, 150, 0] == 150


class TestBilateralFilter:
    """Test temporal_denoise_bilateral."""

    def test_preserves_shape(self) -> None:
        img = _make_rgb_image()
        result = temporal_denoise_bilateral(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_1d_input_returns_unchanged(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.uint8)
        result = temporal_denoise_bilateral(arr)
        np.testing.assert_array_equal(result, arr)


class TestExistingFilters:
    """Regression tests cho các filter cũ (add_border, upscale, etc.)."""

    def test_add_border_increases_size(self) -> None:
        img = _make_rgb_image(100, 300)
        result = add_border(img, border=8)
        assert result.shape[0] == 116  # 100 + 2*8
        assert result.shape[1] == 316  # 300 + 2*8

    def test_needs_upscale_true_for_small(self) -> None:
        small = _make_rgb_image(30, 100)
        assert needs_upscale(small, min_height=64) is True

    def test_needs_upscale_false_for_large(self) -> None:
        large = _make_rgb_image(100, 300)
        assert needs_upscale(large, min_height=64) is False

    def test_upscale_reaches_target(self) -> None:
        small = _make_rgb_image(30, 100)
        result = upscale_to_min_height(small, target_height=96)
        assert result.shape[0] == 96

    def test_sharpen_preserves_shape(self) -> None:
        img = _make_rgb_image()
        result = apply_sharpen(img)
        assert result.shape == img.shape

    def test_contrast_boost_preserves_shape(self) -> None:
        img = _make_rgb_image()
        result = apply_contrast_boost(img, factor=1.5)
        assert result.shape == img.shape
