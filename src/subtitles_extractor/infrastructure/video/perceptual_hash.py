"""Tính :wikipedia:`Perceptual hash` cho ảnh và so khớp Hamming distance.

CẢI TIẾN:
    * pHash: Bỏ qua điểm DC Component để lấy median chuẩn xác, tăng sức mạnh nhận diện.
    * pHash: Thay thế vòng lặp for chậm chạp bằng np.packbits siêu tốc ở C-level.
    * Pixel Diff: Chuyển Gray trước khi diff để tránh lỗi mất mát (suppression) ở kênh màu xanh (Blue/Red).
"""

from __future__ import annotations

import cv2
import numpy as np

_HASH_SIZE: int = 8
_DCT_SIZE: int = 32


def compute_phash(image_rgb: np.ndarray) -> int:
    """Tính pHash 64 bit cho ảnh RGB."""
    if image_rgb.ndim != 3:
        raise ValueError(
            f"Cần ảnh RGB 3 chiều, nhận được shape={image_rgb.shape}."
        )

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(
        gray, (_DCT_SIZE, _DCT_SIZE), interpolation=cv2.INTER_LINEAR
    )

    dct = cv2.dct(np.float32(resized))
    dct_low_freq = dct[:_HASH_SIZE, :_HASH_SIZE]

    # [CRITICAL FIX]: Bỏ qua DC Component (vị trí 0, 0) khi tính Median
    # dct_low_freq.flatten()[1:] lấy 63 giá trị AC còn lại
    median = np.median(dct_low_freq.flatten()[1:])

    bits = (dct_low_freq > median).flatten()

    # [PERFORMANCE FIX]: Vector hóa việc pack bit thay vì dùng vòng lặp Python
    # np.packbits nén 64 bit bool thành 8 bytes. int.from_bytes đọc nó thành số int 64-bit.
    return int.from_bytes(np.packbits(bits).tobytes(), byteorder='big')


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Đếm số bit khác giữa 2 hash siêu tốc bằng bit_count nội tại."""
    return (hash_a ^ hash_b).bit_count()


def pixel_diff_ratio(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Tỉ lệ pixel khác biệt (0.0–1.0) dùng OpenCV tối ưu bộ nhớ.

    Thuật toán: cvtColor (Gray) -> absdiff -> threshold -> countNonZero.
    """
    if image_a.shape != image_b.shape:
        return 1.0

    # [LOGIC FIX]: Chuyển Grayscale TRƯỚC khi tính diff để các trọng số màu
    # (đặc biệt là kênh B = 0.114) không bóp nghẹt giá trị khác biệt cục bộ.
    if image_a.ndim == 3:
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
    else:
        gray_a, gray_b = image_a, image_b

    # Lấy giá trị sai khác tuyệt đối trên không gian độ chói (Luminance)
    diff = cv2.absdiff(gray_a, gray_b)

    # Xử lý Threshold và Count dưới tầng C++ của OpenCV
    _, thresh = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY)
    significant_pixels = cv2.countNonZero(thresh)

    total_pixels = int(gray_a.size)
    return significant_pixels / total_pixels if total_pixels else 0.0


__all__ = ["compute_phash", "hamming_distance", "pixel_diff_ratio"]
