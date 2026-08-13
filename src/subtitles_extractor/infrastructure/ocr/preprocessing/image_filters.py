"""Hàm tiện ích tiền xử lý ảnh cho PaddleOCR (v3.27 — Đại Thống Nhất)."""

from __future__ import annotations
import threading
from typing import Any
import cv2
import numpy as np

_DEFAULT_BORDER_PX: int = 8
_DEFAULT_MIN_HEIGHT_PX: int = 64
_DEFAULT_UPSCALE_TARGET_HEIGHT_PX: int = 96
_DEFAULT_CLAHE_CLIP_LIMIT: float = 3.0
_DEFAULT_CLAHE_TILE_SIZE: int = 8
_MAX_UPSCALE_FACTOR: float = 4.0
_UPSCALE_INTERPOLATION_BREAKPOINT: float = 2.0
_MIN_BINARIZATION_SIDE_PX: int = 20
_CLAHE_MAX_CACHE_PER_THREAD: int = 4

_clahe_thread_local: threading.local = threading.local()

def _get_clahe_object(clip_limit: float, tile_size: int) -> Any:
    cache: dict[tuple[float, int], Any] = getattr(_clahe_thread_local, "cache", None)
    if cache is None:
        cache = {}
        _clahe_thread_local.cache = cache
    key = (float(clip_limit), int(tile_size))
    instance = cache.get(key)
    if instance is None:
        if len(cache) >= _CLAHE_MAX_CACHE_PER_THREAD:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)
        instance = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_size), int(tile_size)))
        cache[key] = instance
    return instance

def add_border(image_rgb: np.ndarray, border: int = _DEFAULT_BORDER_PX) -> np.ndarray:
    if border <= 0 or image_rgb.ndim < 2: return image_rgb
    return cv2.copyMakeBorder(image_rgb, top=border, bottom=border, left=border, right=border, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

def needs_upscale(image_rgb: np.ndarray, min_height: int = _DEFAULT_MIN_HEIGHT_PX) -> bool:
    if image_rgb.ndim < 2: return False
    return image_rgb.shape[0] < min_height

def upscale_to_min_height(image_rgb: np.ndarray, target_height: int = _DEFAULT_UPSCALE_TARGET_HEIGHT_PX) -> np.ndarray:
    if image_rgb.ndim < 2: return image_rgb
    height, width = image_rgb.shape[:2]
    if height >= target_height or height <= 0 or width <= 0: return image_rgb
    scale = min(_MAX_UPSCALE_FACTOR, target_height / height)
    new_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_LINEAR if scale > _UPSCALE_INTERPOLATION_BREAKPOINT else cv2.INTER_CUBIC
    return cv2.resize(image_rgb, (new_width, int(round(height * scale))), interpolation=interpolation)

def apply_sharpen(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim < 2: return image_rgb
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image_rgb, 1.5, blurred, -0.5, 0)

def apply_contrast_boost(image_rgb: np.ndarray, factor: float = 1.20) -> np.ndarray:
    if abs(factor - 1.0) < 1e-4 or image_rgb.ndim < 2: return image_rgb
    return cv2.convertScaleAbs(image_rgb, alpha=factor, beta=0)

def apply_clahe(image_rgb: np.ndarray, clip_limit: float = _DEFAULT_CLAHE_CLIP_LIMIT, tile_grid_size: int = _DEFAULT_CLAHE_TILE_SIZE) -> np.ndarray:
    if image_rgb.ndim < 3 or image_rgb.shape[2] < 3: return image_rgb
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = _get_clahe_object(float(clip_limit), int(tile_grid_size))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

def apply_adaptive_binarization(image_rgb: np.ndarray, *, block_size: int = 11, c_value: int = 2) -> np.ndarray:
    if image_rgb.ndim < 2: return image_rgb
    if min(image_rgb.shape[:2]) < _MIN_BINARIZATION_SIDE_PX: return image_rgb
    block_size = max(block_size, 3)
    if block_size % 2 == 0: block_size += 1
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    binarized = cv2.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=block_size, C=c_value)
    return cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)

def median_blend_frames(frames_rgb: list[np.ndarray]) -> np.ndarray:
    if not frames_rgb: raise ValueError("Cần ít nhất 1 frame để blend.")
    if len(frames_rgb) == 1: return frames_rgb[0].copy()
    target_shape = frames_rgb[0].shape
    valid_frames = [f for f in frames_rgb if f.shape == target_shape]
    if len(valid_frames) < 2: return frames_rgb[0].copy()
    stacked = np.stack(valid_frames, axis=0)
    return np.median(stacked, axis=0).astype(np.uint8)

def temporal_denoise_bilateral(image_rgb: np.ndarray, diameter: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    if image_rgb.ndim < 2: return image_rgb
    return cv2.bilateralFilter(image_rgb, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def apply_cpu_preprocessing_pipeline(
    image_rgb: np.ndarray, *, upscale_small_text: bool, upscale_target_height_px: int,
    apply_clahe_flag: bool, clahe_clip_limit: float, clahe_tile_size: int,
    apply_sharpen_flag: bool, apply_contrast_boost_flag: bool, contrast_factor: float,
    add_white_border: bool, border_thickness_px: int
) -> np.ndarray:
    """[V3.27] Hợp nhất toàn bộ Pipeline Toán học (Single Source of Truth)."""
    working_rgb = image_rgb.copy()
    if upscale_small_text and needs_upscale(working_rgb, max(upscale_target_height_px // 2, 32)):
        working_rgb = upscale_to_min_height(working_rgb, upscale_target_height_px)
    if apply_clahe_flag:
        working_rgb = apply_clahe(working_rgb, clahe_clip_limit, clahe_tile_size)
    if apply_sharpen_flag:
        working_rgb = apply_sharpen(working_rgb)
    if apply_contrast_boost_flag:
        working_rgb = apply_contrast_boost(working_rgb, contrast_factor)
    if add_white_border and border_thickness_px > 0:
        working_rgb = add_border(working_rgb, border_thickness_px)
    return working_rgb

__all__ = [
    "add_border", "apply_adaptive_binarization", "apply_clahe", "apply_contrast_boost",
    "apply_sharpen", "median_blend_frames", "needs_upscale", "temporal_denoise_bilateral",
    "upscale_to_min_height", "apply_cpu_preprocessing_pipeline"
]
