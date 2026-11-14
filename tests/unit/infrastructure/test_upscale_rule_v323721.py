"""[v3.23.721] "Phóng to ảnh" phải phóng tới chiều cao đích khi vùng chữ thấp hơn đích.

Lỗi gốc (cả nhánh CPU ``apply_cpu_preprocessing_pipeline`` lẫn nhánh VRAM ``apply_gpu_preprocessing_pipeline``): chỉ
phóng khi ảnh thấp hơn MỘT NỬA đích. Vùng chữ thật của bộ 86 tập cao 136 px → mọi đích 128–272 px đều KHÔNG đổi kích
thước (bật tuỳ chọn mà không có tác dụng; lượt chạy thật "đích 256" của chủ dự án thực chất là ảnh gốc), còn đích 274 px
nhảy vọt ×2,01. Trang Debug thì phóng theo quy tắc ``h < đích`` → xem trước khác lượt trích xuất thật.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from subtitles_extractor.infrastructure.ocr.preprocessing.image_filters import (
    apply_cpu_preprocessing_pipeline,
    upscale_size,
)

_SRC = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor"


def _pipeline(image: np.ndarray, target: int, border: int = 0) -> np.ndarray:
    return apply_cpu_preprocessing_pipeline(
        image, upscale_small_text=True, upscale_target_height_px=target, apply_clahe_flag=False, clahe_clip_limit=3.0,
        clahe_tile_size=8, apply_sharpen_flag=False, apply_contrast_boost_flag=False, contrast_factor=1.0,
        add_white_border=border > 0, border_thickness_px=border,
    )


def test_roi_of_136_px_is_upscaled_to_the_target() -> None:
    roi = np.zeros((136, 720, 3), dtype=np.uint8)
    assert _pipeline(roi, 272).shape[:2] == (272, 1440)          # mã cũ: (136, 720)
    assert _pipeline(roi, 204).shape[:2] == (204, 1080)          # mã cũ: (136, 720) — ×1,5 trước đây không thể
    assert _pipeline(roi, 256, border=8).shape[:2] == (256 + 16, round(720 * 256 / 136) + 16)


def test_roi_already_tall_enough_is_untouched() -> None:
    roi = np.full((136, 720, 3), 7, dtype=np.uint8)
    for target in (96, 128, 136):
        out = _pipeline(roi, target)
        assert out.shape == roi.shape
        assert np.array_equal(out, roi)


def test_upscale_size_rule() -> None:
    assert upscale_size(136, 720, 272) == (1440, 272)
    assert upscale_size(136, 720, 136) is None
    assert upscale_size(136, 720, 100) is None
    assert upscale_size(0, 720, 272) is None
    assert upscale_size(20, 100, 400) == (400, 80)                 # trần ×4
    assert upscale_size(20, 100, 400, max_factor=3.0) == (300, 60)


def test_gpu_pipeline_uses_the_shared_rule() -> None:
    """Nhánh VRAM không chạy được khi thiếu CUDA → kiểm bằng AST: phải gọi ``upscale_size`` và bỏ cổng "nửa đích"."""
    source = (_SRC / "infrastructure" / "ocr" / "preprocessing" / "gpu_image_filters.py").read_text(encoding="utf-8")
    calls = {getattr(node.func, "id", "") for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)}
    assert "upscale_size" in calls
    assert "target_h // 2" not in source and "upscale_target_height_px // 2" not in source


# ── [v3.23.722] Nhánh NVDEC: tiền xử lý luôn chạy CUDA trên VRAM (zero-copy), phóng bicubic khớp cv2.resize ──────────


def _nvdec_settings(upscale: bool):
    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

    settings = ApplicationSettings()
    settings.hardware.frame_decoder_backend = "pynvvideocodec"
    settings.preprocess.upscale_small_text = upscale
    settings.preprocess.upscale_target_height_px = 272
    settings.preprocess.apply_sharpen = True
    return settings


def _fake_vm(settings):
    from types import SimpleNamespace

    container = SimpleNamespace(settings_service=SimpleNamespace(current=settings))
    vm = SimpleNamespace(_container=container, _video=SimpleNamespace(path=Path("x.mp4")), ocr_language_override=None)
    vm._effective_ocr_language = lambda snapshot: "ch"
    return vm


def _pretend_cupy(monkeypatch) -> None:
    import importlib.util

    real = importlib.util.find_spec
    fake = {"cupy", "PyNvVideoCodec"}            # [v3.23.723] NVDEC cần cả hai (đúng điều kiện của container)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name in fake else real(name, *a))
    from subtitles_extractor.composition import preprocess_routing

    monkeypatch.setattr(preprocess_routing, "find_spec", importlib.util.find_spec)
    monkeypatch.setattr(preprocess_routing, "cupy_usable", lambda: True)     # [v3.23.724] CuPy phải NẠP được


def test_extract_request_keeps_preprocessing_on_vram_with_nvdec(monkeypatch) -> None:
    from subtitles_extractor.domain.value_objects.roi import Roi
    from subtitles_extractor.presentation.view_models.extract_page_view_model import ExtractPageViewModel

    _pretend_cupy(monkeypatch)
    request = ExtractPageViewModel._build_single_request(_fake_vm(_nvdec_settings(True)), Roi(x=0, y=774, width=720, height=136))
    assert request.sampling.vram_upscale_small_text and request.sampling.vram_sharpen          # 3.23.721: đẩy về CPU
    assert request.sampling.vram_upscale_target_height_px == 272
    assert not request.ocr.preprocess.upscale_small_text and not request.ocr.preprocess.apply_sharpen   # không làm 2 lần


def test_reocr_request_keeps_preprocessing_on_vram_with_nvdec(monkeypatch) -> None:
    from subtitles_extractor.presentation.view_models.editor_page_view_model import EditorPageViewModel

    _pretend_cupy(monkeypatch)
    vm = _fake_vm(_nvdec_settings(True))
    request = EditorPageViewModel._build_reocr_request(vm, Path("x.mp4"), [(1.0, 2.0)], ["u1"], None, {}, None, 0.5)
    assert request.sampling.vram_upscale_small_text and not request.ocr.preprocess.upscale_small_text


def _subtitle_like_image() -> np.ndarray:
    rng = np.random.default_rng(3)
    image = rng.integers(0, 60, (136, 720, 3), dtype=np.uint8)
    image[60:120, 200:520] = 245                                      # khối chữ trắng
    image[70:110, 230:250] = 10                                       # viền tối
    return image


def test_vram_resize_matches_opencv() -> None:
    """Cùng mã nhánh VRAM chạy với NumPy phải khớp ``cv2.resize`` tới ±1 mức xám (làm tròn số học điểm cố định)."""
    import cv2

    from subtitles_extractor.infrastructure.ocr.preprocessing.gpu_image_filters import separable_resize

    for image in (_subtitle_like_image(), np.random.default_rng(0).integers(0, 256, (136, 720, 3), dtype=np.uint8)):
        for (width, height), cubic in (((1440, 272), True), ((1080, 204), True), ((2400, 453), False), ((97, 31), True)):
            ref = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR)
            out = separable_resize(image, width, height, cubic=cubic, xp=np)
            assert out.shape == ref.shape and out.dtype == np.uint8
            assert int(np.abs(out.astype(int) - ref).max()) <= 1, (width, height, cubic)


def test_vram_pipeline_upscales_like_the_cpu_pipeline(monkeypatch) -> None:
    """Chạy ĐÚNG ``apply_gpu_preprocessing_pipeline`` với NumPy thay CuPy: ảnh ra trùng nhánh CPU tới ±1."""
    from subtitles_extractor.infrastructure.ocr.preprocessing import gpu_image_filters as gpu

    monkeypatch.setattr(gpu, "HAS_CUPY", True)
    monkeypatch.setattr(gpu, "cp", np, raising=False)
    image = _subtitle_like_image()
    buffers: dict = {}
    for target in (272, 204, 136):
        out = gpu.apply_gpu_preprocessing_pipeline(image, buffers, True, target, False, 1.0, True, 8)
        ref = _pipeline(image, target, border=8)
        assert out.shape == ref.shape, target
        assert int(np.abs(out.astype(int) - ref).max()) <= 1, target
    assert any(key.startswith("resize_tab_") for key in buffers)     # bảng nội suy giữ lại, không dựng mỗi khung
