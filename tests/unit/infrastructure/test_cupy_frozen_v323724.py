"""[v3.23.724] Bản đóng gói 3.23.723: bấm Trích xuất là sập, bản build phình 2,2 GB (log chủ dự án 14/11/2026).

Gốc chung: PyInstaller chép ~2 GB DLL CUDA (phụ thuộc của các .pyd CuPy) vào GỐC gói. CuPy 14 thấy ``nvrtc`` ở
``%TEMP%\\_MEIxxxx`` → suy thư mục CUDA là ``%TEMP%`` → ``os.add_dll_directory("%TEMP%\\bin")`` → ``FileNotFoundError``
ngay trong ``import cupy``; mã chỉ bắt ``ImportError`` nên lỗi lọt tới luồng giao diện.
"""

from __future__ import annotations

import importlib.abc
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _spec_filter(bundle_cuda: bool):
    source = (ROOT / "SubtitlesExtractor.spec").read_text(encoding="utf-8")
    block = source[source.index("_CUDA_DLL_RE = "):source.index("_before = len(a.binaries)")]
    scope = {"_re": re, "_os": os, "_BUNDLE_CUDA": bundle_cuda}
    exec(block, scope)  # noqa: S102 — chạy đúng đoạn lọc trong tệp spec
    return scope["_is_unwanted_cuda_dll"]


def test_spec_drops_cuda_dlls_pulled_in_by_cupy() -> None:
    unwanted = _spec_filter(bundle_cuda=False)
    for name in ("cublas64_12.dll", "cublasLt64_12.dll", "nvrtc64_120_0.dll", "nvrtc-builtins64_129.dll",
                 "cudart64_12.dll", "cufft64_11.dll", "curand64_10.dll", "cusolver64_11.dll", "cusparse64_12.dll",
                 "nvJitLink_120_0.dll", "cudnn_ops64_9.dll", r"nvidia\cublas\bin\cublas64_12.dll"):
        assert unwanted(name), name
    for name in ("python313.dll", r"av\avcodec-61.dll", "libmpv-2.dll", "Qt6Core.dll", r"cupy_backends\cuda\x.pyd",
                 r"PyNvVideoCodec\nvcuvid.dll"):
        assert not unwanted(name), name


def test_bundled_cuda_build_keeps_libraries_in_their_folders_only() -> None:
    unwanted = _spec_filter(bundle_cuda=True)
    assert not unwanted(r"nvidia\cublas\bin\cublas64_12.dll")
    assert unwanted("cublas64_12.dll")                 # bản trùng ở gốc gói vẫn bỏ (nguồn của lỗi %TEMP%\bin)


class _BrokenCupyFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``import cupy`` hành xử như bản đóng gói 3.23.723: thêm thư mục DLL không tồn tại."""

    def find_spec(self, name, path=None, target=None):
        if name == "cupy":
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        os.add_dll_directory(r"C:\Users\x\AppData\Local\Temp\bin")   # ném FileNotFoundError nếu không được che


def test_cupy_loader_survives_the_frozen_dll_directory_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.ocr.preprocessing import gpu_image_filters

    def strict_add_dll_directory(path: str):
        if not os.path.isdir(path):
            raise FileNotFoundError(2, "The system cannot find the file specified", path)
        return None

    monkeypatch.setattr(os, "add_dll_directory", strict_add_dll_directory, raising=False)
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    finder = _BrokenCupyFinder()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    module, usable, reason = gpu_image_filters._load_cupy()
    # Thư mục không tồn tại bị bỏ qua → import xong (mô-đun giả không có ``cuda`` → "không dùng được", không ném).
    assert module is not None and not usable and reason
    assert os.add_dll_directory is strict_add_dll_directory         # đã trả lại hàm gốc


def test_cupy_loader_catches_any_initialisation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.ocr.preprocessing import gpu_image_filters

    class Exploding(_BrokenCupyFinder):
        def exec_module(self, module):
            raise OSError("[WinError 126] cublas64_12.dll")

    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Exploding(), *sys.meta_path])
    module, usable, reason = gpu_image_filters._load_cupy()          # mã cũ: chỉ bắt ImportError → lỗi lọt ra
    assert module is None and not usable and "cublas64_12.dll" in reason


def test_nvdec_is_not_selected_when_cupy_cannot_be_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Có gói CuPy + PyNvVideoCodec nhưng CuPy không nạp được → PyAV + tiền xử lý CPU (mã cũ: vẫn chọn NVDEC rồi sập)."""
    from subtitles_extractor.composition import preprocess_routing
    from subtitles_extractor.infrastructure.ocr.preprocessing import gpu_image_filters
    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

    settings = ApplicationSettings()
    settings.hardware.frame_decoder_backend = "pynvvideocodec"
    monkeypatch.setattr(preprocess_routing, "find_spec", lambda name: object())
    monkeypatch.setattr(gpu_image_filters, "HAS_CUPY", False)
    assert not preprocess_routing.nvdec_sampler_selected(settings)
    monkeypatch.setattr(gpu_image_filters, "HAS_CUPY", True)
    assert preprocess_routing.nvdec_sampler_selected(settings)
