"""[v3.23.460] Nhớ NVENC treo giữa các phiên (marker) — bỏ GPU không phải chờ 90s mỗi lần."""
from __future__ import annotations
import os
import time
from pathlib import Path
import pytest
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
)


def _inst(tmp_path: Path) -> GeminiVideoContextProvider:
    inst = GeminiVideoContextProvider.__new__(GeminiVideoContextProvider)
    inst._cache_db_path = tmp_path / "gemini_video_uploads.db"
    return inst


def test_marker_absent_by_default(tmp_path):
    inst = _inst(tmp_path)
    assert inst._nvenc_marker_path_exists() is False


def test_persist_then_detected(tmp_path):
    inst = _inst(tmp_path)
    inst._persist_nvenc_broken()
    assert inst._nvenc_marker_path_exists() is True
    assert inst._nvenc_marker_path().name == "nvenc_runtime_broken.marker"


def test_marker_expires_after_7_days(tmp_path):
    inst = _inst(tmp_path)
    inst._persist_nvenc_broken()
    marker = inst._nvenc_marker_path()
    old = time.time() - 8 * 86400
    os.utime(marker, (old, old))
    assert inst._nvenc_marker_path_exists() is False   # quá hạn → bỏ
    assert not marker.exists()                          # + xoá để re-probe


def test_no_cache_db_no_marker(tmp_path):
    inst = GeminiVideoContextProvider.__new__(GeminiVideoContextProvider)
    inst._cache_db_path = None
    assert inst._nvenc_marker_path() is None
    assert inst._nvenc_marker_path_exists() is False
    inst._persist_nvenc_broken()  # không crash khi không có nơi ghi


# [v3.23.471] Theo dõi hwaccel giải mã hỏng RIÊNG từng loại (d3d11va, cuda) + persist.
def test_hwdec_broken_per_type(tmp_path):
    inst = _inst(tmp_path)
    assert inst._load_broken_hwdecs() == set()
    inst._hwdec_broken = set()
    inst._persist_broken_hwdec("cuda")
    # phiên mới đọc lại marker
    inst2 = _inst(tmp_path)
    loaded = inst2._load_broken_hwdecs()
    assert "cuda" in loaded and "d3d11va" not in loaded
    inst2._hwdec_broken = loaded
    inst2._persist_broken_hwdec("d3d11va")
    assert inst._load_broken_hwdecs() == {"cuda", "d3d11va"}


def test_hwaccel_of():
    from subtitles_extractor.infrastructure.translation.gemini_video_context import (
        GeminiVideoContextProvider as P,
    )
    assert P._hwaccel_of(["ffmpeg", "-hwaccel", "d3d11va", "-i", "x"]) == "d3d11va"
    assert P._hwaccel_of(["ffmpeg", "-i", "x"]) == ""
