"""[v3.23.622] Nén video ngữ cảnh DỌC không còn thu chiều cao về 360 (720x1280 → 202x360).

Thăm dò thật trên tập mẫu 720x1280: bản nén 540x960 đọc đúng từng dòng phụ đề cứng, bản
202x360 trả dòng bị gộp/thiếu (kể cả ở HIGH). Token không phụ thuộc số điểm ảnh.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    PORTRAIT_MIN_SHORT_SIDE,
    _vf_cpu_scale,
    _vf_gpu_full,
    _vf_gpu_upload,
    target_frame_size,
)


@pytest.mark.parametrize(("w", "h", "setting", "expected"), [
    (720, 1280, 360, (540, 960)),     # dọc: cạnh ngắn ≥ 540
    (1080, 1920, 360, (540, 960)),
    (480, 854, 360, (480, 854)),      # không phóng to quá nguồn
    (1920, 1080, 360, (640, 360)),    # ngang: hành vi cũ
    (1280, 720, 360, (640, 360)),
    (1920, 1080, 720, (1280, 720)),   # ngang + cài đặt cao
    (720, 1280, 720, (720, 1280)),    # dọc + cài đặt cao hơn sàn
    (640, 360, 720, (640, 360)),
    (0, 0, 360, (-2, 360)),           # không đọc được → hành vi cũ
])
def test_target_frame_size(w: int, h: int, setting: int, expected: tuple[int, int]) -> None:
    assert target_frame_size(w, h, setting) == expected


def test_sizes_are_always_even() -> None:
    for w, h in ((721, 1281), (719, 1279), (1001, 563)):
        out_w, out_h = target_frame_size(w, h, 360)
        assert out_w % 2 == 0 and out_h % 2 == 0


def test_filters_carry_both_dimensions() -> None:
    assert "scale=540:960" in _vf_cpu_scale(960, "1", 540)
    assert "w=540:h=960" in _vf_gpu_full(960, "1", 540)
    assert "w=540:h=960" in _vf_gpu_upload(960, "1", 540)
    # Mặc định giữ nguyên hành vi cũ cho người gọi cũ.
    assert "scale=-2:360" in _vf_cpu_scale(360, "1")
    assert PORTRAIT_MIN_SHORT_SIDE == 540


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")
def test_real_ffmpeg_makes_540x960_from_720x1280(tmp_path: Path) -> None:
    source = tmp_path / "p.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=720x1280:d=2:r=5",
         "-pix_fmt", "yuv420p", str(source)], check=True, timeout=60)
    width, height = target_frame_size(720, 1280, 360)
    out = tmp_path / "o.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-vf", _vf_cpu_scale(height, "1", width),
         str(out)], check=True, timeout=60)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    assert probe == "540,960"
