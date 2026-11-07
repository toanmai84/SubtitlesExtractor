"""[v3.23.676] Tự dò vùng phụ đề: lấy mẫu trải đều cả phim dài, chỉ giải mã khung khoá khi bước thưa.

Đo thật (CPU, PP-OCRv6 small):
* Tập 3 phim mẫu (152 s): tự dò vùng 335 s > trích xuất 312 s — OCR toàn khung 1,2 s/khung, 274 khung (giữ nguyên:
  lấy mẫu thưa hơn làm vùng hẹp lại — 0,5 s: rộng 603 px; 1,5 s: 548; 3 s: 433 → cắt câu dài).
* Phim ghép cả bộ 139 phút: trước đây lấy mẫu 0,5 s rồi DỪNG ở 3000 khung (chỉ ~25 phút đầu, ~1 giờ OCR). Nay 400
  khung trải đều cả phim: vùng (56, 818, 605, 122) khớp vùng từng tập (57..660).
* Giải mã cả phim 139 phút: mọi khung ~404 s, chỉ khung khoá 58 s.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


class _Sampler:
    def __init__(self) -> None:
        self.configs = []

    def iter_frames(self, metadata, roi=None, config=None):
        self.configs.append(config)
        return iter(())


def _detector(sampler):
    from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import (
        OcrBasedAutoRoiDetector,
    )

    engine = SimpleNamespace(is_initialized=True, initialize=lambda: None)
    return OcrBasedAutoRoiDetector(engine, sampler)


@pytest.mark.parametrize(("duration", "want_step"), [
    (152.0, 0.5),                       # tập ngắn: giữ nguyên bước 0,5 s (274 khung)
    (8351.7, (8351.7 - 120.0) / 400),   # phim 139 phút: 400 khung trải đều (~20,6 s)
])
def test_buoc_lay_mau_trai_deu_ca_phim(duration: float, want_step: float) -> None:
    sampler = _Sampler()
    meta = SimpleNamespace(duration_sec=duration, width=720, height=1280, path=Path("x.mp4"))
    _detector(sampler)._sample_and_ocr(meta, 500, 8)
    assert sampler.configs[0].sample_step_sec == pytest.approx(want_step, rel=1e-3)


def test_buoc_thua_chi_giai_ma_khung_khoa() -> None:
    from subtitles_extractor.infrastructure.video.decoders import pyav_frame_sampler as mod

    assert mod._KEYFRAME_ONLY_MIN_STEP_SEC == 5.0
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert 'skip_frame = "NONKEY"' in src


def test_giai_ma_khung_khoa_tren_video_that(tmp_path: Path) -> None:
    """Video thật nhỏ (30 s, ui_assets): bước 10 s vẫn trả khung, mốc ≥ mốc mục tiêu."""
    asset = Path("/home/user/series/ui_assets/tap01_30s.mp4")
    if not asset.is_file():
        pytest.skip("không có video mẫu")
    from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
    from subtitles_extractor.infrastructure.video.decoders.pyav_frame_sampler import PyAvFrameSampler

    meta = SimpleNamespace(duration_sec=30.0, width=720, height=1280, path=asset)
    config = FrameSamplingConfig(sample_step_sec=10.0, phash_distance_threshold=64,
                                 pixel_diff_threshold=-1.0)
    frames = [f for f in PyAvFrameSampler().iter_frames(meta, roi=None, config=config)
              if f.image_rgb.size]
    assert 1 <= len(frames) <= 4
    assert all(f.image_rgb.shape[:2] == (1280, 720) for f in frames)
