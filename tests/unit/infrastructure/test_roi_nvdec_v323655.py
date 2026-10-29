"""[v3.23.655] Bug #16: ROI vượt mép khung + upscale → toạ độ ô chữ lệch; NVDEC lỗi gán sai mốc giờ khung."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from subtitles_extractor.application.use_cases.extract_subtitles import ExtractSubtitlesUseCase, effective_roi
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.video.decoders.pynvvideocodec_frame_sampler import (
    error_frame_time,
    nvdec_frame_time,
)


def test_effective_roi_clips_only_when_overflowing() -> None:
    inside = Roi(x=57, y=819, width=603, height=119)
    assert effective_roi(inside, 720, 1280) is inside
    over = Roi(x=57, y=1200, width=603, height=200)          # tràn đáy khung 1280
    assert effective_roi(over, 720, 1280).height == 80
    assert effective_roi(over, 0, 0) is over                    # không biết khung → giữ nguyên


def _box(points):
    from subtitles_extractor.domain.entities.ocr_frame_result import OcrTextBox  # noqa: PLC0415
    from subtitles_extractor.domain.value_objects.confidence import Confidence  # noqa: PLC0415

    return OcrTextBox(text="字", confidence=Confidence(0.9), polygon=points)


def test_reverse_geometry_uses_clipped_height() -> None:
    # ROI khai báo cao 200 nhưng chỉ 80 px nằm trong khung; ảnh cắt 80 px được upscale ×2 → 160 px.
    request = SimpleNamespace(
        roi=Roi(x=0, y=1200, width=720, height=200),
        sampling=SimpleNamespace(vram_upscale_small_text=True, vram_add_border=False, vram_border_thickness_px=0),
    )
    metadata = SimpleNamespace(width=720, height=1280)
    frame = SimpleNamespace(image_rgb=np.zeros((160, 1440, 3), dtype=np.uint8))
    result = OcrFrameResult(frame_index=0, timestamp_sec=0.0, text_boxes=[_box([(100, 40), (300, 40), (300, 120), (100, 120)])])
    out = ExtractSubtitlesUseCase._reverse_vram_geometry(None, [result], [frame], request, metadata)
    assert out[0].text_boxes[0].polygon[0] == (50, 20)          # ÷2 (bản cũ ÷0,8 → (125, 50))


def test_nvdec_pts_zero_is_real_time() -> None:
    assert nvdec_frame_time(0, 1 / 1000, 0.08, stream_has_pts=True) == 0.0
    assert nvdec_frame_time(0, 1 / 1000, 0.08, stream_has_pts=False) == 0.08
    assert nvdec_frame_time(None, 1 / 1000, 0.5, stream_has_pts=True) == 0.5
    assert nvdec_frame_time(2000, 1 / 1000, 0.5, stream_has_pts=True) == 2.0


def test_error_frame_time_never_goes_back() -> None:
    assert error_frame_time(-1.0, 3.0, 0.0) is None              # gói không có PTS → không chen khung rỗng
    assert error_frame_time(2.5, 3.0, 0.0) == 3.0                # thứ tự giải mã lùi → giữ mốc đã phát
    assert error_frame_time(4.0, 3.0, 0.0) == 4.0
    assert error_frame_time(1.0, 0.0, 2.0) is None               # trước điểm bắt đầu


def test_codec_map_cached_and_frame_time_not_v323655() -> None:
    from subtitles_extractor.infrastructure.video.decoders import pynvvideocodec_frame_sampler as module

    assert hasattr(module._get_nv_codec_map, "cache_info")
    assert not hasattr(module.nvdec_frame_time, "cache_info")


def test_extract_page_rebuild_clips_roi_v323655() -> None:
    """[v3.23.655] Rà soát: "Dựng lại từ cache" phải kẹp ROI như lúc trích xuất."""
    from subtitles_extractor.presentation.view_models.extract_page_view_model import ExtractPageViewModel

    meta = SimpleNamespace(width=720, height=1280)
    fake = SimpleNamespace(_container=SimpleNamespace(
        make_load_video_metadata_use_case=lambda: SimpleNamespace(execute=lambda _p: meta),
    ))
    request = SimpleNamespace(roi=Roi(x=0, y=1200, width=720, height=200), video_path="v.mp4")
    assert ExtractPageViewModel._clipped_roi(fake, request).height == 80
    assert ExtractPageViewModel._clipped_roi(fake, SimpleNamespace(roi=None, video_path="v")) is None


def test_debug_page_rebuild_clips_old_cache_roi_v323656(tmp_path) -> None:
    """[v3.23.656] B7: cache OCR thô tạo trước 655 có thể lưu ROI vượt mép — trang Gỡ lỗi dựng lại phải kẹp như trích xuất."""
    from subtitles_extractor.presentation.view_models.debug_page_view_model import DebugPageViewModel

    (tmp_path / "ep.mp4").write_bytes(b"x")
    fake = SimpleNamespace(
        _base_img_dir=tmp_path / "frames", _meta=SimpleNamespace(video_name="ep.mp4"),
        _container=SimpleNamespace(metadata_reader=SimpleNamespace(read=lambda _p: SimpleNamespace(width=720, height=1280))),
    )
    roi = Roi(x=0, y=1200, width=720, height=200)
    assert DebugPageViewModel._clip_roi_to_video(fake, roi).height == 80
    fake._meta = SimpleNamespace(video_name="missing.mp4")
    assert DebugPageViewModel._clip_roi_to_video(fake, roi) is roi
