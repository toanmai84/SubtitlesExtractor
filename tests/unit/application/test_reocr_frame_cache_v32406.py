"""[v3.23.106] Test RAM frame cache cho Re-OCR.

Bảo đảm: (1) khoá phụ thuộc mọi tham số ảnh hưởng pixel; (2) KHÔNG phục vụ tập frame bị
cắt (vượt cap -> bỏ cache, lần sau giải mã đầy đủ) — sửa lỗi "OCR lại thiếu frame".
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesRequest,
)
from subtitles_extractor.application.use_cases import extract_subtitles as es_mod
from subtitles_extractor.application.use_cases.extract_subtitles import (
    ExtractSubtitlesUseCase,
    _ReOcrFrameCacheManager,
    _sampling_signature,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
    SampledFrame,
)
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence


def _frame(idx: int) -> SampledFrame:
    return SampledFrame(
        frame_index=idx, timestamp_sec=idx * 0.1,
        image_rgb=np.zeros((50, 200, 3), dtype=np.uint8),
    )


# ---------- _sampling_signature ----------

def test_signature_changes_with_pixel_affecting_params() -> None:
    base = FrameSamplingConfig()
    assert _sampling_signature(base) == _sampling_signature(FrameSamplingConfig())
    # Tiền xử lý VRAM đổi -> chữ ký phải đổi (frame khác pixel)
    assert _sampling_signature(base) != _sampling_signature(
        FrameSamplingConfig(vram_sharpen=not base.vram_sharpen))
    assert _sampling_signature(base) != _sampling_signature(
        FrameSamplingConfig(vram_contrast_factor=base.vram_contrast_factor + 0.5))
    assert _sampling_signature(base) != _sampling_signature(
        FrameSamplingConfig(sample_step_sec=base.sample_step_sec + 0.1))
    assert _sampling_signature(base) != _sampling_signature(
        FrameSamplingConfig(skip_intro_sec=5.0))


# ---------- _ReOcrFrameCacheManager ----------

def test_cache_manager_put_get_roundtrip() -> None:
    mgr = _ReOcrFrameCacheManager()
    mgr.put("k1", [_frame(0), _frame(1)])
    got = mgr.get("k1")
    assert got is not None and len(got) == 2


def test_cache_manager_key_mismatch_returns_none() -> None:
    mgr = _ReOcrFrameCacheManager()
    mgr.put("k1", [_frame(0)])
    assert mgr.get("k2") is None


def test_cache_manager_rejects_oversized_set() -> None:
    mgr = _ReOcrFrameCacheManager()
    mgr._max_cached_frames = 2
    mgr.put("k", [_frame(i) for i in range(3)])  # 3 > 2 -> không cache
    assert mgr.get("k") is None


# ---------- Tích hợp: chống phục vụ tập frame bị cắt ----------

def _make_use_case(frames: list[SampledFrame], tmp_path: Path) -> tuple:
    meta = VideoMetadata(
        path=tmp_path / "v.mp4", width=200, height=50,
        fps=25.0, total_frames=1000, duration_sec=40.0,
    )
    meta.path.touch()
    reader = MagicMock()
    reader.read.return_value = meta

    sampler = MagicMock()

    def _iter(*_a, **_k) -> Iterator[SampledFrame]:
        yield from frames
    sampler.iter_frames.side_effect = _iter

    ocr = MagicMock()
    ocr.is_initialized = True
    ocr.infer_batch.side_effect = lambda images_rgb, frame_indices, timestamps_sec: [
        OcrFrameResult(
            frame_index=i, timestamp_sec=t,
            text_boxes=[OcrTextBox(text="x", confidence=Confidence(0.9),
                                   polygon=[(0, 10), (100, 10), (100, 40), (0, 40)])],
        )
        for i, t in zip(frame_indices, timestamps_sec, strict=True)
    ]

    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        SubtitleBuilderConfig,
    )
    from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder

    uc = ExtractSubtitlesUseCase(
        metadata_reader=reader, frame_sampler=sampler, ocr_engine=ocr,
        builder=SubtitleBuilder(SubtitleBuilderConfig(min_duration_sec=0.0)),
        exporters={},
    )
    return uc, sampler, meta


def _reocr_request(meta: VideoMetadata) -> ExtractSubtitlesRequest:
    # reocr task: skip_export=True và raw_ocr_output_path=None
    return ExtractSubtitlesRequest(
        video_path=meta.path, output_path=meta.path.with_suffix(".tmp"),
        sampling=FrameSamplingConfig(), ocr=OcrEngineConfig(batch_size=8),
        skip_export=True, raw_ocr_output_path=None,
    )


@pytest.fixture(autouse=True)
def _clear_global_cache() -> Iterator[None]:
    es_mod._GLOBAL_FRAME_CACHE.clear()
    original_cap = es_mod._GLOBAL_FRAME_CACHE._max_cached_frames
    yield
    es_mod._GLOBAL_FRAME_CACHE._max_cached_frames = original_cap
    es_mod._GLOBAL_FRAME_CACHE.clear()


def test_oversized_video_is_not_cached_truncated(tmp_path: Path) -> None:
    # Video vượt cap (3 frame > cap 2) -> KHÔNG cache -> lần 2 phải lấy mẫu LẠI đầy đủ.
    es_mod._GLOBAL_FRAME_CACHE._max_cached_frames = 2
    frames = [_frame(i) for i in range(3)]
    uc, sampler, meta = _make_use_case(frames, tmp_path)
    req = _reocr_request(meta)

    uc.execute(req)
    assert es_mod._GLOBAL_FRAME_CACHE.get(_key(meta)) is None  # không cache

    uc.execute(req)
    # Lấy mẫu lại lần 2 (cache miss) -> iter_frames gọi 2 lần, đủ 3 frame mỗi lần
    assert sampler.iter_frames.call_count == 2


def test_complete_set_is_cached_and_reused(tmp_path: Path) -> None:
    # Video <= cap -> cache trọn bộ -> lần 2 dùng cache (không lấy mẫu lại).
    es_mod._GLOBAL_FRAME_CACHE._max_cached_frames = 10
    frames = [_frame(i) for i in range(3)]
    uc, sampler, meta = _make_use_case(frames, tmp_path)
    req = _reocr_request(meta)

    uc.execute(req)
    assert es_mod._GLOBAL_FRAME_CACHE.get(_key(meta)) is not None  # đã cache trọn bộ

    uc.execute(req)
    assert sampler.iter_frames.call_count == 1  # lần 2 dùng cache


def _key(meta: VideoMetadata) -> str:
    return f"{meta.path.name}|{_sampling_signature(FrameSamplingConfig())}|none"
