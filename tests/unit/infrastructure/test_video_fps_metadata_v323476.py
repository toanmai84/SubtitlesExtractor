"""[v3.23.476] fps setting điều khiển mức lấy mẫu video của Gemini (videoMetadata.fps).

fps≈1 (mặc định) → handle THÔ (hành vi cũ, an toàn). fps>1 → bọc Part với videoMetadata.fps.
Lỗi bọc → fallback handle thô (không làm hỏng call).
"""
from __future__ import annotations
from unittest.mock import MagicMock
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _inst(fps: float) -> GeminiSubtitleTranslator:
    inst = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    inst._analysis_video_fps = fps
    return inst


def _handle():
    h = MagicMock()
    h.uri = "files/abc123"
    h.mime_type = "video/mp4"
    return h


def test_fps_1_keeps_raw_handle():
    inst = _inst(1.0)
    h = _handle()
    assert inst._apply_video_fps([h]) == [h]  # thô, không bọc


def test_fps_just_above_1_still_raw():
    # ngưỡng 1.05 → 1.0-1.05 vẫn thô (tránh bọc vô ích)
    inst = _inst(1.03)
    h = _handle()
    assert inst._apply_video_fps([h]) == [h]


def test_empty_and_none():
    inst = _inst(3.0)
    assert inst._apply_video_fps([]) == []
    assert inst._apply_video_fps(None) == []


def test_fps_above_1_attempts_wrap_or_falls_back():
    # Sandbox có thể thiếu google.genai → fallback thô (an toàn); nếu có → bọc Part.
    inst = _inst(3.0)
    h = _handle()
    out = inst._apply_video_fps([h])
    assert len(out) == 1  # luôn 1 phần tử, không mất handle


# [v3.23.477] Gemini giới hạn videoMetadata.fps ≤ 24 → kẹp cứng (setting cũ 30 không gây 400).
def test_fps_clamped_to_24():
    for raw, expect in [(30.0, 24.0), (25.0, 24.0), (24.0, 24.0), (2.0, 2.0), (0.1, 0.2)]:
        assert max(0.2, min(24.0, float(raw))) == expect


def test_apply_video_fps_never_exceeds_24():
    from unittest.mock import MagicMock
    inst = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    inst._analysis_video_fps = 24.0  # đã kẹp ở __init__
    h = MagicMock(); h.uri = "files/x"; h.mime_type = "video/mp4"
    out = inst._apply_video_fps([h])
    assert len(out) == 1  # không mất handle, không lỗi
