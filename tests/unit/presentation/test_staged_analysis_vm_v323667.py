"""[v3.23.667] Trang Dịch × phân tích tách bước: video chỉ chuẩn bị khi dùng cho bước người nói."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("cues,expect_video", [(False, False), (True, True)])
def test_khong_bat_visual_cues_thi_khong_tai_video(app, monkeypatch, cues, expect_video) -> None:
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.presentation.workers import analyze_context_worker as mod
    from tests.unit.presentation.test_translate_page_bugs_v323609 import _page

    page, container = _page()
    vm = page._view_model
    vm._source_events = [SubtitleEvent(index=1, text="你好", interval=TimeInterval(0.0, 1.0))]
    seen: dict[str, Any] = {}

    class _Sig:
        def connect(self, *_a: Any) -> None:
            pass

    class _Worker:
        """Worker giả: đủ tín hiệu để nối, không chạy luồng, luôn "đã xong"."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.update(kwargs)

        def __getattr__(self, name: str) -> Any:
            return _Sig()

        def start(self) -> None:
            pass

        def isRunning(self) -> bool:  # noqa: N802 — cùng tên QThread
            return False

    monkeypatch.setattr(mod, "AnalyzeContextWorker", _Worker)
    monkeypatch.setattr(container, "make_video_context_provider", lambda *_a: object())
    assert vm.start_context_analysis(api_key="K", target_lang="Vietnamese", model_name="m",
                                     video_path="/phim/01.mp4", enable_visual_cues=cues)
    assert (seen["video_path"] is not None) is expect_video
    assert (seen["video_provider"] is not None) is expect_video
    # Phân tích vẫn lưu theo video của phim.
    assert vm._pending_analysis_video_path == "/phim/01.mp4"
    vm._analyze_worker = None
    vm._set_busy(False)
    page.close()
    page.deleteLater()
