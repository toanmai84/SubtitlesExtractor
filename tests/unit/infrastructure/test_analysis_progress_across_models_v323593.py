"""[v3.23.593] Hồi quy: "đổi model phân tích — các đoạn đã xong được giữ" mà mất hết.

Khi Gemini quá tải, ứng dụng báo:

    Mới phân tích 4/17 đoạn video — còn thiếu đoạn 5, 6, 7, 8… Gemini đang quá tải (503)
    nên đã dừng để giữ hạn mức ngày. Bấm Phân tích lại sau ít phút, hoặc đổi model phân
    tích — các đoạn đã xong được giữ, chỉ gửi đoạn còn thiếu.

nhưng khoá tiến độ gồm cả tên model, nên đổi model là phân tích lại từ đoạn 1. Nhật ký
thật v3.23.592 (phim "灭门惨案II", 17 đoạn, bốn ngày Gemini quá tải): người dùng làm đúng
lời khuyên, đổi qua lại bốn model —

    gemini-3.8-flash  đoạn 1-10   (xong qua nhiều lần chạy)
    gemini-3.6-flash  đoạn 1-6    ← làm lại từ đầu
    gemini-3.7-flash  đoạn 1-4    ← làm lại từ đầu
    gemini-3.5-flash  đoạn 1-17   ← làm lại từ đầu, lần này xong

— 37 request thành công cho một bản phân tích cần 17.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationLine,
)
from subtitles_extractor.infrastructure.translation.analysis_checkpoint import (
    AnalysisCheckpointStore,
    describe_segment_models,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    QuotaExhaustedError,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter

STRONG = "gemini-3.8-flash"
FALLBACK = "gemini-3.5-flash"
REAL_503 = "Gọi Gemini thất bại: 503 UNAVAILABLE"


class _Ref:
    def __init__(self, idx: int) -> None:
        self.chunk_index = idx
        self.remote_name = f"files/f{idx}"
        self.start_sec = idx * 344.0
        self.end_sec = (idx + 1) * 344.0
        self.state = "ACTIVE"


class _Gemini:
    """Gemini giả: ``model`` quá tải (hoặc hết hạn mức) từ đoạn ``stop_at`` trở đi."""

    def __init__(self, stop_at: int | None = None, *, overload: bool = True) -> None:
        self.stop_at = stop_at
        self.overload = overload
        self.calls: list[tuple[str, int]] = []

    def __call__(self, model, _prompt, _config, _validator, video_refs=None, **_kw):
        segment = int(video_refs[0].chunk_index) if video_refs else -1
        if self.stop_at is not None and segment >= self.stop_at:
            if self.overload:
                raise SubtitleTranslationError(REAL_503)
            raise QuotaExhaustedError("Model đã chạm giới hạn 20 request/ngày.")
        self.calls.append((model, segment))
        return {
            "source_lang": "zh",
            "characters": f"人物{segment} (Nhân vật {segment}) — vai",
            "overview": f"đoạn {segment + 1} ({model})",
            "glossary": "",
            "segment_summary": f"Diễn biến đoạn {segment + 1}.",
        }


def _run(
    checkpoint_dir: Path, model: str, gemini: _Gemini, n: int = 6
) -> object:
    adapter: GeminiSubtitleTranslator = bare_analysis_adapter(
        checkpoint_dir=checkpoint_dir
    )
    refs = [_Ref(i) for i in range(n)]
    lines = [
        TranslationLine(index=i, start_ms=0, end_ms=1, text=f"c{i}") for i in range(1, 6)
    ]
    with (
        patch.object(
            adapter, "_resolve_video_handles", lambda r: [f"h{i}" for i in range(n)]
        ),
        patch.object(adapter, "_build_config", lambda *a, **k: MagicMock()),
        patch.object(adapter, "_call_gemini", gemini),
        patch.object(adapter, "_ensure_available", lambda: None),
        # Model hạn mức ngày ít đã được xác nhận → quá tải thì dừng (như lượt thật).
        patch.object(adapter, "_overload_attempt_cap", lambda _model: 5),
    ):
        return adapter.analyze_global_context(
            lines, "vi", model_name=model, video_refs=refs
        )


class TestSwitchingModelKeepsFinishedSegments:
    def test_real_case_fallback_model_only_sends_missing_segments(
        self, tmp_path: Path
    ) -> None:
        strong = _Gemini(stop_at=4)
        partial = _run(tmp_path, STRONG, strong)
        assert partial.missing_segments == (5, 6)
        fallback = _Gemini()
        result = _run(tmp_path, FALLBACK, fallback)
        assert fallback.calls == [(FALLBACK, 4), (FALLBACK, 5)]
        assert result.missing_segments == ()
        assert "Nhân vật 0" in result.characters
        assert "Nhân vật 5" in result.characters

    def test_switching_back_and_forth_never_redoes_a_segment(
        self, tmp_path: Path
    ) -> None:
        _run(tmp_path, STRONG, _Gemini(stop_at=2))
        middle = _Gemini(stop_at=4)
        _run(tmp_path, "gemini-3.6-flash", middle)
        last = _Gemini()
        _run(tmp_path, STRONG, last)
        assert middle.calls == [("gemini-3.6-flash", 2), ("gemini-3.6-flash", 3)]
        assert last.calls == [(STRONG, 4), (STRONG, 5)]

    def test_daily_quota_stop_is_also_resumed_by_another_model(
        self, tmp_path: Path
    ) -> None:
        _run(tmp_path, STRONG, _Gemini(stop_at=3, overload=False))
        fallback = _Gemini()
        _run(tmp_path, FALLBACK, fallback)
        assert [segment for _model, segment in fallback.calls] == [3, 4, 5]

    def test_log_names_the_model_of_the_finished_segments(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _run(tmp_path, STRONG, _Gemini(stop_at=4))
        with caplog.at_level(logging.INFO):
            _run(tmp_path, FALLBACK, _Gemini())
        assert (
            "4/6 đoạn đã xong (đoạn 1-4: gemini-3.8-flash), chỉ gửi 2 đoạn còn thiếu "
            "bằng 'gemini-3.5-flash'" in caplog.text
        )

    def test_progress_records_which_model_did_each_segment(
        self, tmp_path: Path
    ) -> None:
        _run(tmp_path, STRONG, _Gemini(stop_at=2))
        _run(tmp_path, FALLBACK, _Gemini(stop_at=4))
        (saved_file,) = tmp_path.glob("analysis_*.json")
        saved = json.loads(saved_file.read_text(encoding="utf-8"))
        assert saved["segment_models"] == {
            "0": STRONG, "1": STRONG, "2": FALLBACK, "3": FALLBACK,
        }

    def test_finished_analysis_clears_the_shared_progress(self, tmp_path: Path) -> None:
        _run(tmp_path, STRONG, _Gemini(stop_at=3))
        _run(tmp_path, FALLBACK, _Gemini())
        assert list(tmp_path.glob("analysis_*.json")) == []

    def test_message_shown_to_the_user_is_now_true(self, tmp_path: Path) -> None:
        """Lời khuyên "đổi model phân tích — các đoạn đã xong được giữ" (v3.23.585)."""
        from subtitles_extractor.application.services.analysis_completeness import (
            describe_incomplete_analysis,
        )

        partial = _run(tmp_path, STRONG, _Gemini(stop_at=4))
        message = describe_incomplete_analysis(partial)
        assert "đổi model phân tích" in message
        assert "các đoạn đã xong được giữ" in message
        fallback = _Gemini()
        _run(tmp_path, FALLBACK, fallback)
        assert len(fallback.calls) == 2


class TestProgressSavedByOlderVersions:
    def test_old_progress_of_the_same_model_is_taken_over(
        self, tmp_path: Path
    ) -> None:
        """Tiến độ bản cũ lưu theo khoá có tên model: vẫn chạy tiếp được, rồi chuyển
        sang khoá dùng chung."""
        with patch.object(
            GeminiSubtitleTranslator, "_analysis_progress_key",
            lambda self, prompt, refs, cues: self._analysis_checkpoint_key(
                STRONG, prompt, refs, cues
            ),
        ):
            _run(tmp_path, STRONG, _Gemini(stop_at=3))
        resumed = _Gemini()
        _run(tmp_path, STRONG, resumed)
        assert [segment for _model, segment in resumed.calls] == [3, 4, 5]
        assert list(tmp_path.glob("analysis_*.json")) == []

    def test_old_progress_becomes_shared_after_the_first_load(
        self, tmp_path: Path
    ) -> None:
        with patch.object(
            GeminiSubtitleTranslator, "_analysis_progress_key",
            lambda self, prompt, refs, cues: self._analysis_checkpoint_key(
                STRONG, prompt, refs, cues
            ),
        ):
            _run(tmp_path, STRONG, _Gemini(stop_at=2))
        _run(tmp_path, STRONG, _Gemini(stop_at=3))
        fallback = _Gemini()
        _run(tmp_path, FALLBACK, fallback)
        assert [segment for _model, segment in fallback.calls] == [3, 4, 5]


class TestDescribeSegmentModels:
    def test_groups_consecutive_segments(self) -> None:
        models = {0: STRONG, 1: STRONG, 2: STRONG, 4: FALLBACK}
        assert describe_segment_models(models, FALLBACK) == (
            " (đoạn 1-3: gemini-3.8-flash; đoạn 5: gemini-3.5-flash)"
        )

    def test_silent_when_every_segment_is_from_this_model(self) -> None:
        assert describe_segment_models({0: STRONG, 1: STRONG}, STRONG) == ""

    def test_silent_without_records(self) -> None:
        assert describe_segment_models({}, STRONG) == ""


class TestStoreRoundTrip:
    def test_segment_models_survive_save_and_load(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.translation.analysis_checkpoint import (
            AnalysisProgress,
        )

        store = AnalysisCheckpointStore(tmp_path)
        store.save("k", AnalysisProgress(done_segments={0, 1}, segment_models={
            0: STRONG, 1: FALLBACK,
        }))
        loaded = store.load("k")
        assert loaded is not None
        assert loaded.segment_models == {0: STRONG, 1: FALLBACK}

    def test_file_without_segment_models_still_loads(self, tmp_path: Path) -> None:
        (tmp_path / "analysis_k.json").write_text(
            json.dumps({
                "schema": "subtitles_extractor.analysis_checkpoint/1",
                "done_segments": [0], "cues": [], "segment_summaries": {},
            }),
            encoding="utf-8",
        )
        loaded = AnalysisCheckpointStore(tmp_path).load("k")
        assert loaded is not None
        assert loaded.done_segments == {0}
        assert loaded.segment_models == {}
