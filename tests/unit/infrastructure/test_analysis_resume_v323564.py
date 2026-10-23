"""[v3.23.564] Hồi quy: hết hạn mức ngày giữa chừng làm MẤT TRẮNG phân tích ngữ cảnh.

Nhật ký thật v3.23.563: phân tích bằng gemini-3.8-flash (hạn mức 20 request/ngày) chạy 50
phút, xong 20/22 đoạn. Tới lượt thử lại cuối phiên, mọi key đều hết hạn mức →
``QuotaExhaustedError`` (không phải ``SubtitleTranslationError``) thoát khỏi vòng lặp và
TOÀN BỘ kết quả bị vứt; bản phân tích cũ của model yếu hơn vẫn được dùng. Phim cần 22
request mà hạn mức chỉ 20: không lưu tiến độ thì model mạnh nhất không bao giờ xong được.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    SubtitleTranslationError,
    TranslationLine,
)
from subtitles_extractor.infrastructure.translation.analysis_checkpoint import (
    AnalysisCheckpointStore,
    AnalysisProgress,
    analysis_checkpoint_key,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    QuotaExhaustedError,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter


class _Ref:
    def __init__(self, idx: int) -> None:
        self.chunk_index = idx
        self.remote_name = f"files/f{idx}"
        self.start_sec = idx * 100.0
        self.end_sec = (idx + 1) * 100.0
        self.state = "ACTIVE"


def _adapter(checkpoint_dir: Path | None) -> GeminiSubtitleTranslator:
    return bare_analysis_adapter(checkpoint_dir=checkpoint_dir)


class _Gemini:
    """Gemini giả: trả phân tích theo đoạn; có thể hết hạn mức sau N lượt gọi."""

    def __init__(
        self, quota: int = 99, fail_segments: frozenset[int] = frozenset()
    ) -> None:
        self.quota = quota
        self.fail_segments = fail_segments
        self.calls: list[int] = []
        self.failed_once: set[int] = set()

    def __call__(self, _model, _prompt, _config, _validator, video_refs=None, **_kw):
        segment = int(video_refs[0].chunk_index) if video_refs else -1
        if len(self.calls) >= self.quota:
            raise QuotaExhaustedError("Model đã chạm giới hạn 20 request/ngày.")
        self.calls.append(segment)
        if segment in self.fail_segments and segment not in self.failed_once:
            self.failed_once.add(segment)
            raise SubtitleTranslationError("Gọi Gemini thất bại: 503 UNAVAILABLE")
        return {
            "source_lang": "zh",
            "characters": f"人物{segment} (Nhân vật {segment}) — vai",
            "overview": f"đoạn {segment + 1}",
            "glossary": "",
        }


def _run(adapter: GeminiSubtitleTranslator, gemini: _Gemini, n: int = 4):
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
    ):
        return adapter.analyze_global_context(lines, "vi", video_refs=refs)


class TestQuotaNeverDiscardsFinishedSegments:
    def test_quota_on_final_retry_keeps_partial_analysis(self) -> None:
        """Đúng ca thật: một đoạn lỗi 503, lượt thử lại cuối phiên thì hết hạn mức."""
        gemini = _Gemini(quota=3, fail_segments=frozenset({1}))
        result = _run(_adapter(None), gemini)
        assert gemini.calls == [0, 1, 2]
        assert "đoạn 1" in result.overview
        assert "đoạn 3" in result.overview

    def test_quota_mid_run_stops_sending_and_keeps_what_it_has(self) -> None:
        gemini = _Gemini(quota=2)
        result = _run(_adapter(None), gemini)
        assert gemini.calls == [0, 1]
        assert "đoạn 2" in result.overview

    def test_quota_before_any_segment_is_still_reported(self) -> None:
        with pytest.raises(QuotaExhaustedError):
            _run(_adapter(None), _Gemini(quota=0))


class TestResumeFromCheckpoint:
    def test_second_run_only_sends_missing_segments(self, tmp_path: Path) -> None:
        first = _Gemini(quota=2)
        _run(_adapter(tmp_path), first)
        second = _Gemini()
        result = _run(_adapter(tmp_path), second)
        assert second.calls == [2, 3]
        assert result.overview.splitlines()[0] == "đoạn 1"
        assert "đoạn 4" in result.overview

    def test_complete_run_clears_the_checkpoint(self, tmp_path: Path) -> None:
        _run(_adapter(tmp_path), _Gemini())
        assert list(tmp_path.glob("analysis_*.json")) == []

    def test_incomplete_run_keeps_the_checkpoint(self, tmp_path: Path) -> None:
        _run(_adapter(tmp_path), _Gemini(quota=1))
        assert len(list(tmp_path.glob("analysis_*.json"))) == 1

    def test_legacy_key_still_depends_on_the_model(self, tmp_path: Path) -> None:
        """[v3.23.593] Tiến độ nay dùng chung cho mọi model (xem
        ``test_analysis_progress_across_models_v323593``); khoá cũ có tên model chỉ còn
        dùng để nhận tiến độ lưu bởi bản cũ, nên vẫn phải phân biệt model."""
        _run(_adapter(tmp_path), _Gemini(quota=2))
        adapter = _adapter(tmp_path)
        refs = [_Ref(i) for i in range(4)]
        key_a = adapter._analysis_checkpoint_key("gemini-3.8-flash", "p", refs, True)
        key_b = adapter._analysis_checkpoint_key("gemini-3.6-flash", "p", refs, True)
        assert key_a != key_b


class TestCheckpointStore:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = AnalysisCheckpointStore(tmp_path)
        progress = AnalysisProgress(
            done_segments={0, 2},
            accumulated=SubtitleContextAnalysis(
                source_lang="zh", characters="宋钱", overview="o", glossary="g"
            ),
            cues={5: {"id": 5, "spk": "宋钱", "to": "许桃花", "cue": ""}},
        )
        store.save("k", progress)
        loaded = store.load("k")
        assert loaded is not None
        assert loaded.done_segments == {0, 2}
        assert loaded.accumulated == progress.accumulated
        assert loaded.cues == progress.cues

    def test_stale_checkpoint_is_ignored(self, tmp_path: Path) -> None:
        store = AnalysisCheckpointStore(tmp_path, max_age_s=60)
        store.save("k", AnalysisProgress(done_segments={0}))
        old = time.time() - 3600
        os.utime(tmp_path / "analysis_k.json", (old, old))
        assert store.load("k") is None

    def test_corrupt_file_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "analysis_k.json").write_text("{không phải json", encoding="utf-8")
        assert AnalysisCheckpointStore(tmp_path).load("k") is None

    def test_foreign_schema_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "analysis_k.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
        assert AnalysisCheckpointStore(tmp_path).load("k") is None

    def test_key_depends_on_every_part(self) -> None:
        assert analysis_checkpoint_key("a", "b") != analysis_checkpoint_key("a", "c")
        assert analysis_checkpoint_key("a", "b") == analysis_checkpoint_key("a", "b")


class TestSeriesContextDoesNotBreakResume:
    def test_series_context_change_keeps_progress(self, tmp_path: Path) -> None:
        """Ngữ cảnh phim bộ đổi sau mỗi lần dịch — không được làm mất tiến độ."""
        refs = [_Ref(i) for i in range(4)]
        lines = [TranslationLine(index=1, start_ms=0, end_ms=1, text="c1")]
        first = _Gemini(quota=2)
        second = _Gemini()
        for gemini, prior in (
            (first, "# Nhân vật đã thiết lập: A"),
            (second, "khác hẳn"),
        ):
            adapter = _adapter(tmp_path)
            with (
                patch.object(adapter, "_resolve_video_handles", lambda r: ["h"] * 4),
                patch.object(adapter, "_build_config", lambda *a, **k: MagicMock()),
                patch.object(adapter, "_call_gemini", gemini),
                patch.object(adapter, "_ensure_available", lambda: None),
            ):
                adapter.analyze_global_context(
                    lines, "vi", video_refs=refs, prior_context=prior
                )
        assert second.calls == [2, 3]


class TestDriveRootIsNotASeries:
    @pytest.mark.parametrize(
        "path",
        [
            "D:\\傻子落水獲醫神傳承.mp4",
            "d:/phim le.mkv",
            "E:\\Movie.2024.1080p.mkv",
        ],
    )
    def test_film_at_drive_root_has_no_series(self, path: str) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            derive_series_key,
        )

        assert derive_series_key(path) == ""

    def test_episode_pattern_at_drive_root_still_groups(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            derive_series_key,
        )

        assert derive_series_key("G:\\NOVA.S51E16.mkv") == "NOVA"
        assert derive_series_key("D:\\Phim\\TuTien\\tap01.mkv") == "TuTien"


class TestHeldProgress:
    """[v3.23.643] Tiến độ phân tích sống tới khi các bước hoàn thiện (người nói, tự kiểm tra) xong."""

    def test_held_progress_survives_a_complete_run_until_released(self, tmp_path: Path) -> None:
        adapter = _adapter(tmp_path)
        adapter.hold_analysis_progress()
        _run(adapter, _Gemini())
        assert len(list(tmp_path.glob("analysis_*.json"))) == 1       # chưa xoá: bước sau còn chạy
        adapter.release_analysis_progress(clear=True)
        assert list(tmp_path.glob("analysis_*.json")) == []

    def test_interrupted_post_steps_let_the_rerun_resend_nothing(self, tmp_path: Path) -> None:
        adapter = _adapter(tmp_path)
        adapter.hold_analysis_progress()
        first = _Gemini()
        _run(adapter, first)
        adapter.release_analysis_progress(clear=False)                # bị huỷ ở bước sau
        rerun = _Gemini()
        result = _run(_adapter(tmp_path), rerun)
        assert first.calls == [0, 1, 2, 3] and rerun.calls == []      # 4 đoạn không gửi lại
        assert "đoạn 4" in result.overview

    def test_without_hold_the_old_behaviour_is_unchanged(self, tmp_path: Path) -> None:
        adapter = _adapter(tmp_path)
        _run(adapter, _Gemini())
        assert list(tmp_path.glob("analysis_*.json")) == []
