"""[v3.23.561] Hồi quy: sau khi xoay key, cả lượt dịch chạy KHÔNG video mà không ai hay.

Nhật ký thật v3.23.560:

1. Quét bổ sung Visual Cues gọi Gemini KHÔNG kèm ``video_refs`` và KHÔNG tính token
   video → quota không điều tiết TPM → lô 2 dính 429 → 429 bị coi là lô "không đính
   video" nên xoay key tự do → handle video thành của key cũ → 403 → heal không chạy vì
   thiếu ``video_refs`` → quét bổ sung thất bại (độ phủ cues vẫn 21%).
2. Khâu dịch sau đó chạy trên key mới: mỗi lô resolve handle → 403 → BỎ QUA LẶNG LẼ → lô
   gửi đi không video. 261 lần như vậy, cả 3 giai đoạn dịch mất ngữ cảnh hình dù người
   dùng bật đính video.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    replace_refs_by_range,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    RemoteVideoRef,
)


def _ref(idx: int, start: float, end: float, name: str) -> RemoteVideoRef:
    return RemoteVideoRef(idx, name, start, end, "ACTIVE")


def _old_refs() -> list[RemoteVideoRef]:
    return [_ref(0, 0.0, 10.0, "files/OLD0"), _ref(1, 10.0, 20.0, "files/OLD1")]


def _fresh_refs() -> list[RemoteVideoRef]:
    return [_ref(0, 0.0, 10.0, "files/NEW0"), _ref(1, 10.0, 20.0, "files/NEW1")]


class _Harness:
    """Adapter mà file 'OLD' luôn 403 (thuộc key cũ), file 'NEW' resolve được."""

    def __init__(self, *, stale: bool = False, reupload_ok: bool = True) -> None:
        self.adapter = GeminiSubtitleTranslator(api_key="KEY1")
        self.reuploads: list[list[tuple[float, float]]] = []
        self.adapter._video_handles_stale = stale

        def _resolve(refs: list[Any] | None) -> list[Any]:
            return [
                SimpleNamespace(name=ref.remote_name)
                for ref in refs or []
                if "NEW" in ref.remote_name
            ]

        def _reupload(_key: str, ranges: list[tuple[float, float]]) -> list[Any]:
            self.reuploads.append(list(ranges))
            if not reupload_ok:
                return []
            wanted = set(ranges)
            return [r for r in _fresh_refs() if (r.start_sec, r.end_sec) in wanted]

        self.adapter._resolve_video_handles = _resolve  # type: ignore[assignment]
        self.adapter._video_reupload_cb = _reupload


class TestReplaceRefsByRange:
    def test_replaces_matching_ranges_in_place(self) -> None:
        target = _old_refs()
        replace_refs_by_range(target, [_ref(1, 10.0, 20.0, "files/NEW1")])
        assert [r.remote_name for r in target] == ["files/OLD0", "files/NEW1"]

    def test_unmatched_fresh_refs_are_ignored(self) -> None:
        target = _old_refs()
        replace_refs_by_range(target, [_ref(9, 90.0, 99.0, "files/X")])
        assert [r.remote_name for r in target] == ["files/OLD0", "files/OLD1"]


class TestHandlesForBatch:
    def test_cached_handles_cost_no_reupload(self) -> None:
        harness = _Harness()
        refs = _fresh_refs()
        handles = harness.adapter._handles_for_batch([refs[0]], refs, 0)
        assert [h.name for h in handles] == ["files/NEW0"]
        assert harness.reuploads == []

    def test_403_handle_is_healed_and_written_back(self) -> None:
        harness = _Harness()
        all_refs = _old_refs()
        covering = [all_refs[1]]
        handles = harness.adapter._handles_for_batch(covering, all_refs, 0)
        assert [h.name for h in handles] == ["files/NEW1"]
        assert harness.reuploads == [[(10.0, 20.0)]]
        assert covering[0].remote_name == "files/NEW1"
        assert [r.remote_name for r in all_refs] == ["files/OLD0", "files/NEW1"]

    def test_next_batch_reuses_healed_ref(self) -> None:
        harness = _Harness()
        all_refs = _old_refs()
        harness.adapter._handles_for_batch([all_refs[1]], all_refs, 0)
        harness.adapter._handles_for_batch([all_refs[1]], all_refs, 1)
        assert len(harness.reuploads) == 1

    def test_stale_flag_goes_straight_to_reupload(self) -> None:
        harness = _Harness(stale=True)
        all_refs = _fresh_refs()
        handles = harness.adapter._handles_for_batch([all_refs[0]], all_refs, 0)
        assert harness.reuploads == [[(0.0, 10.0)]]
        assert len(handles) == 1

    def test_failed_reupload_warns_and_returns_what_it_has(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = _Harness(reupload_ok=False)
        all_refs = _old_refs()
        with caplog.at_level("WARNING"):
            handles = harness.adapter._handles_for_batch([all_refs[0]], all_refs, 4)
        assert handles == []
        assert "Lô 5 gửi THIẾU video" in caplog.text
        events = get_process_log().snapshot()["events"]
        assert any(e.get("event") == "video_missing" for e in events)


def _lines(count: int = 20) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=i + 1,
            start_ms=i * 1000,
            end_ms=i * 1000 + 900,
            text=f"句{i}",
            original_text=f"句{i}",
        )
        for i in range(count)
    ]


class TestTranslateStageKeepsVideoAfterKeyRotation:
    def test_every_batch_gets_its_video_after_heal(self) -> None:
        harness = _Harness(stale=True)
        adapter = harness.adapter
        adapter._ensure_available = lambda: None  # type: ignore[assignment]
        sent: list[list[str]] = []

        def _single_batch(*, batch, video_files, **_kwargs):
            sent.append([h.name for h in video_files or []])
            return [replace(line, text=f"dịch {line.index}") for line in batch]

        adapter._translate_single_batch = _single_batch  # type: ignore[assignment]
        stage = TranslationStageConfig(
            kind=TranslationStageKind.LITERAL, model_name="m", batch_size=10
        )
        lines = _lines()
        adapter.translate_stage(
            stage=stage,
            context=TranslationContext(target_lang="Vietnamese"),
            source_lines=lines,
            input_lines=lines,
            has_prior_translation=False,
            video_refs=_old_refs(),
            attach_video=True,
        )
        assert sent == [["files/NEW0"], ["files/NEW1"]]
        assert len(harness.reuploads) == 2


class TestCueScanPassesVideoContextToCall:
    def test_video_refs_and_video_tokens_reach_the_call(self) -> None:
        harness = _Harness()
        adapter = harness.adapter
        adapter._ensure_available = lambda: None  # type: ignore[assignment]
        adapter._build_config = lambda *args, **kwargs: None  # type: ignore[assignment]
        captured: list[dict[str, Any]] = []

        def _call(_model: str, prompt: str, *_args: Any, **kwargs: Any) -> dict[str, Any]:
            captured.append(kwargs)
            return {"cues": []}

        adapter._call_gemini = _call  # type: ignore[assignment]
        adapter.analyze_visual_cues(
            _lines(),
            "Vietnamese",
            video_refs=_fresh_refs(),
            batch_size=150,
            sleep_between_s=0.0,
        )
        assert len(captured) == 2
        for kwargs, expected in zip(captured, ("files/NEW0", "files/NEW1"), strict=True):
            assert [r.remote_name for r in kwargs["video_refs"]] == [expected]
            # 10 giây video nhân token/giây phân tích — không chỉ token chữ.
            assert kwargs["est_tokens"] >= 10 * adapter._analysis_video_tps

    def test_unresolvable_chunk_never_falls_back_to_whole_film(self) -> None:
        harness = _Harness(reupload_ok=False)
        adapter = harness.adapter
        adapter._ensure_available = lambda: None  # type: ignore[assignment]
        adapter._build_config = lambda *args, **kwargs: None  # type: ignore[assignment]
        sent: list[int] = []

        def _call(*_args: Any, video_files=None, **_kwargs: Any) -> dict[str, Any]:
            sent.append(len(video_files or []))
            return {"cues": []}

        adapter._call_gemini = _call  # type: ignore[assignment]
        refs = [_ref(0, 0.0, 10.0, "files/NEW0"), _ref(1, 10.0, 20.0, "files/OLD1")]
        adapter.analyze_visual_cues(
            _lines(), "Vietnamese", video_refs=refs, batch_size=150, sleep_between_s=0.0
        )
        assert sent == [1, 0]
