"""[v3.23.560] Hồi quy: bộ quét Visual Cues nhiều lô làm đổ cả lượt dịch.

Nhật ký thật sau v3.23.559: bước quét bổ sung 3060 dòng thiếu cues chạy xong lô đầu rồi
ném "Đã huỷ trong lúc nghỉ giữa các lô" — người dùng KHÔNG bấm huỷ — và cả lượt dịch dừng.
Lặp lại 3 lần liền (09:51, 10:32, 11:00).

1. ``_interruptible_sleep`` trả True khi BỊ HUỶ, nhưng bộ quét kiểm ``not ...`` → mọi lần
   nghỉ bình thường đều thành "đã huỷ". Quét cues nhiều lô chưa từng chạy hết.
2. Bước quét là phụ trợ nhưng một lỗi "huỷ" giả vẫn làm đổ cả lượt dịch.
3. Lô chia theo số dòng vắt qua ranh giới đoạn video → kèm 2-4 đoạn (~138k token mỗi
   đoạn), model nhẹ trả 503/504 DEADLINE_EXCEEDED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import pytest

from subtitles_extractor.application.services.visual_cue_batching import (
    batch_lines_by_video_chunk,
    chunk_index_for_time,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationCancelledError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


@dataclass(frozen=True)
class _Ref:
    remote_name: str
    start_sec: float
    end_sec: float


_REFS = [
    _Ref("files/a", 0.0, 10.0),
    _Ref("files/b", 10.0, 20.0),
    _Ref("files/c", 20.0, 30.0),
]
_BOUNDS = [(ref.start_sec, ref.end_sec) for ref in _REFS]


def _line(index: int, start_sec: float, duration_sec: float = 0.8) -> TranslationLine:
    return TranslationLine(
        index=index,
        start_ms=int(start_sec * 1000),
        end_ms=int((start_sec + duration_sec) * 1000),
        text=f"句{index}",
    )


def _lines_every_second(count: int = 30) -> list[TranslationLine]:
    return [_line(i + 1, float(i)) for i in range(count)]


class TestBatchingByVideoChunk:
    def test_no_batch_spans_two_chunks(self) -> None:
        batches = batch_lines_by_video_chunk(_lines_every_second(), _BOUNDS, 150)
        assert [[ln.index for ln in batch] for batch in batches] == [
            list(range(1, 11)),
            list(range(11, 21)),
            list(range(21, 31)),
        ]

    def test_batch_size_still_applies_inside_a_chunk(self) -> None:
        batches = batch_lines_by_video_chunk(_lines_every_second(10), _BOUNDS, 4)
        assert [len(batch) for batch in batches] == [4, 4, 2]

    def test_line_is_assigned_by_start_time(self) -> None:
        """Câu bắt đầu ở 9.5s kéo tới 10.5s vẫn thuộc đoạn đầu, không kéo thêm đoạn kế."""
        batches = batch_lines_by_video_chunk(
            [_line(1, 9.5, 1.0), _line(2, 10.2)], _BOUNDS, 150
        )
        assert [[ln.index for ln in batch] for batch in batches] == [[1], [2]]

    def test_sparse_subset_keeps_order(self) -> None:
        subset = [_line(5, 4.0), _line(900, 25.0), _line(901, 26.0)]
        batches = batch_lines_by_video_chunk(subset, _BOUNDS, 150)
        assert [[ln.index for ln in batch] for batch in batches] == [[5], [900, 901]]

    def test_without_video_only_batch_size_applies(self) -> None:
        batches = batch_lines_by_video_chunk(_lines_every_second(7), [], 3)
        assert [len(batch) for batch in batches] == [3, 3, 1]

    def test_invalid_batch_size_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            batch_lines_by_video_chunk(_lines_every_second(3), _BOUNDS, 0)

    @pytest.mark.parametrize(
        ("start_sec", "expected"),
        [(0.0, 0), (9.999, 0), (10.0, 1), (29.0, 2), (45.0, 2), (-1.0, 0)],
    )
    def test_chunk_index_for_time(self, start_sec: float, expected: int) -> None:
        assert chunk_index_for_time(start_sec, _BOUNDS) == expected


def _scanner(calls: list[list[str]]) -> GeminiSubtitleTranslator:
    """Adapter dựng bằng ``__new__``, lời gọi mạng thay bằng hàm giả ghi lại lô/đoạn."""
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    adapter._ensure_available = lambda: None
    adapter._build_config = lambda *args, **kwargs: None
    adapter._resolve_handles_with_heal = lambda refs, _extra: list(refs or [])

    def _resolve(refs: list[Any] | None) -> list[Any]:
        calls.append([ref.remote_name for ref in refs or []])
        return list(refs or [])

    def _call(_model: str, prompt: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        ids = [int(token) for token in _ids_in(prompt)]
        return {"cues": [{"id": i, "spk": "A", "to": "B", "cue": ""} for i in ids]}

    adapter._resolve_video_handles = _resolve
    adapter._call_gemini = _call
    return adapter


def _ids_in(prompt: str) -> list[str]:
    start = prompt.index("[", prompt.index("<lines>"))
    end = prompt.rindex("]") + 1
    return [str(item["id"]) for item in json.loads(prompt[start:end])]


class TestScannerRunsEveryBatch:
    def test_multi_batch_scan_completes_without_false_cancel(self) -> None:
        calls: list[list[str]] = []
        cues = _scanner(calls).analyze_visual_cues(
            _lines_every_second(),
            "Vietnamese",
            video_refs=list(_REFS),
            batch_size=150,
            sleep_between_s=0.01,
            cancel_cb=lambda: False,
        )
        assert [cue.line_no for cue in cues] == list(range(1, 31))

    def test_each_batch_attaches_exactly_one_chunk(self) -> None:
        calls: list[list[str]] = []
        _scanner(calls).analyze_visual_cues(
            _lines_every_second(),
            "Vietnamese",
            video_refs=list(_REFS),
            batch_size=150,
            sleep_between_s=0.0,
        )
        assert calls == [["files/a"], ["files/b"], ["files/c"]]

    def test_real_cancel_during_pause_still_stops(self) -> None:
        calls: list[list[str]] = []
        with pytest.raises(TranslationCancelledError):
            _scanner(calls).analyze_visual_cues(
                _lines_every_second(),
                "Vietnamese",
                video_refs=list(_REFS),
                batch_size=150,
                sleep_between_s=0.05,
                cancel_cb=lambda: len(calls) >= 1,
            )


class _FalseCancelTranslator:
    """Bộ quét ném 'đã huỷ' dù không ai huỷ — đúng hành vi lỗi của bản cũ."""

    def __init__(self) -> None:
        self.translated = False

    def analyze_visual_cues(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        raise TranslationCancelledError("Đã huỷ trong lúc nghỉ giữa các lô.")

    def translate_stage(self, *, input_lines, **_kwargs) -> list[TranslationLine]:
        self.translated = True
        return [replace(line, text=f"dịch {line.index}") for line in input_lines]

    def is_available(self) -> bool:
        return True


def _request(visual_cues: str) -> TranslateSubtitlesRequest:
    events = [
        SubtitleEvent(index=i, text=f"句{i}", interval=TimeInterval(i - 1.0, i - 0.2))
        for i in range(1, 11)
    ]
    return TranslateSubtitlesRequest(
        events=events,
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        ],
        context=TranslationContext(target_lang="Vietnamese", visual_cues=visual_cues),
        video_refs=(object(),),
        enable_visual_cues=True,
    )


class TestFalseCancelNeverKillsTranslation:
    def test_top_up_false_cancel_translation_continues(self) -> None:
        low_coverage = json.dumps([{"id": 1, "spk": "A", "to": "B", "cue": ""}])
        translator = _FalseCancelTranslator()
        response = TranslateSubtitlesUseCase(translator=translator).execute(
            _request(low_coverage), cancel_cb=lambda: False
        )
        assert translator.translated
        assert len(response.events) == 10
        assert response.visual_cues == ""

    def test_full_scan_false_cancel_translation_continues(self) -> None:
        translator = _FalseCancelTranslator()
        response = TranslateSubtitlesUseCase(translator=translator).execute(
            _request(""), cancel_cb=lambda: False
        )
        assert translator.translated
        assert len(response.events) == 10

    def test_user_cancel_during_full_scan_still_stops(self) -> None:
        translator = _FalseCancelTranslator()
        cancelled = {"flag": False}

        def _cancel() -> bool:
            return cancelled["flag"]

        def _scan(*_args: Any, **_kwargs: Any) -> list[Any]:
            cancelled["flag"] = True
            raise TranslationCancelledError("Người dùng đã huỷ phân tích Visual Cues.")

        translator.analyze_visual_cues = _scan
        with pytest.raises(TranslationCancelledError):
            TranslateSubtitlesUseCase(translator=translator).execute(
                _request(""), cancel_cb=_cancel
            )
        assert not translator.translated


class _SlowScanner:
    """Bộ quét báo tiến độ 2 lô — để kiểm giao diện nhận được "đang quét x%"."""

    def analyze_visual_cues(self, lines, _lang, progress_cb=None, **_kwargs):
        from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue

        for fraction in (0.5, 1.0):
            if progress_cb is not None:
                progress_cb(fraction)
        return [
            VisualCue(line_no=line.index, speaker="A", addressee="B") for line in lines
        ]


def test_top_up_reports_progress_so_ui_does_not_look_frozen() -> None:
    labels: list[str] = []
    low_coverage = json.dumps([{"id": 1, "spk": "A", "to": "B", "cue": ""}])
    lines = [_line(i, float(i)) for i in range(1, 11)]
    TranslateSubtitlesUseCase(translator=_SlowScanner())._top_up_context_cues(
        lines, _request(low_coverage), None, None, lambda _f, label: labels.append(label)
    )
    assert labels[-1].endswith("100%")
    assert len(labels) == 3
