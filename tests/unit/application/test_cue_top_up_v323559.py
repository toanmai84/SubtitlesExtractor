"""[v3.23.559] Hồi quy: Visual Cues từ phân tích ngữ cảnh phủ QUÁ THẤP mà không ai hay.

Chuẩn đoán thật v3.23.558: phân tích bằng gemini-3.6-flash chạy đủ 22 đoạn không lỗi nhưng
chỉ trả cues cho 27% số dòng (đoạn 19: 1/208 dòng). Hệ quả:

1. Chuẩn đoán ngữ cảnh im lặng (cờ cũ chỉ báo dưới 2%).
2. Khâu dịch thấy "đã có cues" nên bỏ qua bộ quét riêng → 73% số dòng dịch mà không biết
   ai nói với ai.
3. Số liệu xưng hô trong chuẩn đoán dịch "đẹp ảo" (cặp xét giảm 67 → 23).
4. Bộ quét cues vứt oan mọi cue có id > số dòng GỬI ĐI — chặn đứng việc quét bổ sung
   một tập con dòng rải tới cuối phim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from subtitles_extractor.application.services.context_diagnostics import (
    CUES_DEAD_ZONE_WINDOW,
    build_context_diagnostics,
    cue_dead_zones,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    address_cue_coverage,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    MIN_CONTEXT_CUE_COVERAGE,
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
    _StageCheckpoint,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationCancelledError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
    VisualCue,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
    SqliteTranslationSessionStore,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

TOTAL_LINES = 20


def _lines(count: int = TOTAL_LINES) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"句子{i}"
        )
        for i in range(1, count + 1)
    ]


def _cues_json(
    line_nos: list[int], speaker: str = "宋钱", addressee: str = "许桃花"
) -> str:
    return json.dumps(
        [{"id": n, "spk": speaker, "to": addressee, "cue": ""} for n in line_nos],
        ensure_ascii=False,
    )


class _ScanningTranslator:
    """Translator giả: ghi lại dòng được gửi quét cues, trả cue 'quét' cho mỗi dòng."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.scanned_batches: list[list[int]] = []
        self.error = error

    def analyze_visual_cues(
        self, source_lines, target_lang, **_kwargs
    ) -> list[VisualCue]:
        self.scanned_batches.append([line.index for line in source_lines])
        if self.error is not None:
            raise self.error
        return [
            VisualCue(line_no=line.index, speaker="QUÉT", addressee="QUÉT")
            for line in source_lines
        ]

    def translate_stage(self, *, input_lines, **_kwargs) -> list[TranslationLine]:
        return [replace(line, text=f"dịch {line.index}") for line in input_lines]

    def is_available(self) -> bool:
        return True


def _request(visual_cues: str) -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        ],
        context=TranslationContext(target_lang="Vietnamese", visual_cues=visual_cues),
        video_refs=(object(),),
        enable_visual_cues=True,
    )


def _merged(request: TranslateSubtitlesRequest) -> dict[int, str]:
    return {
        cue.line_no: cue.speaker
        for cue in deserialize_visual_cues(request.context.visual_cues)
    }


class TestTopUpContextCues:
    def test_low_coverage_scans_only_missing_lines(self) -> None:
        translator = _ScanningTranslator()
        request = _request(_cues_json([1, 2, 3, 4, 5]))  # 25% < 60%
        result = TranslateSubtitlesUseCase(translator=translator)._top_up_context_cues(
            _lines(), request, None, None
        )
        assert translator.scanned_batches == [list(range(6, TOTAL_LINES + 1))]
        assert set(_merged(result)) == set(range(1, TOTAL_LINES + 1))

    def test_existing_cues_take_precedence(self) -> None:
        """Cues có sẵn (có thể người dùng đã sửa tay) KHÔNG bao giờ bị bản quét đè."""
        translator = _ScanningTranslator()
        request = _request(_cues_json([1, 2, 3]))
        result = TranslateSubtitlesUseCase(translator=translator)._top_up_context_cues(
            _lines(), request, None, None
        )
        merged = _merged(result)
        assert all(merged[n] == "宋钱" for n in (1, 2, 3))
        assert merged[4] == "QUÉT"

    def test_sufficient_coverage_costs_nothing(self) -> None:
        translator = _ScanningTranslator()
        covered = list(range(1, int(TOTAL_LINES * MIN_CONTEXT_CUE_COVERAGE) + 1))
        request = _request(_cues_json(covered))
        result = TranslateSubtitlesUseCase(translator=translator)._top_up_context_cues(
            _lines(), request, None, None
        )
        assert result is request
        assert translator.scanned_batches == []

    def test_no_context_cues_is_left_to_the_full_scanner(self) -> None:
        translator = _ScanningTranslator()
        request = _request("")
        result = TranslateSubtitlesUseCase(translator=translator)._top_up_context_cues(
            _lines(), request, None, None
        )
        assert result is request
        assert translator.scanned_batches == []

    def test_scan_failure_keeps_the_request(self) -> None:
        translator = _ScanningTranslator(
            error=SubtitleTranslationError("503 UNAVAILABLE")
        )
        request = _request(_cues_json([1, 2]))
        result = TranslateSubtitlesUseCase(translator=translator)._top_up_context_cues(
            _lines(), request, None, None
        )
        assert result is request

    def test_cancellation_is_not_swallowed(self) -> None:
        """Người dùng THẬT SỰ huỷ thì phải dừng (từ v3.23.560 phân biệt với huỷ 'giả')."""
        translator = _ScanningTranslator(error=TranslationCancelledError("huỷ"))
        request = _request(_cues_json([1, 2]))
        with pytest.raises(TranslationCancelledError):
            TranslateSubtitlesUseCase(translator=translator)._top_up_context_cues(
                _lines(), request, None, lambda: True
            )

    def test_checkpoint_reuse_skips_second_scan(self, tmp_path: Path) -> None:
        request = _request(_cues_json([1, 2, 3]))
        first = _ScanningTranslator()
        checkpoint = _StageCheckpoint(tmp_path, "k")
        TranslateSubtitlesUseCase(translator=first)._top_up_context_cues(
            _lines(), request, checkpoint, None
        )
        second = _ScanningTranslator()
        result = TranslateSubtitlesUseCase(translator=second)._top_up_context_cues(
            _lines(), request, _StageCheckpoint(tmp_path, "k"), None
        )
        assert len(first.scanned_batches) == 1
        assert second.scanned_batches == []
        assert set(_merged(result)) == set(range(1, TOTAL_LINES + 1))

    def test_execute_returns_topped_up_cues_for_persistence(self) -> None:
        translator = _ScanningTranslator()
        events = [
            SubtitleEvent(
                index=i, text=f"句子{i}", interval=TimeInterval(i - 1.0, i - 0.1)
            )
            for i in range(1, TOTAL_LINES + 1)
        ]
        request = replace(_request(_cues_json([1, 2, 3])), events=events)
        response = TranslateSubtitlesUseCase(translator=translator).execute(request)
        assert translator.scanned_batches == [list(range(4, TOTAL_LINES + 1))]
        assert {c.line_no for c in deserialize_visual_cues(response.visual_cues)} == set(
            range(1, TOTAL_LINES + 1)
        )

    def test_execute_without_top_up_returns_empty_cues(self) -> None:
        translator = _ScanningTranslator()
        events = [
            SubtitleEvent(
                index=i, text=f"句子{i}", interval=TimeInterval(i - 1.0, i - 0.1)
            )
            for i in range(1, TOTAL_LINES + 1)
        ]
        request = replace(
            _request(_cues_json(list(range(1, TOTAL_LINES + 1)))), events=events
        )
        response = TranslateSubtitlesUseCase(translator=translator).execute(request)
        assert translator.scanned_batches == []
        assert response.visual_cues == ""


class TestScannerKeepsCuesOfSparseSubset:
    def test_ids_beyond_subset_length_are_kept(self) -> None:
        """Gửi 2 dòng (id 3900, 3902): trước đây trần id = 2 nên vứt SẠCH cả hai cue."""
        adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        adapter._ensure_available = lambda: None
        adapter._build_config = lambda *args, **kwargs: None
        adapter._resolve_handles_with_heal = lambda refs, _extra: []
        adapter._call_gemini = lambda *args, **kwargs: {
            "cues": [
                {"id": 3900, "spk": "宋钱", "to": "许桃花", "cue": ""},
                {"id": 3902, "spk": "许桃花", "to": "宋钱", "cue": ""},
                {"id": 3999, "spk": "x", "to": "y", "cue": ""},
            ]
        }
        subset = [
            TranslationLine(index=3900, start_ms=0, end_ms=900, text="你好"),
            TranslationLine(index=3902, start_ms=1000, end_ms=1900, text="走吧"),
        ]
        cues = adapter.analyze_visual_cues(subset, "Vietnamese", video_refs=None)
        assert [cue.line_no for cue in cues] == [3900, 3902]


@dataclass
class _Ctx:
    target_lang: str = "Vietnamese"
    source_lang: str = "zh"
    overview: str = "x"
    characters: str = "宋钱: nam chính"
    glossary: str = ""
    visual_cues: str = ""
    enable_tags: bool = False
    include_desc: bool = True


@dataclass
class _Interval:
    start_sec: float = 0.0
    end_sec: float = 0.0


@dataclass
class _Event:
    index: int
    text: str
    interval: _Interval = field(default_factory=_Interval)


def _flag(report: dict, code: str) -> dict | None:
    return next((f for f in report["quality_flags"] if f["code"] == code), None)


class TestContextDiagnosticsCoverageFlag:
    @staticmethod
    def _report(covered: int, total: int = 400) -> dict:
        events = [_Event(i, f"宋钱{i}") for i in range(1, total + 1)]
        ctx = _Ctx(visual_cues=_cues_json(list(range(1, covered + 1))))
        return build_context_diagnostics(context=ctx, source_events=events)

    def test_real_case_27_percent_is_error(self) -> None:
        flag = _flag(self._report(covered=108), "cues_low_coverage")
        assert flag is not None
        assert flag["severity"] == "error"

    def test_half_coverage_is_warning(self) -> None:
        flag = _flag(self._report(covered=200), "cues_low_coverage")
        assert flag is not None
        assert flag["severity"] == "warning"

    def test_good_coverage_is_silent(self) -> None:
        assert _flag(self._report(covered=320), "cues_low_coverage") is None

    def test_dead_zone_is_named_in_message(self) -> None:
        flag = _flag(self._report(covered=108), "cues_low_coverage")
        assert flag is not None
        assert "109-400" in flag["message"]


class TestCueDeadZones:
    def test_even_sparse_cues_are_not_a_dead_zone(self) -> None:
        every_fifth = list(range(1, 1001, 5))  # 20% đều khắp
        assert cue_dead_zones(every_fifth, 1000) == []

    def test_blank_stretch_is_merged_into_one_zone(self) -> None:
        cues = list(range(1, 301)) + list(range(701, 1001))
        assert cue_dead_zones(cues, 1000) == [(301, 700)]

    def test_short_film_has_no_zones(self) -> None:
        assert cue_dead_zones([], CUES_DEAD_ZONE_WINDOW - 1) == []


class TestAddressCueCoverage:
    def test_counts_only_dialogue_pairs(self) -> None:
        cues = json.dumps(
            [
                {"id": 1, "spk": "A", "to": "B"},
                {"id": 2, "spk": "A", "to": "A"},  # độc thoại — không đo được xưng hô
                {"id": 3, "spk": "A", "to": ""},
                {"id": 9, "spk": "A", "to": "B"},  # ngoài bản dịch
            ]
        )
        assert address_cue_coverage(cues, {1, 2, 3, 4}) == 0.25

    def test_empty_inputs(self) -> None:
        assert address_cue_coverage("", {1, 2}) == 0.0
        assert address_cue_coverage(_cues_json([1]), set()) == 0.0


class TestSessionStoreSavesOnlyCues:
    def test_save_visual_cues_keeps_the_rest_of_the_analysis(
        self, tmp_path: Path
    ) -> None:
        store = SqliteTranslationSessionStore(tmp_path / "s.db")
        store.save_analysis(
            "video.mp4",
            characters="宋钱",
            overview="tóm tắt",
            source_lang="zh",
            target_lang="Vietnamese",
            input_hash="h1",
            glossary="g",
            visual_cues="[]",
        )
        store.save_visual_cues("video.mp4", _cues_json([1, 2]))
        session = store.get("video.mp4")
        assert session is not None
        assert len(deserialize_visual_cues(session.analysis_visual_cues)) == 2
        assert session.analysis_characters == "宋钱"
        assert session.has_valid_analysis("h1")
        store.close()

    def test_empty_key_is_rejected(self, tmp_path: Path) -> None:
        store = SqliteTranslationSessionStore(tmp_path / "s.db")
        with pytest.raises(ValueError):
            store.save_visual_cues("", "[]")
        store.close()
