"""Tests cho pipeline dịch phụ đề (use case + adapter helpers).

Bao phủ:
    * Điều phối đa giai đoạn: chaining đúng thứ tự, giữ uid/timing.
    * Ghép văn bản dịch + nhãn người nói/mô tả.
    * Xử lý đầu vào rỗng và huỷ giữa chừng.
    * Hành vi adapter khi không khả dụng (thiếu genai/API key).
    * Các hàm thuần của adapter: sanitize JSON, validate batch, context window.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _BatchPartialError,
    _sanitize_json_text,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    MIN_CHAR_BUDGET,
    readable_syllable_budget,
    syllable_budget_to_chars,
)


def _ngan_sach(khung_s: float) -> int:
    """[v3.23.238] Ngân sách dịch: âm tiết -> ký tự (xem test_syllable_budget_v32638)."""
    return max(
        MIN_CHAR_BUDGET,
        syllable_budget_to_chars(readable_syllable_budget(khung_s)),
    )


def _make_events(texts: list[str]) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i + 1, text=text, interval=TimeInterval(float(i), float(i) + 1.0))
        for i, text in enumerate(texts)
    ]


class _FakeTranslator:
    """Translator giả lập: thêm hậu tố theo giai đoạn, ghi lại các lần gọi."""

    def __init__(self, *, assign_speaker: bool = False) -> None:
        self.calls: list[TranslationStageKind] = []
        self._assign_speaker = assign_speaker

    def is_available(self) -> bool:
        return True

    def translate_stage(
        self, *, stage, context, source_lines, input_lines,
        has_prior_translation, progress_cb=None, cancel_cb=None,
        video_refs=None, attach_video=False,
    ) -> list[TranslationLine]:
        self.calls.append(stage.kind)
        out: list[TranslationLine] = []
        for line in input_lines:
            speaker = "A" if (self._assign_speaker and stage.kind is TranslationStageKind.LITERAL) else line.speaker
            out.append(
                TranslationLine(
                    index=line.index, start_ms=line.start_ms, end_ms=line.end_ms,
                    text=f"{line.text}+{stage.kind.value}", speaker=speaker,
                    description=line.description,
                )
            )
        if progress_cb is not None:
            progress_cb(1.0)
        return out


# ── Điều phối use case ───────────────────────────────────────────────────────

class TestTranslateUseCaseOrchestration:
    def test_chains_stages_in_order(self) -> None:
        translator = _FakeTranslator()
        use_case = TranslateSubtitlesUseCase(translator)
        events = _make_events(["a", "b"])
        stages = [
            TranslationStageConfig(TranslationStageKind.PREPROCESS, "m"),
            TranslationStageConfig(TranslationStageKind.LITERAL, "m"),
            TranslationStageConfig(TranslationStageKind.LOCALIZE, "m"),
        ]
        resp = use_case.execute(
            TranslateSubtitlesRequest(events, stages, TranslationContext("Vietnamese"))
        )
        assert translator.calls == [
            TranslationStageKind.PREPROCESS,
            TranslationStageKind.LITERAL,
            TranslationStageKind.LOCALIZE,
        ]
        # Hậu tố tích luỹ đúng thứ tự
        assert resp.events[0].text == "a+preprocess+literal+localize"

    def test_preserves_uid_and_timing(self) -> None:
        translator = _FakeTranslator()
        use_case = TranslateSubtitlesUseCase(translator)
        events = _make_events(["x", "y"])
        uids = [e.uid for e in events]
        stages = [TranslationStageConfig(TranslationStageKind.LITERAL, "m")]
        resp = use_case.execute(
            TranslateSubtitlesRequest(events, stages, TranslationContext("Vietnamese"))
        )
        assert [e.uid for e in resp.events] == uids
        assert resp.events[0].start_sec == 0.0 and resp.events[0].end_sec == 1.0
        assert resp.events[1].start_sec == 1.0 and resp.events[1].end_sec == 2.0

    def test_composes_speaker_prefix(self) -> None:
        translator = _FakeTranslator(assign_speaker=True)
        use_case = TranslateSubtitlesUseCase(translator)
        events = _make_events(["hello"])
        stages = [TranslationStageConfig(TranslationStageKind.LITERAL, "m")]
        resp = use_case.execute(
            TranslateSubtitlesRequest(events, stages, TranslationContext("Vietnamese", enable_tags=True))
        )
        assert resp.events[0].text.startswith("[A:]")

    def test_empty_events_raises(self) -> None:
        use_case = TranslateSubtitlesUseCase(_FakeTranslator())
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(
                TranslateSubtitlesRequest([], [TranslationStageConfig(TranslationStageKind.LITERAL, "m")], TranslationContext("Vietnamese"))
            )

    def test_no_stages_raises(self) -> None:
        use_case = TranslateSubtitlesUseCase(_FakeTranslator())
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(
                TranslateSubtitlesRequest(_make_events(["a"]), [], TranslationContext("Vietnamese"))
            )

    def test_cancellation_stops_before_stage(self) -> None:
        translator = _FakeTranslator()
        use_case = TranslateSubtitlesUseCase(translator)
        events = _make_events(["a"])
        stages = [TranslationStageConfig(TranslationStageKind.LITERAL, "m")]
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(
                TranslateSubtitlesRequest(events, stages, TranslationContext("Vietnamese")),
                cancel_cb=lambda: True,
            )
        assert translator.calls == []  # huỷ trước khi gọi stage đầu

    def test_progress_reaches_one(self) -> None:
        translator = _FakeTranslator()
        use_case = TranslateSubtitlesUseCase(translator)
        events = _make_events(["a", "b"])
        stages = [
            TranslationStageConfig(TranslationStageKind.LITERAL, "m"),
            TranslationStageConfig(TranslationStageKind.STYLE, "m"),
        ]
        progress: list[float] = []
        use_case.execute(
            TranslateSubtitlesRequest(events, stages, TranslationContext("Vietnamese")),
            progress_cb=lambda p, _label: progress.append(p),
        )
        assert progress[-1] == 1.0
        assert all(0.0 <= p <= 1.0 for p in progress)

    def test_preprocess_does_not_set_prior_translation(self) -> None:
        """Sau PREPROCESS, has_prior phải vẫn False khi vào LITERAL."""
        seen_prior: list[bool] = []

        class _RecordingTranslator(_FakeTranslator):
            def translate_stage(self, *, has_prior_translation, **kwargs):
                seen_prior.append(has_prior_translation)
                return super().translate_stage(has_prior_translation=has_prior_translation, **kwargs)

        use_case = TranslateSubtitlesUseCase(_RecordingTranslator())
        events = _make_events(["a"])
        stages = [
            TranslationStageConfig(TranslationStageKind.PREPROCESS, "m"),
            TranslationStageConfig(TranslationStageKind.LITERAL, "m"),
            TranslationStageConfig(TranslationStageKind.STYLE, "m"),
        ]
        use_case.execute(
            TranslateSubtitlesRequest(events, stages, TranslationContext("Vietnamese"))
        )
        # PREPROCESS: False, LITERAL: False (preprocess không bật prior), STYLE: True
        assert seen_prior == [False, False, True]


# ── Hàm thuần của adapter ────────────────────────────────────────────────────

class TestAdapterPureFunctions:
    def test_sanitize_json_strips_fences(self) -> None:
        assert _sanitize_json_text('```json\n{"a":1}\n```') == '{"a":1}'
        assert _sanitize_json_text('  {"a":1}  ') == '{"a":1}'

    def test_line_to_payload_includes_optional_fields(self) -> None:
        line = TranslationLine(1, 0, 1000, "hi", speaker="Bob", description="laughs")
        payload = GeminiSubtitleTranslator._line_to_payload(line)
        # [v3.23.238] payload kèm max_chars theo ngân sách ÂM TIẾT (quy sang ký tự) —
        # mô hình theo KÝ TỰ cũ đã bị bác bỏ bằng dữ liệu (R²=0.07 vs 0.89).
        assert payload == {
            "line_no": 1, "text": "hi", "speaker": "Bob", "description": "laughs",
            "max_chars": _ngan_sach(1.0),
        }

    def test_line_to_payload_omits_empty_optional(self) -> None:
        line = TranslationLine(2, 0, 1000, "hi")
        payload = GeminiSubtitleTranslator._line_to_payload(line)
        assert payload == {"line_no": 2, "text": "hi", "max_chars": _ngan_sach(1.0)}

    def test_context_window_zero_size(self) -> None:
        lines = [TranslationLine(i, 0, 1, str(i)) for i in range(1, 6)]
        before, after = GeminiSubtitleTranslator._context_window(lines, start_idx=2, batch_len=1, ctx_size=0)
        assert before == [] and after == []

    def test_context_window_clamps_bounds(self) -> None:
        lines = [TranslationLine(i, 0, 1, str(i)) for i in range(1, 6)]
        before, after = GeminiSubtitleTranslator._context_window(lines, start_idx=0, batch_len=2, ctx_size=10)
        assert before == []  # không có gì trước index 0
        assert len(after) == 3  # phần tử 2,3,4 (sau batch [0:2])

    def test_validate_batch_count_mismatch(self) -> None:
        # [v3.23.18] 1 item cho batch 2 → renumber + vá đuôi (PartialError) thay vì
        # ValidationError (model giữ đúng thứ tự, chỉ thiếu dòng cuối).
        adapter = GeminiSubtitleTranslator(api_key="", retry_count=1)
        with pytest.raises(_BatchPartialError):
            adapter._validate_batch({"subtitles": [{"line_no": 1, "text": "a"}]}, expected_count=2, start_idx=0)

    def test_validate_batch_line_no_mismatch(self) -> None:
        # [v3.23.18] 1 item line_no=5 cho batch 1 dòng → số lượng khớp → chấp nhận,
        # renumber theo vị trí (không raise). Model đánh số lệch nhưng đủ.
        adapter = GeminiSubtitleTranslator(api_key="", retry_count=1)
        payload = {"subtitles": [{"line_no": 5, "text": "a"}]}
        adapter._validate_batch(payload, expected_count=1, start_idx=0)
        assert payload["subtitles"][0]["line_no"] == 1  # đã renumber

    def test_validate_batch_ok(self) -> None:
        adapter = GeminiSubtitleTranslator(api_key="", retry_count=1)
        adapter._validate_batch(
            {"subtitles": [{"line_no": 1, "text": "a"}, {"line_no": 2, "text": "b"}]},
            expected_count=2, start_idx=0,
        )

    def test_merge_ai_items_preprocess_keeps_speaker(self) -> None:
        batch = [TranslationLine(1, 0, 1000, "old", speaker="X", description="d")]
        ai_items = [{"line_no": 1, "text": "fixed"}]
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, ai_items, is_preprocess=True)
        assert merged[0].text == "fixed"
        assert merged[0].speaker == "X"  # preprocess không đổi speaker

    def test_merge_ai_items_translation_updates_speaker(self) -> None:
        batch = [TranslationLine(1, 0, 1000, "old")]
        ai_items = [{"line_no": 1, "text": "moi", "speaker": "Bob", "description": "cuoi"}]
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, ai_items, is_preprocess=False)
        assert merged[0].text == "moi"
        assert merged[0].speaker == "Bob"
        assert merged[0].description == "cuoi"

    def test_merge_ai_items_fallback_on_missing(self) -> None:
        batch = [TranslationLine(1, 0, 1000, "original")]
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, [], is_preprocess=False)
        assert merged[0].text == "original"  # thiếu AI item → giữ gốc


# ── Khả dụng của adapter ─────────────────────────────────────────────────────

class TestAdapterAvailability:
    def test_unavailable_without_api_key(self) -> None:
        adapter = GeminiSubtitleTranslator(api_key="", retry_count=3)
        # Không có genai trong môi trường test → is_available False bất kể key
        assert adapter.is_available() is False

    def test_translate_stage_raises_when_unavailable(self) -> None:
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslatorUnavailableError,
        )

        adapter = GeminiSubtitleTranslator(api_key="", retry_count=1)
        with pytest.raises(TranslatorUnavailableError):
            adapter.translate_stage(
                stage=TranslationStageConfig(TranslationStageKind.LITERAL, "m"),
                context=TranslationContext("Vietnamese"),
                source_lines=[TranslationLine(1, 0, 1, "a")],
                input_lines=[TranslationLine(1, 0, 1, "a")],
                has_prior_translation=False,
            )
